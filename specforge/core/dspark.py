# coding=utf-8
"""DSpark online training wrapper: DFlash backbone + Markov / L1 / confidence losses.

Ported from TorchSpec PR #129 (``torchspec/models/dspark.py``). Reuses SpecForge's
:class:`OnlineDFlashModel` anchor sampling, block-causal mask construction, and
MASK-token noise stream verbatim (via ``super()``), then layers on the DSpark
training objective:

  - Markov-biased draft logits (teacher-forced previous token).
  - Cross-entropy against the ground-truth next tokens (hard labels).
  - L1 distribution distillation: ``|softmax(draft) - softmax(target)|`` where the
    target distribution is the frozen LM head applied to the *target's* final
    hidden state at the aligned position (requires ``last_hidden_states``).
  - Confidence head BCE against the empirical per-token accept rate.

Combined: ``ce_alpha*ce + l1_alpha*l1 + confidence_alpha*confidence``.

Loss formulation adapted from DeepSeek's DeepSpec (``deepspec/modeling/dspark/loss.py``,
MIT), including its pooled global-mean reduction: local numerators over a
cross-rank all-reduced denominator, scaled by world_size to cancel FSDP's mean
gradient reduction.

Key SpecForge differences vs TorchSpec (see port notes in the PR):
  - SpecForge's :class:`OnlineDFlashModel.forward` returns ``(loss, accuracy)``;
    DSpark needs the per-component losses, so this forward returns a 6-tuple
    ``(loss, accuracy, loss_per_position, acc_per_position, count_per_position,
    loss_components)``. ``train_dspark.py`` consumes the extra elements.
  - The target ``lm_head`` is a frozen ``nn.Linear`` module on the wrapper
    (``self.lm_head``); the L1 path uses ``self.lm_head.weight`` for ``F.linear``.
  - The fused multi-layer context feature (``hidden_states``) is produced upstream
    by ``generate_dflash_data`` and fed straight to the draft as ``target_hidden``.
"""

import math
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.dflash import (
    FLEX_ATTENTION_AVAILABLE,
    OnlineDFlashModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.legacy.dspark import DSparkDraftModel
from specforge.sampling import processed_log_probs
from specforge.bv_loss import bv_beta, bv_loss_terms, validate_bv_options
from specforge.predictive_auxiliary import (
    auxiliary_scale, predecessor_source_weights, reference_calibration_terms,
)


def _target_fusion_source_metrics(normalized_fusion):
    """Scalar diagnostics for [layer, source] or [layer, group, source].

    Call under no_grad. Preserve existing source keys as group means and expose
    individual groups separately. All DP ranks produce the same key shapes.
    """
    if normalized_fusion.ndim not in (2, 3):
        raise ValueError("Expected fusion weights [layer, source] or [layer, group, source]")
    metrics = {}
    for layer, weights in enumerate(normalized_fusion):
        source_weights = weights
        if weights.ndim == 2:
            source_weights = weights.mean(dim=0)
            for group, group_weights in enumerate(weights):
                for source, weight in enumerate(group_weights):
                    metrics[f"target_fusion_d{layer + 1}_g{group + 1}_s{source + 1}"] = weight
        for source, weight in enumerate(source_weights):
            metrics[f"target_fusion_d{layer + 1}_s{source + 1}"] = weight
    return metrics


class _ForwardValueReferenceGradient(torch.autograd.Function):
    """Use ``value`` in forward while routing its gradient to ``reference``.

    This is the allocation-free form of
    ``reference + (value - reference).detach()``.  The latter creates two
    additional full-vocabulary tensors, which is prohibitive for DSpark's
    block-parallel training logits.
    """

    @staticmethod
    def forward(ctx, reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        del ctx, reference
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        del ctx
        return grad_output, None


def _forward_value_reference_gradient(
    reference: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    return _ForwardValueReferenceGradient.apply(reference, value.detach())


def _distributed_any(value: torch.Tensor) -> bool:
    """Return whether ``value`` is true on any training rank.

    Auxiliary verifier passes must have identical control flow on every FSDP
    rank.  A rank-local ``Tensor.any()`` is not sufficient: one rank may have
    no eligible rows while another rank enters an additional target forward,
    leaving the following collective operations permanently out of order.
    """
    result = value.any().to(dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return bool(result.item())


def _carh_predecessor_context_kwargs(
    markov_head: nn.Module, block_hidden: torch.Tensor, depth: int
) -> dict:
    """Pass the previous slot's backbone state only for the new CARH modes.

    ``block_hidden`` has the draft-depth axis immediately before hidden width.
    The first slot never reads a state from another block.
    """
    if (
        depth <= 0
        or getattr(markov_head, "predecessor_context_mode", "none") == "none"
    ):
        return {}
    return {"previous_hidden_states": block_hidden[..., depth - 1, :]}


class OnlineDSparkModel(OnlineDFlashModel):
    """DSpark online training wrapper (DFlash backbone + Markov/L1/confidence heads)."""

    def __init__(
        self,
        draft_model: DSparkDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 7,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = 4.0,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        bv_loss_alpha: float = 0.0,
        bv_temperature: float = 1.0,
        bv_anneal_ratio: float = 0.5,
        bv_block_chunk_size: int = 8,
        offline_acceptance_objective: str = "none",
        offline_acceptance_greedy_weight: float = 0.35,
        offline_acceptance_stochastic_weight: float = 0.65,
        offline_acceptance_temperature: float = 1.0,
        offline_acceptance_lk_eta: float = 1.0,
        offline_acceptance_dpace_rho: float = 0.5,
        offline_acceptance_cold_start_ratio: float = 0.20,
        offline_acceptance_transition_ratio: float = 0.10,
        confidence_head_alpha: float = 1.0,
        confidence_target_mode: str = "l1-overlap",
        confidence_detach_backbone: bool = False,
        carh_predecessor_diagnostics_interval: int = 0,
        carh_sampled_prefix_memory_start_ratio: float = 0.25,
        carh_sampled_prefix_memory_ramp_ratio: float = 0.10,
        recall_correction_alpha: float = 0.0,
        recall_partition_top_k: int = 16,
        recall_correction_loss_budget: float = 0.08,
        pace_mode: str = "none",
        pace_alpha: float = 0.5,
        pace_hybrid_beta: float = 0.5,
        pace_apply_to: str = "ce-l1",
        pace_blend_max: float = 1.0,
        pace_warmup_ratio: float = 0.0,
        pace_ramp_ratio: float = 0.0,
        pace_decay_start_ratio: float = 1.0,
        pace_blend_final: Optional[float] = None,
        pace_residual_beta: float = 0.5,
        pace_residual_min: float = 0.9,
        pace_residual_max: float = 1.1,
        prefix_credit_mode: str = "none",
        prefix_credit_alpha: float = 0.0,
        prefix_credit_warmup_ratio: float = 0.10,
        prefix_credit_ramp_ratio: float = 0.15,
        prefix_credit_backbone_grad_scale: float = 0.2,
        shallow_frc_alpha: float = 0.0,
        shallow_frc_max_depth: int = 3,
        shallow_frc_margin: float = 0.0,
        shallow_frc_temperature: float = 1.0,
        dfap_alpha: float = 0.0,
        dfap_min_depth: int = 4,
        dfap_margin: float = 0.5,
        dfap_temperature: float = 1.0,
        state_credit_partition: str = "overlap",
        conv_gate_loss_alpha: float = 0.0,
        conv_source_semantic_alpha: float = 0.0,
        carh_reference_calibration_alpha: float = 0.0,
        predictive_aux_warmup_ratio: float = 0.10,
        predictive_aux_ramp_ratio: float = 0.15,
        source_semantic_codebook: Optional[torch.Tensor] = None,
        block_summary_counterfactual_alpha: float = 0.0,
        transition_credit_alpha: float = 0.0,
        transition_credit_max_depth: int = 3,
        transition_credit_margin: float = 0.0,
        transition_credit_temperature: float = 1.0,
        transition_credit_loss_budget: float = 0.10,
        transition2_margin_alpha: float = 0.0,
        transition2_margin_floor: float = 0.5,
        transition2_margin_temperature: float = 0.25,
        transition2_margin_loss_budget: float = 0.10,
        prefix_bottleneck_alpha: float = 0.0,
        prefix_bottleneck_max_depth: int = 3,
        prefix_bottleneck_margin_floor: float = 0.5,
        prefix_bottleneck_softmin_temperature: float = 0.25,
        prefix_bottleneck_loss_temperature: float = 0.25,
        prefix_bottleneck_loss_budget: float = 0.10,
        deep_survival_guard_alpha: float = 0.0,
        deep_survival_guard_min_prefix: int = 5,
        deep_survival_guard_start_depth: int = 4,
        deep_survival_guard_margin_floor: float = 0.3,
        deep_survival_guard_temperature: float = 0.25,
        deep_survival_guard_loss_budget: float = 0.10,
        carh_recoverability_top_k: int = 0,
        carh_recovery_loss_alpha: float = 0.0,
        carh_preservation_loss_alpha: float = 0.0,
        carh_noop_loss_alpha: float = 0.0,
        carh_preservation_margin: float = 0.0,
        carh_preservation_temperature: float = 0.25,
        carh_recoverability_loss_budget: float = 0.10,
        carh_gate_calibration_alpha: float = 0.0,
        carh_gate_noop_alpha: float = 0.0,
        carh_gate_margin_threshold: float = 0.0,
        carh_gate_calibration_warmup_ratio: float = 0.10,
        selector_loss_alpha: float = 0.0,
        selector_rollout_ratio: float = 0.0,
        selector_training_mode: str = "legacy",
        selector_distill_alpha: float = 0.0,
        selector_preservation_alpha: float = 0.0,
        selector_distill_temperature: float = 1.0,
        selector_advantage_margin: float = 0.0,
        selector_teacher_warmup_ratio: float = 0.10,
        selector_distill_start_ratio: float = 0.30,
        selector_consolidation_ratio: float = 0.15,
        selector_loss_budget: float = 0.10,
        selector_margin_threshold: float = 0.0,
        selector_tree_loss_alpha: float = 0.0,
        selector_tree_branch_width: int = 2,
        selector_tree_margin: float = 0.2,
        selector_tree_temperature: float = 0.25,
        on_policy_survival_alpha: float = 0.0,
        on_policy_full_alpha: float = 0.0,
        on_policy_first_rejection_max_depth: int = 2,
        on_policy_boundary_max_depth: int = 0,
        on_policy_deep_alpha: float = 0.0,
        on_policy_deep_start_depth: int = 4,
        on_policy_margin_floor: float = 0.3,
        on_policy_temperature: float = 0.25,
        on_policy_rollout_temperatures: Sequence[float] = (0.0,),
        on_policy_rollout_temperature_probs: Sequence[float] = (1.0,),
        on_policy_anchor_sampling: str = "first",
        on_policy_hazard_power: float = 1.0,
        on_policy_flatness_power_max: float = 1.0,
        on_policy_flatness_warmup_ratio: float = 0.3,
        on_policy_uniform_exploration: float = 0.1,
        dialogue_phase_masses: Sequence[float] = (0.15, 0.20, 0.30, 0.20, 0.15),
        dialogue_boundary_boost: float = 1.5,
        dialogue_script_transition_boost: float = 1.5,
        dialogue_occupancy_diagnostics: bool = False,
        dialogue_num_sources: int = 1,
        dialogue_token_class_lookup: Optional[torch.Tensor] = None,
        dialogue_prefix_phase_weighting: bool = False,
        dialogue_t2cm_boundary_weighting: bool = False,
        dialogue_dsg_adaptive_margin: bool = False,
        dialogue_dsg_late_margin_bonus: float = 0.10,
        dialogue_dsg_boundary_margin_bonus: float = 0.05,
        on_policy_distributional_top_k: int = 0,
        on_policy_distributional_min_top_k: int = 0,
        on_policy_distributional_mass_threshold: float = 0.0,
        on_policy_distributional_alpha: float = 0.0,
        on_policy_distributional_deep_alpha: float = 0.0,
        on_policy_distributional_cold_start_ratio: float = 0.15,
        on_policy_rejection_aligned_alpha: float = 0.0,
        on_policy_rejection_aligned_survival_blend: float = 0.5,
        on_policy_temperature_exclusive_routing: bool = False,
        on_policy_mixed_kl_alpha: float = 0.0,
        on_policy_mixed_kl_accepted_weight: float = 1.0,
        on_policy_mixed_kl_rejected_weight: float = 1.0,
        on_policy_mixed_kl_rejection_decay: float = 0.8,
        on_policy_clipped_rkl_alpha: float = 0.0,
        on_policy_clipped_rkl_clip: float = 0.01,
        on_policy_clipped_rkl_temperature_weights: Sequence[float] = (1.0,),
        on_policy_target_distribution_temperature_floor: float = 0.0,
        on_policy_target_margin_alpha: float = 0.0,
        on_policy_target_margin_scale: float = 0.5,
        on_policy_target_margin_offset: float = 0.05,
        on_policy_target_margin_min: float = 0.05,
        on_policy_target_margin_max: float = 1.0,
        on_policy_target_margin_max_depth: int = 3,
        on_policy_deployment_top_k: int = 0,
        on_policy_deployment_top_p: float = 1.0,
        on_policy_credit_partition: str = "legacy",
        on_policy_middle_max_depth: int = 4,
        on_policy_middle_alpha: float = 0.0,
        on_policy_temperature_hazard_credit: bool = False,
        on_policy_hazard_ema_decay: float = 0.99,
        on_policy_hazard_weight_min: float = 0.5,
        on_policy_hazard_weight_max: float = 2.0,
        on_policy_marginal_value_credit: bool = False,
        on_policy_marginal_value_temperature: float = 0.5,
        on_policy_marginal_value_weight_min: float = 0.5,
        on_policy_marginal_value_weight_max: float = 2.0,
        on_policy_language_hazard_credit: bool = False,
        on_policy_language_han_threshold: float = 0.5,
        on_policy_language_hazard_ema_decay: float = 0.98,
        on_policy_language_hazard_weight_min: float = 0.75,
        on_policy_language_hazard_weight_max: float = 1.5,
        on_policy_mix_ratio_max: float = 0.5,
        on_policy_start_step: Optional[int] = None,
        on_policy_ramp_steps: int = 2000,
        on_policy_interval: int = 4,
        on_policy_loss_budget: float = 0.10,
        on_policy_preservation_alpha: float = 0.0,
        on_policy_regression_alpha: float = 0.0,
        on_policy_regression_margin: float = 0.0,
        on_policy_greedy_preservation_alpha: float = 0.0,
        on_policy_greedy_preservation_margin_floor: float = 0.5,
        on_policy_greedy_preservation_temperature: float = 0.25,
        on_policy_repair_value: bool = False,
        on_policy_repair_value_horizon: int = 2,
        on_policy_reset_replay_alpha: float = 0.0,
        on_policy_reset_replay_horizon: int = 3,
        on_policy_reset_replay_max_rejection_depth: int = 4,
        on_policy_reset_replay_start_ratio: float = 0.25,
        on_policy_reset_replay_ramp_ratio: float = 0.15,
        on_policy_reset_replay_loss_budget: float = 0.05,
        on_policy_reset_replay_advantage_threshold: float = 0.0,
        on_policy_reset_replay_value_clip: float = 3.0,
        on_policy_preference_alpha: float = 0.0,
        on_policy_preference_value_gap: float = 0.5,
        on_policy_preference_temperature: float = 0.25,
        on_policy_preference_loss_budget: float = 0.03,
        on_policy_pareto_credit: bool = False,
        on_policy_pareto_ema_decay: float = 0.95,
        on_policy_pareto_temperature: float = 0.25,
        on_policy_pareto_weight_min: float = 0.5,
        on_policy_pareto_weight_max: float = 2.0,
        parallel_refiner_hazard_alpha: float = 0.0,
        parallel_refiner_recovery_alpha: float = 0.0,
        parallel_refiner_preservation_alpha: float = 0.0,
        parallel_refiner_margin_floor: float = 0.3,
        parallel_refiner_temperature: float = 0.25,
        parallel_refiner_loss_budget: float = 0.05,
        refiner_advantage_mode: str = "none",
        refiner_advantage_threshold: float = 0.0,
        refiner_advantage_gate_alpha: float = 0.0,
        refiner_advantage_regression_alpha: float = 0.0,
        refiner_advantage_distill_alpha: float = 0.0,
        refiner_advantage_preservation_alpha: float = 0.0,
        refiner_advantage_greedy_preservation_alpha: float = 0.0,
        refiner_advantage_greedy_margin_floor: float = 0.5,
        refiner_advantage_gate_distill_min_probability: float = 0.0,
        refiner_advantage_distill_top_k: int = 32,
        refiner_advantage_distill_temperature: float = 1.0,
        refiner_advantage_value_clip: float = 7.0,
        refiner_advantage_distill_start_ratio: float = 0.20,
        refiner_advantage_consolidation_ratio: float = 0.15,
        refiner_advantage_loss_budget: float = 0.05,
        vat_enabled: bool = False,
        vat_hard_loss_alpha: float = 1.0,
        vat_soft_loss_alpha: float = 1.0,
        vat_verification_head_alpha: float = 1.0,
        vat_post_rejection_decay_gamma: float = 4.0,
        vat_simulation_temperature: float = 0.0,
        branch_value_mode: str = "none",
        branch_value_top_m: int = 4,
        branch_value_horizon: int = 3,
        branch_value_alpha: float = 0.0,
        branch_value_temperature: float = 0.5,
        branch_value_loss_budget: float = 0.05,
        branch_value_warmup_ratio: float = 0.10,
        branch_value_consolidation_ratio: float = 0.15,
        elastic_horizon_enabled: bool = False,
        elastic_short_horizon: int = 7,
        elastic_long_horizon: int = 10,
        elastic_warmup_ratio: float = 0.10,
        elastic_late_ratio: float = 0.70,
        elastic_consolidation_ratio: float = 0.15,
        elastic_middle_long_prob: float = 0.30,
        elastic_late_long_prob: float = 0.50,
        elastic_pair_probability: float = 0.25,
        elastic_projective_num_anchors: int = 32,
        elastic_projective_alpha: float = 0.0025,
        elastic_projective_top_k: int = 64,
        elastic_overlap_gap: float = 0.005,
        elastic_margin_gap: float = 0.02,
        elastic_loss_budget: float = 0.05,
    ):
        sampled_prefix_memory_enabled = (
            int(getattr(getattr(draft_model, "markov_head", None),
                        "sampled_prefix_memory_rank", 0)) > 0
        )
        if sampled_prefix_memory_enabled:
            unsupported = {
                "on_policy_repair_value": bool(on_policy_repair_value),
                "on_policy_reset_replay_alpha": on_policy_reset_replay_alpha > 0,
                "on_policy_preference_alpha": on_policy_preference_alpha > 0,
                "branch_value_mode": branch_value_mode != "none",
                "vat_enabled": bool(vat_enabled),
            }
            active = sorted(name for name, enabled in unsupported.items() if enabled)
            if active:
                raise ValueError(
                    "sampled-prefix CARH has no state-aware implementation for "
                    f"auxiliary paths: {active}"
                )
        if not 0 <= carh_sampled_prefix_memory_start_ratio <= 1:
            raise ValueError("sampled-prefix memory start ratio must be in [0, 1]")
        if carh_sampled_prefix_memory_ramp_ratio < 0:
            raise ValueError("sampled-prefix memory ramp ratio must be non-negative")
        if (carh_sampled_prefix_memory_start_ratio
                + carh_sampled_prefix_memory_ramp_ratio > 1):
            raise ValueError("sampled-prefix memory start+ramp ratios must be <= 1")
        self.carh_sampled_prefix_memory_start_ratio = float(
            carh_sampled_prefix_memory_start_ratio
        )
        self.carh_sampled_prefix_memory_ramp_ratio = float(
            carh_sampled_prefix_memory_ramp_ratio
        )
        if getattr(getattr(draft_model, "markov_head", None), "predecessor_count", 1) >= 2:
            # This minimal ablation supports the teacher-forced block path.
            # Legacy auxiliary rollouts do not pass the lag-2 sampled token.
            unsupported = [
                name for name, value in locals().items()
                if name.startswith(("on_policy_", "selector_", "parallel_refiner_"))
                and name.endswith("_alpha") and value != 0
            ]
            for name, enabled in {
                # CLI advantage weights have nonzero defaults even when the
                # feature is off. Its mode, not dormant weights, enables it.
                "refiner_advantage_mode": refiner_advantage_mode != "none",
                "pace_mode=rollout-residual": pace_mode == "rollout-residual",
                "prefix_credit_mode": prefix_credit_mode != "none",
                "shallow_frc_alpha": shallow_frc_alpha != 0,
                "dfap_alpha": dfap_alpha != 0,
                "transition_credit_alpha": transition_credit_alpha != 0,
                "transition2_margin_alpha": transition2_margin_alpha != 0,
                "prefix_bottleneck_alpha": prefix_bottleneck_alpha != 0,
                "deep_survival_guard_alpha": deep_survival_guard_alpha != 0,
                "branch_value_mode": branch_value_mode != "none",
                "confidence_target_mode": confidence_target_mode != "l1-overlap",
                "selector_rollout_ratio": selector_rollout_ratio != 0,
                "vat_enabled": vat_enabled,
            }.items():
                if enabled:
                    unsupported.append(name)
            if unsupported:
                raise ValueError(
                    "Multi-predecessor CARH minimal offline experiment does not support auxiliary "
                    f"rollout modes: {sorted(unsupported)}"
                )
        # Reuse DFlash anchor/mask/noise machinery. loss_type="dflash" is only a
        # placeholder to satisfy the parent validator — DSpark overrides forward()
        # entirely and never dispatches on loss_type.
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        validate_bv_options(bv_loss_alpha, bv_temperature, bv_anneal_ratio,
                            bv_block_chunk_size, l1_loss_alpha,
                            offline_acceptance_objective, vat_enabled)
        self.bv_loss_alpha = float(bv_loss_alpha)
        self.bv_temperature = float(bv_temperature)
        self.bv_anneal_ratio = float(bv_anneal_ratio)
        self.bv_block_chunk_size = int(bv_block_chunk_size)
        if (not math.isfinite(conv_source_semantic_alpha) or conv_source_semantic_alpha < 0
                or not math.isfinite(carh_reference_calibration_alpha) or carh_reference_calibration_alpha < 0):
            raise ValueError("Predictive auxiliary weights must be finite and non-negative")
        if not (0 <= predictive_aux_warmup_ratio <= 1 and 0 <= predictive_aux_ramp_ratio <= 1
                and predictive_aux_warmup_ratio + predictive_aux_ramp_ratio <= 1):
            raise ValueError("Invalid predictive auxiliary schedule")
        if conv_source_semantic_alpha > 0 and source_semantic_codebook is None:
            raise ValueError("Source semantic supervision requires a fixed codebook")
        self.conv_source_semantic_alpha = float(conv_source_semantic_alpha)
        self.carh_reference_calibration_alpha = float(carh_reference_calibration_alpha)
        self.predictive_aux_warmup_ratio = float(predictive_aux_warmup_ratio)
        self.predictive_aux_ramp_ratio = float(predictive_aux_ramp_ratio)
        self.register_buffer("source_semantic_codebook", source_semantic_codebook, persistent=False)
        if offline_acceptance_objective not in {"none", "angel-lk-e2e"}:
            raise ValueError(
                "offline_acceptance_objective must be none or angel-lk-e2e"
            )
        if offline_acceptance_greedy_weight < 0:
            raise ValueError("offline acceptance greedy weight must be non-negative")
        if offline_acceptance_stochastic_weight < 0:
            raise ValueError(
                "offline acceptance stochastic weight must be non-negative"
            )
        if (
            offline_acceptance_objective != "none"
            and offline_acceptance_greedy_weight
            + offline_acceptance_stochastic_weight
            <= 0
        ):
            raise ValueError("offline acceptance weights must have positive mass")
        if offline_acceptance_temperature <= 0:
            raise ValueError("offline acceptance temperature must be positive")
        if offline_acceptance_lk_eta < 0:
            raise ValueError("offline acceptance LK eta must be non-negative")
        if not 0.0 <= offline_acceptance_dpace_rho <= 1.0:
            raise ValueError("offline acceptance D-PACE rho must be in [0, 1]")
        if not 0.0 <= offline_acceptance_cold_start_ratio <= 1.0:
            raise ValueError("offline acceptance cold-start ratio must be in [0, 1]")
        if not 0.0 <= offline_acceptance_transition_ratio <= 1.0:
            raise ValueError("offline acceptance transition ratio must be in [0, 1]")
        if (
            offline_acceptance_cold_start_ratio
            + offline_acceptance_transition_ratio
            > 1.0
        ):
            raise ValueError(
                "offline acceptance cold-start + transition ratios must be <= 1"
            )
        self.offline_acceptance_objective = offline_acceptance_objective
        self.offline_acceptance_greedy_weight = float(
            offline_acceptance_greedy_weight
        )
        self.offline_acceptance_stochastic_weight = float(
            offline_acceptance_stochastic_weight
        )
        self.offline_acceptance_temperature = float(
            offline_acceptance_temperature
        )
        self.offline_acceptance_lk_eta = float(offline_acceptance_lk_eta)
        self.offline_acceptance_dpace_rho = float(
            offline_acceptance_dpace_rho
        )
        self.offline_acceptance_cold_start_ratio = float(
            offline_acceptance_cold_start_ratio
        )
        self.offline_acceptance_transition_ratio = float(
            offline_acceptance_transition_ratio
        )
        self.confidence_head_alpha = float(confidence_head_alpha)
        if carh_predecessor_diagnostics_interval < 0:
            raise ValueError("carh_predecessor_diagnostics_interval must be nonnegative")
        self.carh_predecessor_diagnostics_interval = int(carh_predecessor_diagnostics_interval)
        if confidence_target_mode not in {"l1-overlap", "on-policy-survival"}:
            raise ValueError("unknown confidence_target_mode")
        self.confidence_target_mode = confidence_target_mode
        self.confidence_detach_backbone = bool(confidence_detach_backbone)
        vocab_size = int(target_lm_head.weight.shape[0])
        if recall_correction_alpha < 0:
            raise ValueError("recall_correction_alpha must be non-negative")
        if not 1 <= recall_partition_top_k <= vocab_size:
            raise ValueError(
                "recall_partition_top_k must be in [1, vocab_size]"
            )
        if not 0.0 <= recall_correction_loss_budget <= 1.0:
            raise ValueError("recall correction loss budget must be in [0, 1]")
        if recall_correction_alpha > 0 and getattr(
            draft_model, "recall_correction", None
        ) is None:
            raise ValueError(
                "recall_correction_alpha requires --recall-correction-rank"
            )
        self.recall_correction_alpha = float(recall_correction_alpha)
        self.recall_partition_top_k = int(recall_partition_top_k)
        self.recall_correction_loss_budget = float(
            recall_correction_loss_budget
        )
        if pace_mode not in {
            "none",
            "token",
            "overlap",
            "hybrid",
            "rollout-residual",
        }:
            raise ValueError(
                "pace_mode must be one of none/token/overlap/hybrid/"
                "rollout-residual, "
                f"got {pace_mode!r}"
            )
        if not 0.0 <= pace_alpha <= 1.0:
            raise ValueError(f"pace_alpha must be in [0, 1], got {pace_alpha}")
        if not 0.0 <= pace_hybrid_beta <= 1.0:
            raise ValueError(
                "pace_hybrid_beta must be in [0, 1], "
                f"got {pace_hybrid_beta}"
            )
        if pace_apply_to not in {"ce", "ce-l1"}:
            raise ValueError(
                "pace_apply_to must be ce or ce-l1, "
                f"got {pace_apply_to!r}"
            )
        if not 0.0 <= pace_blend_max <= 1.0:
            raise ValueError(
                f"pace_blend_max must be in [0, 1], got {pace_blend_max}"
            )
        if not 0.0 <= pace_warmup_ratio <= 1.0:
            raise ValueError(
                "pace_warmup_ratio must be in [0, 1], "
                f"got {pace_warmup_ratio}"
            )
        if not 0.0 <= pace_ramp_ratio <= 1.0:
            raise ValueError(
                f"pace_ramp_ratio must be in [0, 1], got {pace_ramp_ratio}"
            )
        if pace_warmup_ratio + pace_ramp_ratio > 1.0:
            raise ValueError("PACE warmup + ramp ratios must be <= 1")
        if not 0.0 <= pace_decay_start_ratio <= 1.0:
            raise ValueError(
                "pace_decay_start_ratio must be in [0, 1], "
                f"got {pace_decay_start_ratio}"
            )
        if pace_decay_start_ratio < pace_warmup_ratio + pace_ramp_ratio:
            raise ValueError(
                "PACE decay must start after the warmup and ramp"
            )
        if pace_blend_final is None:
            pace_blend_final = pace_blend_max
        if not 0.0 <= pace_blend_final <= pace_blend_max:
            raise ValueError(
                "pace_blend_final must be in [0, pace_blend_max], "
                f"got {pace_blend_final}"
            )
        if pace_residual_beta < 0:
            raise ValueError("pace_residual_beta must be non-negative")
        if not 0.0 < pace_residual_min <= 1.0 <= pace_residual_max:
            raise ValueError(
                "PACE residual bounds must satisfy 0 < min <= 1 <= max"
            )
        self.pace_mode = pace_mode
        self.pace_alpha = float(pace_alpha)
        self.pace_hybrid_beta = float(pace_hybrid_beta)
        self.pace_apply_to = pace_apply_to
        self.pace_blend_max = float(pace_blend_max)
        self.pace_warmup_ratio = float(pace_warmup_ratio)
        self.pace_ramp_ratio = float(pace_ramp_ratio)
        self.pace_decay_start_ratio = float(pace_decay_start_ratio)
        self.pace_blend_final = float(pace_blend_final)
        self.pace_residual_beta = float(pace_residual_beta)
        self.pace_residual_min = float(pace_residual_min)
        self.pace_residual_max = float(pace_residual_max)
        if prefix_credit_mode not in {"none", "full", "residual-only", "partial"}:
            raise ValueError(
                "prefix_credit_mode must be none/full/residual-only/partial, "
                f"got {prefix_credit_mode!r}"
            )
        if prefix_credit_alpha < 0:
            raise ValueError("prefix_credit_alpha must be non-negative")
        if not 0.0 <= prefix_credit_warmup_ratio <= 1.0:
            raise ValueError("prefix_credit_warmup_ratio must be in [0, 1]")
        if not 0.0 <= prefix_credit_ramp_ratio <= 1.0:
            raise ValueError("prefix_credit_ramp_ratio must be in [0, 1]")
        if prefix_credit_warmup_ratio + prefix_credit_ramp_ratio > 1.0:
            raise ValueError("prefix credit warmup + ramp ratios must be <= 1")
        if prefix_credit_mode != "none" and prefix_credit_alpha == 0:
            raise ValueError(
                "prefix_credit_alpha must be positive when prefix credit is enabled"
            )
        if not 0.0 <= prefix_credit_backbone_grad_scale <= 1.0:
            raise ValueError("prefix_credit_backbone_grad_scale must be in [0, 1]")
        if prefix_credit_mode == "partial" and not (
            0.0 < prefix_credit_backbone_grad_scale < 1.0
        ):
            raise ValueError(
                "partial prefix credit requires backbone_grad_scale strictly in (0, 1)"
            )
        self.prefix_credit_mode = prefix_credit_mode
        self.prefix_credit_alpha = float(prefix_credit_alpha)
        self.prefix_credit_warmup_ratio = float(prefix_credit_warmup_ratio)
        self.prefix_credit_ramp_ratio = float(prefix_credit_ramp_ratio)
        self.prefix_credit_backbone_grad_scale = float(
            prefix_credit_backbone_grad_scale
        )
        if shallow_frc_alpha < 0 or dfap_alpha < 0:
            raise ValueError("shallow_frc_alpha and dfap_alpha must be non-negative")
        if shallow_frc_max_depth <= 0 or shallow_frc_max_depth > block_size:
            raise ValueError("shallow_frc_max_depth must be in [1, block_size]")
        if dfap_min_depth <= 0 or dfap_min_depth > block_size:
            raise ValueError("dfap_min_depth must be in [1, block_size]")
        if shallow_frc_temperature <= 0 or dfap_temperature <= 0:
            raise ValueError("FRC/DFAP temperatures must be positive")
        if state_credit_partition not in {"overlap", "strict"}:
            raise ValueError(
                "state_credit_partition must be overlap or strict, got "
                f"{state_credit_partition!r}"
            )
        self.shallow_frc_alpha = float(shallow_frc_alpha)
        self.shallow_frc_max_depth = int(shallow_frc_max_depth)
        self.shallow_frc_margin = float(shallow_frc_margin)
        self.shallow_frc_temperature = float(shallow_frc_temperature)
        self.dfap_alpha = float(dfap_alpha)
        self.dfap_min_depth = int(dfap_min_depth)
        self.dfap_margin = float(dfap_margin)
        self.dfap_temperature = float(dfap_temperature)
        self.state_credit_partition = state_credit_partition
        if conv_gate_loss_alpha < 0:
            raise ValueError("conv_gate_loss_alpha must be non-negative")
        if block_summary_counterfactual_alpha < 0:
            raise ValueError("block_summary_counterfactual_alpha must be non-negative")
        if block_summary_counterfactual_alpha and getattr(draft_model, "block_summary", None) is None:
            raise ValueError("counterfactual summary loss requires block_summary_rank > 0")
        if transition_credit_alpha < 0:
            raise ValueError("transition_credit_alpha must be non-negative")
        if transition_credit_max_depth < 2 or transition_credit_max_depth > block_size:
            raise ValueError(
                "transition_credit_max_depth must be in [2, block_size]"
            )
        if transition_credit_temperature <= 0:
            raise ValueError("transition_credit_temperature must be positive")
        if not 0.0 <= transition_credit_loss_budget <= 1.0:
            raise ValueError("transition_credit_loss_budget must be in [0, 1]")
        self.conv_gate_loss_alpha = float(conv_gate_loss_alpha)
        self.block_summary_counterfactual_alpha = float(block_summary_counterfactual_alpha)
        self.transition_credit_alpha = float(transition_credit_alpha)
        self.transition_credit_max_depth = int(transition_credit_max_depth)
        self.transition_credit_margin = float(transition_credit_margin)
        self.transition_credit_temperature = float(transition_credit_temperature)
        self.transition_credit_loss_budget = float(transition_credit_loss_budget)
        if transition2_margin_alpha < 0:
            raise ValueError("transition2_margin_alpha must be non-negative")
        if transition2_margin_temperature <= 0:
            raise ValueError("transition2_margin_temperature must be positive")
        if not 0.0 <= transition2_margin_loss_budget <= 1.0:
            raise ValueError("transition2_margin_loss_budget must be in [0, 1]")
        if prefix_bottleneck_alpha < 0:
            raise ValueError("prefix_bottleneck_alpha must be non-negative")
        if not 1 <= prefix_bottleneck_max_depth <= block_size:
            raise ValueError(
                "prefix_bottleneck_max_depth must be in [1, block_size]"
            )
        if (
            prefix_bottleneck_softmin_temperature <= 0
            or prefix_bottleneck_loss_temperature <= 0
        ):
            raise ValueError("prefix bottleneck temperatures must be positive")
        if not 0.0 <= prefix_bottleneck_loss_budget <= 1.0:
            raise ValueError("prefix_bottleneck_loss_budget must be in [0, 1]")
        if deep_survival_guard_alpha < 0:
            raise ValueError("deep_survival_guard_alpha must be non-negative")
        if not 1 <= deep_survival_guard_min_prefix <= block_size:
            raise ValueError(
                "deep_survival_guard_min_prefix must be in [1, block_size]"
            )
        if not 1 <= deep_survival_guard_start_depth <= block_size:
            raise ValueError(
                "deep_survival_guard_start_depth must be in [1, block_size]"
            )
        if deep_survival_guard_temperature <= 0:
            raise ValueError("deep_survival_guard_temperature must be positive")
        if not 0.0 <= deep_survival_guard_loss_budget <= 1.0:
            raise ValueError("deep_survival_guard_loss_budget must be in [0, 1]")
        self.transition2_margin_alpha = float(transition2_margin_alpha)
        self.transition2_margin_floor = float(transition2_margin_floor)
        self.transition2_margin_temperature = float(
            transition2_margin_temperature
        )
        self.transition2_margin_loss_budget = float(
            transition2_margin_loss_budget
        )
        self.prefix_bottleneck_alpha = float(prefix_bottleneck_alpha)
        self.prefix_bottleneck_max_depth = int(prefix_bottleneck_max_depth)
        self.prefix_bottleneck_margin_floor = float(
            prefix_bottleneck_margin_floor
        )
        self.prefix_bottleneck_softmin_temperature = float(
            prefix_bottleneck_softmin_temperature
        )
        self.prefix_bottleneck_loss_temperature = float(
            prefix_bottleneck_loss_temperature
        )
        self.prefix_bottleneck_loss_budget = float(
            prefix_bottleneck_loss_budget
        )
        self.deep_survival_guard_alpha = float(deep_survival_guard_alpha)
        self.deep_survival_guard_min_prefix = int(
            deep_survival_guard_min_prefix
        )
        self.deep_survival_guard_start_depth = int(
            deep_survival_guard_start_depth
        )
        self.deep_survival_guard_margin_floor = float(
            deep_survival_guard_margin_floor
        )
        self.deep_survival_guard_temperature = float(
            deep_survival_guard_temperature
        )
        self.deep_survival_guard_loss_budget = float(
            deep_survival_guard_loss_budget
        )
        if carh_recoverability_top_k < 0:
            raise ValueError("carh_recoverability_top_k must be non-negative")
        if min(
            carh_recovery_loss_alpha,
            carh_preservation_loss_alpha,
            carh_noop_loss_alpha,
        ) < 0:
            raise ValueError("CARH recoverability loss weights must be non-negative")
        if carh_preservation_temperature <= 0:
            raise ValueError("carh_preservation_temperature must be positive")
        if not 0.0 <= carh_recoverability_loss_budget <= 1.0:
            raise ValueError("carh_recoverability_loss_budget must be in [0, 1]")
        recoverability_weight = max(
            carh_recovery_loss_alpha,
            carh_preservation_loss_alpha,
            carh_noop_loss_alpha,
        )
        if (carh_recoverability_top_k > 0) != (recoverability_weight > 0):
            raise ValueError(
                "CARH recoverability requires both a positive top-k and at "
                "least one positive recoverability loss weight"
            )
        if carh_recoverability_top_k > 0 and getattr(
            draft_model.markov_head, "markov_head_type", None
        ) != "carh":
            raise ValueError("CARH recoverability requires --markov-head-type carh")
        if carh_recoverability_top_k > vocab_size:
            raise ValueError(
                "carh_recoverability_top_k must not exceed vocab size, got "
                f"{carh_recoverability_top_k} > {vocab_size}"
            )
        self.carh_recoverability_top_k = int(carh_recoverability_top_k)
        self.carh_recovery_loss_alpha = float(carh_recovery_loss_alpha)
        self.carh_preservation_loss_alpha = float(
            carh_preservation_loss_alpha
        )
        self.carh_noop_loss_alpha = float(carh_noop_loss_alpha)
        self.carh_preservation_margin = float(carh_preservation_margin)
        self.carh_preservation_temperature = float(
            carh_preservation_temperature
        )
        self.carh_recoverability_loss_budget = float(
            carh_recoverability_loss_budget
        )
        if min(carh_gate_calibration_alpha, carh_gate_noop_alpha) < 0:
            raise ValueError("CARH gate calibration weights must be non-negative")
        if carh_gate_margin_threshold < 0:
            raise ValueError("carh_gate_margin_threshold must be non-negative")
        if not 0.0 <= carh_gate_calibration_warmup_ratio <= 1.0:
            raise ValueError(
                "carh_gate_calibration_warmup_ratio must be in [0, 1]"
            )
        if max(carh_gate_calibration_alpha, carh_gate_noop_alpha) > 0 and getattr(
            draft_model.markov_head, "markov_head_type", None
        ) != "carh":
            raise ValueError("CARH gate calibration requires --markov-head-type carh")
        self.carh_gate_calibration_alpha = float(carh_gate_calibration_alpha)
        self.carh_gate_noop_alpha = float(carh_gate_noop_alpha)
        self.carh_gate_margin_threshold = float(carh_gate_margin_threshold)
        self.carh_gate_calibration_warmup_ratio = float(
            carh_gate_calibration_warmup_ratio
        )
        if selector_training_mode not in {
            "legacy",
            "recoverability-aware",
            "survival-distill",
        }:
            raise ValueError(
                "selector_training_mode must be legacy, recoverability-aware, "
                "or survival-distill"
            )
        if min(
            selector_loss_alpha,
            selector_distill_alpha,
            selector_preservation_alpha,
            selector_tree_loss_alpha,
        ) < 0:
            raise ValueError("selector loss weights must be non-negative")
        if not 0.0 <= selector_rollout_ratio <= 1.0:
            raise ValueError("selector_rollout_ratio must be in [0, 1]")
        if selector_distill_temperature <= 0:
            raise ValueError("selector_distill_temperature must be positive")
        if selector_advantage_margin < 0:
            raise ValueError("selector_advantage_margin must be non-negative")
        if not 0.0 <= selector_teacher_warmup_ratio <= 1.0:
            raise ValueError("selector_teacher_warmup_ratio must be in [0, 1]")
        if not 0.0 <= selector_distill_start_ratio <= 1.0:
            raise ValueError("selector_distill_start_ratio must be in [0, 1]")
        if selector_distill_start_ratio < selector_teacher_warmup_ratio:
            raise ValueError(
                "selector_distill_start_ratio must be >= teacher warmup ratio"
            )
        if not 0.0 <= selector_consolidation_ratio < 1.0:
            raise ValueError("selector_consolidation_ratio must be in [0, 1)")
        if selector_distill_start_ratio >= 1.0 - selector_consolidation_ratio:
            raise ValueError(
                "selector distillation must start before consolidation"
            )
        if not 0.0 <= selector_loss_budget <= 1.0:
            raise ValueError("selector_loss_budget must be in [0, 1]")
        if selector_margin_threshold < 0:
            raise ValueError("selector_margin_threshold must be non-negative")
        if selector_tree_branch_width < 1:
            raise ValueError("selector_tree_branch_width must be positive")
        if selector_tree_margin < 0:
            raise ValueError("selector_tree_margin must be non-negative")
        if selector_tree_temperature <= 0:
            raise ValueError("selector_tree_temperature must be positive")
        selector_weight = max(
            selector_loss_alpha,
            selector_distill_alpha,
            selector_preservation_alpha,
            selector_tree_loss_alpha,
        )
        if selector_training_mode == "legacy" and max(
            selector_distill_alpha, selector_preservation_alpha
        ) > 0:
            raise ValueError(
                "selector distillation/preservation requires survival-distill mode"
            )
        if (
            selector_training_mode == "recoverability-aware"
            and selector_distill_alpha > 0
        ):
            raise ValueError(
                "recoverability-aware selector training does not use CARH "
                "distillation; set selector_distill_alpha=0"
            )
        if selector_distill_alpha > 0 and selector_loss_alpha <= 0:
            raise ValueError(
                "selector_distill_alpha requires a positive selector_loss_alpha"
            )
        if selector_tree_loss_alpha > 0 and selector_tree_branch_width < 2:
            raise ValueError(
                "selector_tree_loss_alpha requires selector_tree_branch_width >= 2"
            )
        if selector_weight > 0 and getattr(
            draft_model, "candidate_selector", None
        ) is None:
            raise ValueError(
                "selector objectives require selector_rank/top_k on the model"
            )
        if (
            selector_training_mode == "survival-distill"
            and selector_weight > 0
            and getattr(draft_model.markov_head, "markov_head_type", None) != "carh"
        ):
            raise ValueError("survival-distill selector training requires CARH")
        if (
            selector_training_mode in {"recoverability-aware", "survival-distill"}
            and selector_weight > 0
            and int(draft_model.candidate_selector.top_k) < 2
        ):
            raise ValueError("strict selector training requires top_k >= 2")
        if (
            selector_weight > 0
            and selector_tree_branch_width
            > int(draft_model.candidate_selector.top_k)
        ):
            raise ValueError(
                "selector_tree_branch_width must not exceed selector_top_k"
            )
        self.selector_loss_alpha = float(selector_loss_alpha)
        self.selector_rollout_ratio = float(selector_rollout_ratio)
        self.selector_training_mode = selector_training_mode
        self.selector_distill_alpha = float(selector_distill_alpha)
        self.selector_preservation_alpha = float(selector_preservation_alpha)
        self.selector_distill_temperature = float(selector_distill_temperature)
        self.selector_advantage_margin = float(selector_advantage_margin)
        self.selector_teacher_warmup_ratio = float(
            selector_teacher_warmup_ratio
        )
        self.selector_distill_start_ratio = float(selector_distill_start_ratio)
        self.selector_consolidation_ratio = float(selector_consolidation_ratio)
        self.selector_loss_budget = float(selector_loss_budget)
        self.selector_margin_threshold = float(selector_margin_threshold)
        self.selector_tree_loss_alpha = float(selector_tree_loss_alpha)
        self.selector_tree_branch_width = int(selector_tree_branch_width)
        self.selector_tree_margin = float(selector_tree_margin)
        self.selector_tree_temperature = float(selector_tree_temperature)
        if (
            on_policy_survival_alpha < 0
            or on_policy_full_alpha < 0
            or on_policy_deep_alpha < 0
        ):
            raise ValueError("on-policy survival weights must be non-negative")
        if on_policy_first_rejection_max_depth < 1:
            raise ValueError("on_policy_first_rejection_max_depth must be >= 1")
        if on_policy_boundary_max_depth < 0:
            raise ValueError("on_policy_boundary_max_depth must be >= 0")
        if on_policy_deep_start_depth < 1:
            raise ValueError("on_policy_deep_start_depth must be >= 1")
        if on_policy_deep_alpha > 0 and on_policy_boundary_max_depth < 1:
            raise ValueError(
                "on_policy_deep_alpha requires an explicit positive "
                "on_policy_boundary_max_depth"
            )
        if (
            on_policy_deep_alpha > 0
            and on_policy_boundary_max_depth >= on_policy_deep_start_depth
        ):
            raise ValueError(
                "on-policy boundary and deep depth ranges must not overlap"
            )
        if on_policy_temperature <= 0 or on_policy_interval < 1:
            raise ValueError("on-policy temperature/interval must be positive")
        rollout_temperatures = tuple(
            float(value) for value in on_policy_rollout_temperatures
        )
        rollout_temperature_probs = tuple(
            float(value) for value in on_policy_rollout_temperature_probs
        )
        if not rollout_temperatures:
            raise ValueError("on_policy_rollout_temperatures must not be empty")
        if len(rollout_temperatures) != len(rollout_temperature_probs):
            raise ValueError(
                "on-policy rollout temperatures/probabilities must have the "
                "same length"
            )
        if any(value < 0 for value in rollout_temperatures):
            raise ValueError("on-policy rollout temperatures must be non-negative")
        if any(value < 0 for value in rollout_temperature_probs):
            raise ValueError(
                "on-policy rollout temperature probabilities must be "
                "non-negative"
            )
        probability_sum = sum(rollout_temperature_probs)
        if probability_sum <= 0:
            raise ValueError(
                "on-policy rollout temperature probabilities must have positive mass"
            )
        if not 0.0 <= on_policy_mix_ratio_max <= 1.0:
            raise ValueError("on_policy_mix_ratio_max must be in [0, 1]")
        if on_policy_anchor_sampling not in {
            "first",
            "uniform",
            "hazard",
            "hazard-flatness",
            "frontier-value",
            "dialogue-frontier",
        }:
            raise ValueError(
                "on_policy_anchor_sampling must be one of first/uniform/"
                "hazard/hazard-flatness/frontier-value/dialogue-frontier, got "
                f"{on_policy_anchor_sampling!r}"
            )
        if on_policy_hazard_power < 0 or on_policy_flatness_power_max < 0:
            raise ValueError("on-policy hazard/flatness powers must be non-negative")
        if on_policy_flatness_warmup_ratio < 0:
            raise ValueError("on_policy_flatness_warmup_ratio must be non-negative")
        if not 0.0 <= on_policy_uniform_exploration <= 1.0:
            raise ValueError("on_policy_uniform_exploration must be in [0, 1]")
        dialogue_phase_masses = tuple(float(value) for value in dialogue_phase_masses)
        if len(dialogue_phase_masses) != 5 or any(
            value < 0 for value in dialogue_phase_masses
        ) or sum(dialogue_phase_masses) <= 0:
            raise ValueError(
                "dialogue_phase_masses must contain five non-negative values "
                "with positive total mass"
            )
        if dialogue_boundary_boost < 0 or dialogue_script_transition_boost < 0:
            raise ValueError("dialogue boundary/script boosts must be non-negative")
        if dialogue_num_sources < 1:
            raise ValueError("dialogue_num_sources must be positive")
        if (
            dialogue_dsg_late_margin_bonus < 0
            or dialogue_dsg_boundary_margin_bonus < 0
        ):
            raise ValueError("dialogue DSG margin bonuses must be non-negative")
        if not 0.0 <= on_policy_language_han_threshold <= 1.0:
            raise ValueError("on_policy_language_han_threshold must be in [0, 1]")
        if not 0.0 <= on_policy_language_hazard_ema_decay < 1.0:
            raise ValueError(
                "on_policy_language_hazard_ema_decay must be in [0, 1)"
            )
        if not (
            0.0
            < on_policy_language_hazard_weight_min
            <= on_policy_language_hazard_weight_max
        ):
            raise ValueError("invalid on-policy language hazard weight range")
        if on_policy_distributional_top_k < 0:
            raise ValueError("on_policy_distributional_top_k must be non-negative")
        if on_policy_distributional_min_top_k < 0:
            raise ValueError(
                "on_policy_distributional_min_top_k must be non-negative"
            )
        if on_policy_distributional_min_top_k > on_policy_distributional_top_k:
            raise ValueError(
                "on_policy_distributional_min_top_k must not exceed "
                "on_policy_distributional_top_k"
            )
        if not 0.0 <= on_policy_distributional_mass_threshold < 1.0:
            raise ValueError(
                "on_policy_distributional_mass_threshold must be in [0, 1)"
            )
        if (
            on_policy_distributional_mass_threshold > 0
            and on_policy_distributional_min_top_k < 1
        ):
            raise ValueError(
                "adaptive-mass OPSC requires "
                "on_policy_distributional_min_top_k > 0"
            )
        if (
            on_policy_distributional_alpha < 0
            or on_policy_distributional_deep_alpha < 0
        ):
            raise ValueError("distributional OPSC weights must be non-negative")
        if not 0.0 <= on_policy_distributional_cold_start_ratio <= 1.0:
            raise ValueError(
                "on_policy_distributional_cold_start_ratio must be in [0, 1]"
            )
        if on_policy_rejection_aligned_alpha < 0:
            raise ValueError("rejection-aligned OPSC weight must be non-negative")
        if not 0.0 <= on_policy_rejection_aligned_survival_blend <= 1.0:
            raise ValueError(
                "on_policy_rejection_aligned_survival_blend must be in [0, 1]"
            )
        if on_policy_temperature_exclusive_routing:
            if on_policy_rejection_aligned_alpha <= 0:
                raise ValueError(
                    "temperature-exclusive routing requires positive "
                    "on_policy_rejection_aligned_alpha"
                )
            if not any(value <= 1e-6 for value in rollout_temperatures) or not any(
                value > 1e-6 for value in rollout_temperatures
            ):
                raise ValueError(
                    "temperature-exclusive routing requires both greedy and "
                    "positive rollout temperatures"
                )
        if min(
            on_policy_mixed_kl_alpha,
            on_policy_mixed_kl_accepted_weight,
            on_policy_mixed_kl_rejected_weight,
        ) < 0:
            raise ValueError("mixed-KL weights must be non-negative")
        if (
            on_policy_mixed_kl_alpha > 0
            and on_policy_mixed_kl_accepted_weight
            + on_policy_mixed_kl_rejected_weight
            <= 0
        ):
            raise ValueError(
                "mixed-KL requires a positive accepted or rejected weight"
            )
        if not 0.0 < on_policy_mixed_kl_rejection_decay <= 1.0:
            raise ValueError(
                "on_policy_mixed_kl_rejection_decay must be in (0, 1]"
            )
        clipped_rkl_temperature_weights = tuple(
            float(value) for value in on_policy_clipped_rkl_temperature_weights
        )
        if len(clipped_rkl_temperature_weights) == 1 and len(rollout_temperatures) > 1:
            clipped_rkl_temperature_weights = (
                clipped_rkl_temperature_weights * len(rollout_temperatures)
            )
        if on_policy_clipped_rkl_alpha < 0:
            raise ValueError("clipped reverse-KL weight must be non-negative")
        if on_policy_clipped_rkl_clip <= 0:
            raise ValueError("clipped reverse-KL clip must be positive")
        if len(clipped_rkl_temperature_weights) != len(rollout_temperatures):
            raise ValueError(
                "clipped reverse-KL temperature weights must match the rollout "
                "temperature schedule"
            )
        if any(value < 0 for value in clipped_rkl_temperature_weights):
            raise ValueError(
                "clipped reverse-KL temperature weights must be non-negative"
            )
        if on_policy_target_distribution_temperature_floor < 0:
            raise ValueError(
                "target distribution temperature floor must be non-negative"
            )
        if on_policy_target_margin_alpha < 0:
            raise ValueError("target-margin frontier weight must be non-negative")
        if on_policy_target_margin_scale < 0 or on_policy_target_margin_offset < 0:
            raise ValueError("target-margin scale/offset must be non-negative")
        if not 0 <= on_policy_target_margin_min <= on_policy_target_margin_max:
            raise ValueError("invalid target-margin clamp range")
        if (
            on_policy_target_margin_alpha > 0
            and not 1 <= on_policy_target_margin_max_depth <= block_size
        ):
            raise ValueError(
                "target-margin max depth must be in [1, block_size]"
            )
        if on_policy_deployment_top_k < 0:
            raise ValueError("on_policy_deployment_top_k must be non-negative")
        if not 0.0 < on_policy_deployment_top_p <= 1.0:
            raise ValueError("on_policy_deployment_top_p must be in (0, 1]")
        if on_policy_credit_partition not in {"legacy", "strict-v2"}:
            raise ValueError(
                "on_policy_credit_partition must be legacy or strict-v2"
            )
        if on_policy_credit_partition == "strict-v2" and not (
            on_policy_first_rejection_max_depth
            < on_policy_middle_max_depth
            < on_policy_deep_start_depth
        ):
            raise ValueError(
                "strict on-policy depth boundaries must satisfy "
                "first_rejection_max_depth < middle_max_depth < deep_start_depth"
            )
        if on_policy_middle_max_depth > block_size:
            raise ValueError("on_policy_middle_max_depth must not exceed block_size")
        if on_policy_middle_alpha < 0:
            raise ValueError("on_policy_middle_alpha must be non-negative")
        if not 0.0 <= on_policy_hazard_ema_decay < 1.0:
            raise ValueError("on_policy_hazard_ema_decay must be in [0, 1)")
        if not 0.0 < on_policy_hazard_weight_min <= on_policy_hazard_weight_max:
            raise ValueError("invalid on-policy hazard weight range")
        if on_policy_marginal_value_temperature <= 0:
            raise ValueError(
                "on_policy_marginal_value_temperature must be positive"
            )
        if not (
            0.0
            < on_policy_marginal_value_weight_min
            <= on_policy_marginal_value_weight_max
        ):
            raise ValueError("invalid on-policy marginal-value weight range")
        if min(
            on_policy_preservation_alpha,
            on_policy_regression_alpha,
            on_policy_greedy_preservation_alpha,
            on_policy_reset_replay_alpha,
        ) < 0:
            raise ValueError("on-policy consolidation weights must be non-negative")
        if on_policy_regression_margin < 0:
            raise ValueError("on_policy_regression_margin must be non-negative")
        if on_policy_greedy_preservation_margin_floor < 0:
            raise ValueError("greedy preservation margin floor must be non-negative")
        if on_policy_greedy_preservation_temperature <= 0:
            raise ValueError("greedy preservation temperature must be positive")
        if on_policy_reset_replay_horizon < 1:
            raise ValueError("reset replay horizon must be positive")
        if not 1 <= on_policy_reset_replay_max_rejection_depth <= block_size:
            raise ValueError(
                "reset replay max rejection depth must be in [1, block_size]"
            )
        if not 0.0 <= on_policy_reset_replay_start_ratio <= 1.0:
            raise ValueError("reset replay start ratio must be in [0, 1]")
        if on_policy_reset_replay_ramp_ratio < 0:
            raise ValueError("reset replay ramp ratio must be non-negative")
        if not 0.0 <= on_policy_reset_replay_loss_budget <= 1.0:
            raise ValueError("reset replay loss budget must be in [0, 1]")
        if on_policy_reset_replay_advantage_threshold < 0:
            raise ValueError("reset replay advantage threshold must be non-negative")
        if on_policy_reset_replay_value_clip <= 0:
            raise ValueError("reset replay value clip must be positive")
        if on_policy_preference_alpha < 0:
            raise ValueError("on-policy preference alpha must be non-negative")
        if on_policy_preference_value_gap < 0:
            raise ValueError("on-policy preference value gap must be non-negative")
        if on_policy_preference_temperature <= 0:
            raise ValueError("on-policy preference temperature must be positive")
        if not 0.0 <= on_policy_preference_loss_budget <= 1.0:
            raise ValueError("on-policy preference loss budget must be in [0, 1]")
        if not 0.0 <= on_policy_pareto_ema_decay < 1.0:
            raise ValueError("on_policy_pareto_ema_decay must be in [0, 1)")
        if on_policy_pareto_temperature <= 0:
            raise ValueError("on_policy_pareto_temperature must be positive")
        if not 0.0 < on_policy_pareto_weight_min <= on_policy_pareto_weight_max:
            raise ValueError("invalid on-policy Pareto weight range")
        if (
            max(
                on_policy_distributional_alpha,
                on_policy_distributional_deep_alpha,
                on_policy_rejection_aligned_alpha,
                on_policy_mixed_kl_alpha,
                on_policy_clipped_rkl_alpha,
                on_policy_target_margin_alpha,
            )
            > 0
            and on_policy_distributional_top_k < 1
        ):
            raise ValueError(
                "distributional OPSC requires on_policy_distributional_top_k > 0"
            )
        if on_policy_target_margin_alpha > 0 and on_policy_distributional_top_k < 2:
            raise ValueError(
                "target-margin frontier requires on_policy_distributional_top_k >= 2"
            )
        self.on_policy_survival_alpha = float(on_policy_survival_alpha)
        self.on_policy_full_alpha = float(on_policy_full_alpha)
        self.on_policy_first_rejection_max_depth = int(
            on_policy_first_rejection_max_depth
        )
        self.on_policy_boundary_max_depth = int(on_policy_boundary_max_depth)
        self.on_policy_deep_alpha = float(on_policy_deep_alpha)
        self.on_policy_deep_start_depth = int(on_policy_deep_start_depth)
        self.on_policy_margin_floor = float(on_policy_margin_floor)
        self.on_policy_temperature = float(on_policy_temperature)
        self.on_policy_rollout_temperatures = rollout_temperatures
        self.on_policy_rollout_temperature_probs = tuple(
            value / probability_sum for value in rollout_temperature_probs
        )
        self.on_policy_anchor_sampling = on_policy_anchor_sampling
        self.on_policy_hazard_power = float(on_policy_hazard_power)
        self.on_policy_flatness_power_max = float(on_policy_flatness_power_max)
        self.on_policy_flatness_warmup_ratio = float(
            on_policy_flatness_warmup_ratio
        )
        self.on_policy_uniform_exploration = float(on_policy_uniform_exploration)
        self.dialogue_phase_masses = tuple(
            value / sum(dialogue_phase_masses) for value in dialogue_phase_masses
        )
        self.dialogue_boundary_boost = float(dialogue_boundary_boost)
        self.dialogue_script_transition_boost = float(
            dialogue_script_transition_boost
        )
        self.dialogue_occupancy_diagnostics = bool(
            dialogue_occupancy_diagnostics
        )
        self.dialogue_num_sources = int(dialogue_num_sources)
        self.dialogue_prefix_phase_weighting = bool(
            dialogue_prefix_phase_weighting
        )
        self.dialogue_t2cm_boundary_weighting = bool(
            dialogue_t2cm_boundary_weighting
        )
        self.dialogue_dsg_adaptive_margin = bool(
            dialogue_dsg_adaptive_margin
        )
        self.dialogue_dsg_late_margin_bonus = float(
            dialogue_dsg_late_margin_bonus
        )
        self.dialogue_dsg_boundary_margin_bonus = float(
            dialogue_dsg_boundary_margin_bonus
        )
        if dialogue_token_class_lookup is None:
            dialogue_token_class_lookup = torch.zeros(
                int(target_lm_head.weight.shape[0]),
                dtype=torch.uint8,
                device=target_lm_head.weight.device,
            )
        self.register_buffer(
            "dialogue_token_class_lookup",
            dialogue_token_class_lookup.to(
                device=target_lm_head.weight.device, dtype=torch.uint8
            ),
            persistent=False,
        )
        self.on_policy_distributional_top_k = int(on_policy_distributional_top_k)
        self.on_policy_distributional_min_top_k = int(
            on_policy_distributional_min_top_k
        )
        self.on_policy_distributional_mass_threshold = float(
            on_policy_distributional_mass_threshold
        )
        self.on_policy_distributional_alpha = float(
            on_policy_distributional_alpha
        )
        self.on_policy_distributional_deep_alpha = float(
            on_policy_distributional_deep_alpha
        )
        self.on_policy_distributional_cold_start_ratio = float(
            on_policy_distributional_cold_start_ratio
        )
        self.on_policy_rejection_aligned_alpha = float(
            on_policy_rejection_aligned_alpha
        )
        self.on_policy_rejection_aligned_survival_blend = float(
            on_policy_rejection_aligned_survival_blend
        )
        self.on_policy_temperature_exclusive_routing = bool(
            on_policy_temperature_exclusive_routing
        )
        self.on_policy_mixed_kl_alpha = float(on_policy_mixed_kl_alpha)
        self.on_policy_mixed_kl_accepted_weight = float(
            on_policy_mixed_kl_accepted_weight
        )
        self.on_policy_mixed_kl_rejected_weight = float(
            on_policy_mixed_kl_rejected_weight
        )
        self.on_policy_mixed_kl_rejection_decay = float(
            on_policy_mixed_kl_rejection_decay
        )
        self.on_policy_clipped_rkl_alpha = float(on_policy_clipped_rkl_alpha)
        self.on_policy_clipped_rkl_clip = float(on_policy_clipped_rkl_clip)
        self.on_policy_clipped_rkl_temperature_weights = (
            clipped_rkl_temperature_weights
        )
        self.on_policy_target_distribution_temperature_floor = float(
            on_policy_target_distribution_temperature_floor
        )
        self.on_policy_target_margin_alpha = float(on_policy_target_margin_alpha)
        self.on_policy_target_margin_scale = float(on_policy_target_margin_scale)
        self.on_policy_target_margin_offset = float(on_policy_target_margin_offset)
        self.on_policy_target_margin_min = float(on_policy_target_margin_min)
        self.on_policy_target_margin_max = float(on_policy_target_margin_max)
        self.on_policy_target_margin_max_depth = int(
            on_policy_target_margin_max_depth
        )
        self.on_policy_deployment_top_k = int(on_policy_deployment_top_k)
        self.on_policy_deployment_top_p = float(on_policy_deployment_top_p)
        self.on_policy_credit_partition = on_policy_credit_partition
        self.on_policy_middle_max_depth = int(on_policy_middle_max_depth)
        self.on_policy_middle_alpha = float(on_policy_middle_alpha)
        self.on_policy_temperature_hazard_credit = bool(
            on_policy_temperature_hazard_credit
        )
        self.on_policy_hazard_ema_decay = float(on_policy_hazard_ema_decay)
        self.on_policy_hazard_weight_min = float(on_policy_hazard_weight_min)
        self.on_policy_hazard_weight_max = float(on_policy_hazard_weight_max)
        self.on_policy_marginal_value_credit = bool(
            on_policy_marginal_value_credit
        )
        self.on_policy_marginal_value_temperature = float(
            on_policy_marginal_value_temperature
        )
        self.on_policy_marginal_value_weight_min = float(
            on_policy_marginal_value_weight_min
        )
        self.on_policy_marginal_value_weight_max = float(
            on_policy_marginal_value_weight_max
        )
        self.register_buffer(
            "on_policy_temperature_hazard_ema",
            torch.ones(len(rollout_temperatures), block_size),
            persistent=False,
        )
        self.register_buffer(
            "on_policy_temperature_hazard_initialized",
            torch.zeros(len(rollout_temperatures), block_size, dtype=torch.bool),
            persistent=False,
        )
        self.on_policy_language_hazard_credit = bool(
            on_policy_language_hazard_credit
        )
        self.on_policy_language_han_threshold = float(
            on_policy_language_han_threshold
        )
        self.on_policy_language_hazard_ema_decay = float(
            on_policy_language_hazard_ema_decay
        )
        self.on_policy_language_hazard_weight_min = float(
            on_policy_language_hazard_weight_min
        )
        self.on_policy_language_hazard_weight_max = float(
            on_policy_language_hazard_weight_max
        )
        # Two response-language groups (other/Chinese) by three transition
        # groups (ordinary/boundary/Han-Latin switch) and draft depth.
        self.register_buffer(
            "on_policy_language_hazard_ema",
            torch.ones(2, 3, block_size),
            persistent=False,
        )
        self.register_buffer(
            "on_policy_language_hazard_initialized",
            torch.zeros(2, 3, block_size, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "on_policy_language_global_hazard_ema",
            torch.ones(block_size),
            persistent=False,
        )
        self.register_buffer(
            "on_policy_language_global_hazard_initialized",
            torch.zeros(block_size, dtype=torch.bool),
            persistent=False,
        )
        self.on_policy_mix_ratio_max = float(on_policy_mix_ratio_max)
        if on_policy_start_step is not None and on_policy_start_step < 1:
            raise ValueError("on_policy_start_step must be positive")
        if on_policy_ramp_steps < 0:
            raise ValueError("on_policy_ramp_steps must be non-negative")
        self.on_policy_start_step = on_policy_start_step
        self.on_policy_ramp_steps = int(on_policy_ramp_steps)
        self.on_policy_interval = int(on_policy_interval)
        self.on_policy_loss_budget = float(on_policy_loss_budget)
        self.on_policy_preservation_alpha = float(on_policy_preservation_alpha)
        self.on_policy_regression_alpha = float(on_policy_regression_alpha)
        self.on_policy_regression_margin = float(on_policy_regression_margin)
        self.on_policy_greedy_preservation_alpha = float(
            on_policy_greedy_preservation_alpha
        )
        self.on_policy_greedy_preservation_margin_floor = float(
            on_policy_greedy_preservation_margin_floor
        )
        self.on_policy_greedy_preservation_temperature = float(
            on_policy_greedy_preservation_temperature
        )
        self.on_policy_reset_replay_alpha = float(on_policy_reset_replay_alpha)
        self.on_policy_repair_value = bool(on_policy_repair_value)
        if on_policy_repair_value_horizon < 1:
            raise ValueError("on_policy_repair_value_horizon must be positive")
        self.on_policy_repair_value_horizon = int(on_policy_repair_value_horizon)
        self.on_policy_reset_replay_horizon = int(on_policy_reset_replay_horizon)
        self.on_policy_reset_replay_max_rejection_depth = int(
            on_policy_reset_replay_max_rejection_depth
        )
        self.on_policy_reset_replay_start_ratio = float(
            on_policy_reset_replay_start_ratio
        )
        self.on_policy_reset_replay_ramp_ratio = float(
            on_policy_reset_replay_ramp_ratio
        )
        self.on_policy_reset_replay_loss_budget = float(
            on_policy_reset_replay_loss_budget
        )
        self.on_policy_reset_replay_advantage_threshold = float(
            on_policy_reset_replay_advantage_threshold
        )
        self.on_policy_reset_replay_value_clip = float(
            on_policy_reset_replay_value_clip
        )
        self.on_policy_preference_alpha = float(on_policy_preference_alpha)
        self.on_policy_preference_value_gap = float(
            on_policy_preference_value_gap
        )
        self.on_policy_preference_temperature = float(
            on_policy_preference_temperature
        )
        self.on_policy_preference_loss_budget = float(
            on_policy_preference_loss_budget
        )
        self.on_policy_pareto_credit = bool(on_policy_pareto_credit)
        self.on_policy_pareto_ema_decay = float(on_policy_pareto_ema_decay)
        self.on_policy_pareto_temperature = float(on_policy_pareto_temperature)
        self.on_policy_pareto_weight_min = float(on_policy_pareto_weight_min)
        self.on_policy_pareto_weight_max = float(on_policy_pareto_weight_max)
        if min(
            parallel_refiner_hazard_alpha,
            parallel_refiner_recovery_alpha,
            parallel_refiner_preservation_alpha,
        ) < 0:
            raise ValueError("parallel refiner loss weights must be non-negative")
        if parallel_refiner_margin_floor < 0:
            raise ValueError("parallel_refiner_margin_floor must be non-negative")
        if parallel_refiner_temperature <= 0:
            raise ValueError("parallel_refiner_temperature must be positive")
        if not 0.0 <= parallel_refiner_loss_budget <= 1.0:
            raise ValueError("parallel_refiner_loss_budget must be in [0, 1]")
        refiner_objective_enabled = max(
            parallel_refiner_hazard_alpha,
            parallel_refiner_recovery_alpha,
            parallel_refiner_preservation_alpha,
        ) > 0
        if refiner_objective_enabled and getattr(
            draft_model, "parallel_refiner", None
        ) is None:
            raise ValueError(
                "parallel refiner objectives require --parallel-refiner-rank "
                "and --parallel-refiner-steps"
            )
        self.parallel_refiner_hazard_alpha = float(
            parallel_refiner_hazard_alpha
        )
        self.parallel_refiner_recovery_alpha = float(
            parallel_refiner_recovery_alpha
        )
        self.parallel_refiner_preservation_alpha = float(
            parallel_refiner_preservation_alpha
        )
        self.parallel_refiner_margin_floor = float(
            parallel_refiner_margin_floor
        )
        self.parallel_refiner_temperature = float(parallel_refiner_temperature)
        self.parallel_refiner_loss_budget = float(parallel_refiner_loss_budget)
        if refiner_advantage_mode not in {"none", "diagnostics", "distill"}:
            raise ValueError(
                "refiner_advantage_mode must be none/diagnostics/distill"
            )
        if min(
            refiner_advantage_gate_alpha,
            refiner_advantage_regression_alpha,
            refiner_advantage_distill_alpha,
            refiner_advantage_preservation_alpha,
            refiner_advantage_greedy_preservation_alpha,
        ) < 0:
            raise ValueError("refiner advantage weights must be non-negative")
        if refiner_advantage_distill_top_k < 2:
            raise ValueError("refiner_advantage_distill_top_k must be >= 2")
        if refiner_advantage_distill_temperature <= 0:
            raise ValueError("refiner advantage temperature must be positive")
        if refiner_advantage_value_clip <= 0:
            raise ValueError("refiner advantage value clip must be positive")
        if not 0.0 <= refiner_advantage_distill_start_ratio < 1.0:
            raise ValueError("invalid refiner advantage distill start ratio")
        if not 0.0 <= refiner_advantage_consolidation_ratio < 1.0:
            raise ValueError("invalid refiner advantage consolidation ratio")
        if (
            refiner_advantage_distill_start_ratio
            + refiner_advantage_consolidation_ratio
            >= 1.0
        ):
            raise ValueError("advantage distill start + consolidation must be < 1")
        if not 0.0 <= refiner_advantage_loss_budget <= 1.0:
            raise ValueError("refiner advantage loss budget must be in [0, 1]")
        if refiner_advantage_mode != "none" and getattr(
            draft_model, "parallel_refiner", None
        ) is None:
            raise ValueError("refiner advantage training requires a refiner")
        self.refiner_advantage_mode = str(refiner_advantage_mode)
        self.refiner_advantage_threshold = float(refiner_advantage_threshold)
        self.refiner_advantage_gate_alpha = float(refiner_advantage_gate_alpha)
        self.refiner_advantage_regression_alpha = float(
            refiner_advantage_regression_alpha
        )
        self.refiner_advantage_distill_alpha = float(
            refiner_advantage_distill_alpha
        )
        self.refiner_advantage_preservation_alpha = float(
            refiner_advantage_preservation_alpha
        )
        self.refiner_advantage_greedy_preservation_alpha = float(
            refiner_advantage_greedy_preservation_alpha
        )
        self.refiner_advantage_greedy_margin_floor = float(
            refiner_advantage_greedy_margin_floor
        )
        if self.refiner_advantage_greedy_margin_floor < 0:
            raise ValueError("greedy preservation margin floor must be non-negative")
        self.refiner_advantage_gate_distill_min_probability = float(
            refiner_advantage_gate_distill_min_probability
        )
        if not 0.0 <= self.refiner_advantage_gate_distill_min_probability < 1.0:
            raise ValueError("advantage gate distill probability must be in [0, 1)")
        self.refiner_advantage_distill_top_k = int(
            refiner_advantage_distill_top_k
        )
        self.refiner_advantage_distill_temperature = float(
            refiner_advantage_distill_temperature
        )
        self.refiner_advantage_value_clip = float(refiner_advantage_value_clip)
        self.refiner_advantage_distill_start_ratio = float(
            refiner_advantage_distill_start_ratio
        )
        self.refiner_advantage_consolidation_ratio = float(
            refiner_advantage_consolidation_ratio
        )
        self.refiner_advantage_loss_budget = float(
            refiner_advantage_loss_budget
        )
        self.register_buffer(
            "on_policy_temperature_shortfall_ema",
            torch.ones(len(rollout_temperatures)),
            persistent=False,
        )
        self.register_buffer(
            "on_policy_temperature_shortfall_initialized",
            torch.zeros(len(rollout_temperatures), dtype=torch.bool),
            persistent=False,
        )
        if min(
            vat_hard_loss_alpha,
            vat_soft_loss_alpha,
            vat_verification_head_alpha,
        ) < 0:
            raise ValueError("VAT loss weights must be non-negative")
        if vat_post_rejection_decay_gamma <= 0:
            raise ValueError("VAT post-rejection decay gamma must be positive")
        if vat_simulation_temperature < 0:
            raise ValueError("VAT simulation temperature must be non-negative")
        if vat_enabled and getattr(draft_model, "confidence_head", None) is None:
            raise ValueError("VAT requires the DSpark confidence/verification head")
        if vat_enabled and confidence_detach_backbone:
            raise ValueError(
                "VAT verification-head gradients must reach the backbone"
            )
        if vat_enabled and pace_mode != "none":
            raise ValueError("VAT matched baseline requires pace_mode='none'")
        self.vat_enabled = bool(vat_enabled)
        self.vat_hard_loss_alpha = float(vat_hard_loss_alpha)
        self.vat_soft_loss_alpha = float(vat_soft_loss_alpha)
        self.vat_verification_head_alpha = float(
            vat_verification_head_alpha
        )
        self.vat_post_rejection_decay_gamma = float(
            vat_post_rejection_decay_gamma
        )
        self.vat_simulation_temperature = float(vat_simulation_temperature)
        if branch_value_mode not in {"none", "diagnostics", "distill"}:
            raise ValueError("branch_value_mode must be none/diagnostics/distill")
        if not 2 <= branch_value_top_m <= vocab_size:
            raise ValueError("branch_value_top_m must be in [2, vocab_size]")
        if not 1 <= branch_value_horizon <= block_size:
            raise ValueError("branch_value_horizon must be in [1, block_size]")
        if branch_value_alpha < 0 or branch_value_temperature <= 0:
            raise ValueError("invalid branch value alpha/temperature")
        if not 0.0 <= branch_value_loss_budget <= 1.0:
            raise ValueError("branch value loss budget must be in [0, 1]")
        if not 0.0 <= branch_value_warmup_ratio < 1.0:
            raise ValueError("branch value warmup ratio must be in [0, 1)")
        if not 0.0 <= branch_value_consolidation_ratio < 1.0:
            raise ValueError("branch value consolidation ratio must be in [0, 1)")
        if branch_value_warmup_ratio + branch_value_consolidation_ratio >= 1.0:
            raise ValueError("branch value warmup + consolidation must be < 1")
        if branch_value_mode == "distill" and branch_value_alpha <= 0:
            raise ValueError("branch value distillation requires positive alpha")
        self.branch_value_mode = branch_value_mode
        self.branch_value_top_m = int(branch_value_top_m)
        self.branch_value_horizon = int(branch_value_horizon)
        self.branch_value_alpha = float(branch_value_alpha)
        self.branch_value_temperature = float(branch_value_temperature)
        self.branch_value_loss_budget = float(branch_value_loss_budget)
        self.branch_value_warmup_ratio = float(branch_value_warmup_ratio)
        self.branch_value_consolidation_ratio = float(
            branch_value_consolidation_ratio
        )
        if elastic_horizon_enabled:
            if not 1 <= elastic_short_horizon < elastic_long_horizon <= block_size:
                raise ValueError(
                    "elastic horizons must satisfy 1 <= short < long <= block_size"
                )
            if elastic_long_horizon != block_size:
                raise ValueError(
                    "elastic_long_horizon must equal the model's fixed block_size"
                )
            if not 0.0 <= elastic_warmup_ratio < elastic_late_ratio < 1.0:
                raise ValueError("invalid elastic warmup/late schedule")
            if not 0.0 <= elastic_consolidation_ratio < 1.0:
                raise ValueError("elastic consolidation ratio must be in [0, 1)")
            if elastic_late_ratio >= 1.0 - elastic_consolidation_ratio:
                raise ValueError("elastic late phase must precede consolidation")
            if not all(
                0.0 <= value <= 1.0
                for value in (
                    elastic_middle_long_prob,
                    elastic_late_long_prob,
                    elastic_pair_probability,
                    elastic_loss_budget,
                )
            ):
                raise ValueError("elastic probabilities/budget must be in [0, 1]")
            if elastic_projective_alpha < 0:
                raise ValueError("elastic_projective_alpha must be non-negative")
            if elastic_projective_num_anchors < 1:
                raise ValueError("elastic_projective_num_anchors must be positive")
            if not 2 <= elastic_projective_top_k <= vocab_size:
                raise ValueError("elastic_projective_top_k must be in [2, vocab_size]")
            if elastic_overlap_gap < 0 or elastic_margin_gap < 0:
                raise ValueError("elastic verifier gates must be non-negative")
        self.elastic_horizon_enabled = bool(elastic_horizon_enabled)
        self.elastic_short_horizon = int(elastic_short_horizon)
        self.elastic_long_horizon = int(elastic_long_horizon)
        self.elastic_warmup_ratio = float(elastic_warmup_ratio)
        self.elastic_late_ratio = float(elastic_late_ratio)
        self.elastic_consolidation_ratio = float(elastic_consolidation_ratio)
        self.elastic_middle_long_prob = float(elastic_middle_long_prob)
        self.elastic_late_long_prob = float(elastic_late_long_prob)
        self.elastic_pair_probability = float(elastic_pair_probability)
        self.elastic_projective_num_anchors = int(elastic_projective_num_anchors)
        self.elastic_projective_alpha = float(elastic_projective_alpha)
        self.elastic_projective_top_k = int(elastic_projective_top_k)
        self.elastic_overlap_gap = float(elastic_overlap_gap)
        self.elastic_margin_gap = float(elastic_margin_gap)
        self.elastic_loss_budget = float(elastic_loss_budget)
        # Installed by train_dspark.py before FSDP wrapping.  Keeping the frozen
        # verifier outside this nn.Module avoids registering target parameters.
        self.on_policy_scorer: Optional[Callable] = None
        # Plain Python hooks keep offline teacher caches out of FSDP state and
        # out of the exported serving checkpoint.
        self.multi_teacher_export_sink: Optional[Callable] = None
        self.multi_teacher_oracle_provider: Optional[Callable] = None
        self.multi_teacher_oracle_mode = "none"
        self.multi_teacher_oracle_alpha = 0.0
        self.multi_teacher_oracle_advantage_threshold = 0.0
        self.multi_teacher_oracle_value_clip = float(block_size)
        self.multi_teacher_oracle_loss_budget = 0.05

    def configure_multi_teacher_oracle(
        self,
        mode: str = "none",
        alpha: float = 0.0,
        advantage_threshold: float = 0.0,
        value_clip: Optional[float] = None,
        loss_budget: float = 0.05,
    ) -> None:
        if mode not in {"none", "diagnostics", "distill"}:
            raise ValueError("multi-teacher oracle mode must be none/diagnostics/distill")
        if alpha < 0 or advantage_threshold < 0:
            raise ValueError("multi-teacher oracle alpha/threshold must be non-negative")
        if not 0.0 <= loss_budget <= 1.0:
            raise ValueError("multi-teacher oracle loss budget must be in [0, 1]")
        self.multi_teacher_oracle_mode = mode
        self.multi_teacher_oracle_alpha = float(alpha)
        self.multi_teacher_oracle_advantage_threshold = float(advantage_threshold)
        self.multi_teacher_oracle_value_clip = float(value_clip or self.block_size)
        self.multi_teacher_oracle_loss_budget = float(loss_budget)

    def _vat_verification_targets(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        eval_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Simulate sequential verification on teacher-forced VAT states.

        VAT deliberately differs from OPSC: it does not run a proposal-induced
        target forward.  Draft and target distributions are aligned on the
        training sequence, exactly matching the inexpensive simulated
        verification baseline.  Greedy simulation is deterministic; positive
        temperatures use the standard speculative-sampling acceptance ratio.
        """
        valid = eval_mask.bool()
        with torch.no_grad():
            if self.vat_simulation_temperature == 0:
                draft_tokens = draft_logits.argmax(dim=-1)
                target_tokens = target_logits.argmax(dim=-1)
                accepted = draft_tokens.eq(target_tokens) & valid
            else:
                temperature = self.vat_simulation_temperature
                draft_probs = torch.softmax(
                    draft_logits.float() / temperature, dim=-1
                )
                target_probs = torch.softmax(
                    target_logits.float() / temperature, dim=-1
                )
                draft_tokens = torch.multinomial(
                    draft_probs.reshape(-1, draft_probs.size(-1)), 1
                ).view(draft_probs.shape[:-1])
                draft_mass = draft_probs.gather(
                    -1, draft_tokens.unsqueeze(-1)
                ).squeeze(-1)
                target_mass = target_probs.gather(
                    -1, draft_tokens.unsqueeze(-1)
                ).squeeze(-1)
                acceptance_probability = (
                    target_mass / draft_mass.clamp_min(1e-12)
                ).clamp(max=1.0)
                accepted = (
                    torch.rand_like(acceptance_probability)
                    < acceptance_probability
                ) & valid

            # Invalid suffix positions are neutral in the cumulative product.
            accepted_or_invalid = accepted | ~valid
            survival = torch.cumprod(
                accepted_or_invalid.to(torch.int32), dim=-1
            ).bool() & valid
            rejected = ~accepted & valid
            depth = torch.arange(
                draft_logits.size(-2), device=draft_logits.device
            ).view(*([1] * (draft_logits.dim() - 2)), -1)
            rejection_depth = torch.where(
                rejected,
                depth,
                torch.full_like(depth, draft_logits.size(-2)),
            ).amin(dim=-1, keepdim=True)
            post_rejection_offset = (depth - rejection_depth).clamp_min(0)
            adaptive_weight = torch.exp(
                -post_rejection_offset.float()
                / self.vat_post_rejection_decay_gamma
            ) * eval_mask

            valid_blocks = valid[..., 0].sum().clamp_min(1)
            first_rejection_depth = torch.where(
                rejection_depth.squeeze(-1).lt(draft_logits.size(-2)),
                rejection_depth.squeeze(-1).float() + 1.0,
                torch.full_like(
                    rejection_depth.squeeze(-1).float(),
                    float(draft_logits.size(-2) + 1),
                ),
            )
            diagnostics = {
                "vat_first_rejection_depth": (
                    first_rejection_depth * valid[..., 0]
                ).sum()
                / valid_blocks,
                "vat_full_survival_rate": (
                    survival[..., -1].float() * valid[..., 0]
                ).sum()
                / valid_blocks,
                "vat_adaptive_weight_mean": adaptive_weight.sum()
                / eval_mask.sum().clamp_min(1.0),
            }
            for depth_index in range(draft_logits.size(-2)):
                diagnostics[f"vat_survival_{depth_index + 1}"] = (
                    survival[..., depth_index].float() * valid[..., 0]
                ).sum() / valid_blocks
        return survival.float(), adaptive_weight, diagnostics

    def _elastic_progress(self) -> float:
        if self._total_steps <= 0:
            return 0.0
        return min(max(self._global_step / self._total_steps, 0.0), 1.0)

    @staticmethod
    def _step_uniform(step: int, salt: int) -> float:
        """Deterministic cross-rank pseudo-random value in [0, 1)."""
        value = (int(step) + int(salt)) & 0xFFFFFFFF
        value ^= value >> 16
        value = (value * 0x7FEB352D) & 0xFFFFFFFF
        value ^= value >> 15
        value = (value * 0x846CA68B) & 0xFFFFFFFF
        value ^= value >> 16
        return value / float(0x100000000)

    def _elastic_active_horizon(self) -> int:
        if not self.elastic_horizon_enabled:
            return self.block_size
        progress = self._elastic_progress()
        if (
            progress < self.elastic_warmup_ratio
            or progress >= 1.0 - self.elastic_consolidation_ratio
        ):
            return self.elastic_short_horizon
        long_probability = (
            self.elastic_middle_long_prob
            if progress < self.elastic_late_ratio
            else self.elastic_late_long_prob
        )
        return (
            self.elastic_long_horizon
            if self._step_uniform(self._global_step, 12345) < long_probability
            else self.elastic_short_horizon
        )

    def _elastic_pair_enabled(self) -> bool:
        if (
            not self.elastic_horizon_enabled
            or self.elastic_projective_alpha <= 0
            or self._total_steps <= 0
        ):
            return False
        progress = self._elastic_progress()
        if (
            progress < self.elastic_warmup_ratio
            or progress >= 1.0 - self.elastic_consolidation_ratio
        ):
            return False
        return (
            self._step_uniform(self._global_step, 67891)
            < self.elastic_pair_probability
        )

    def _elastic_gate_temperature(self) -> float:
        draw = self._step_uniform(self._global_step, 98765)
        cumulative = 0.0
        for temperature, probability in zip(
            self.on_policy_rollout_temperatures,
            self.on_policy_rollout_temperature_probs,
        ):
            cumulative += probability
            if draw < cumulative:
                return float(temperature)
        return float(self.on_policy_rollout_temperatures[-1])

    @staticmethod
    def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
        """Preserve the forward value while multiplying its backward gradient."""
        if scale >= 1.0:
            return value
        if scale <= 0.0:
            return value.detach()
        return value.detach() + float(scale) * (value - value.detach())

    def _budget_auxiliary_loss(
        self,
        weighted_loss: torch.Tensor,
        loss_budget: float,
        ce_num: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cap an auxiliary numerator against the detached main CE numerator.

        The cap prevents a small set of fragile blocks from dominating the
        batch.  A zero budget means no cap, matching the convention used by
        the new survival-transition objectives.
        """
        if loss_budget <= 0:
            return weighted_loss, weighted_loss.new_ones(())
        allowed = float(loss_budget) * self.ce_loss_alpha * ce_num.detach()
        scale = torch.clamp(
            allowed / weighted_loss.detach().clamp_min(1e-6), max=1.0
        )
        return weighted_loss * scale, scale

    @staticmethod
    def _carh_recoverability_partition(
        base_logits: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Partition tokens into correct, recoverable, and recall-miss states.

        Membership is deliberately computed under ``no_grad`` from the unary
        backbone distribution.  The model therefore cannot manipulate Top-k
        membership to enter or leave CARH's correction objective.
        """
        with torch.no_grad():
            candidate_ids = base_logits.detach().topk(top_k, dim=-1).indices
            covered = candidate_ids.eq(target_ids.unsqueeze(-1)).any(dim=-1)
            base_correct = base_logits.detach().argmax(dim=-1).eq(target_ids)
            valid = eval_mask.bool()
            correct = valid & base_correct
            recoverable = valid & covered & ~base_correct
            recall_miss = valid & ~covered
        return candidate_ids, correct, recoverable, recall_miss

    def _selector_phase_scales(self) -> Tuple[float, float]:
        """Return teacher and distillation gates for the staged objective.

        The final consolidation window deliberately removes both selector
        losses while the normal DSpark/CARH objectives remain active.  This
        makes the last checkpoint exercise the same student-only computation
        that will be used when ``selector_runtime_enabled`` is false.
        """
        if self.selector_training_mode != "survival-distill":
            return 1.0, 0.0
        if self._total_steps <= 0:
            return 0.0, 0.0
        progress = self._global_step / self._total_steps
        consolidation_start = 1.0 - self.selector_consolidation_ratio
        if progress >= consolidation_start:
            return 0.0, 0.0
        teacher = float(progress >= self.selector_teacher_warmup_ratio)
        distill = float(progress >= self.selector_distill_start_ratio)
        return teacher, distill

    def _effective_prefix_credit_alpha(self) -> float:
        if self.prefix_credit_mode == "none" or self.prefix_credit_alpha <= 0:
            return 0.0
        if self._total_steps <= 0:
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.prefix_credit_warmup_ratio:
            return 0.0
        if self.prefix_credit_ramp_ratio <= 0:
            return self.prefix_credit_alpha
        ramp_progress = (
            progress - self.prefix_credit_warmup_ratio
        ) / self.prefix_credit_ramp_ratio
        return self.prefix_credit_alpha * min(max(ramp_progress, 0.0), 1.0)

    def _effective_state_credit_scale(self) -> float:
        """Use Prefix-Full's schedule for the state-partitioned auxiliaries."""
        if (
            self.shallow_frc_alpha <= 0
            and self.dfap_alpha <= 0
            and self.transition_credit_alpha <= 0
            and self.conv_gate_loss_alpha <= 0
            and self.transition2_margin_alpha <= 0
            and self.prefix_bottleneck_alpha <= 0
            and self.deep_survival_guard_alpha <= 0
        ):
            return 0.0
        if self._total_steps <= 0:
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.prefix_credit_warmup_ratio:
            return 0.0
        if self.prefix_credit_ramp_ratio <= 0:
            return 1.0
        return min(
            max(
                (progress - self.prefix_credit_warmup_ratio)
                / self.prefix_credit_ramp_ratio,
                0.0,
            ),
            1.0,
        )

    def _effective_on_policy_mix(self) -> float:
        if getattr(self, "_reference_replay_active", False):
            return 0.0
        if (
            max(
                self.on_policy_survival_alpha,
                self.on_policy_full_alpha,
                self.on_policy_deep_alpha,
                self.on_policy_distributional_alpha,
                self.on_policy_distributional_deep_alpha,
                self.on_policy_rejection_aligned_alpha,
                self.on_policy_mixed_kl_alpha,
                self.on_policy_clipped_rkl_alpha,
                self.on_policy_target_margin_alpha,
                self.on_policy_preservation_alpha,
                self.on_policy_regression_alpha,
                self.on_policy_greedy_preservation_alpha,
                self.on_policy_reset_replay_alpha,
                self.on_policy_preference_alpha,
                self.carh_gate_calibration_alpha,
                self.carh_gate_noop_alpha,
                self.branch_value_alpha,
                self.parallel_refiner_hazard_alpha,
                self.parallel_refiner_recovery_alpha,
                self.parallel_refiner_preservation_alpha,
                float(self.refiner_advantage_mode != "none"),
                float(self.branch_value_mode == "diagnostics"),
                float(getattr(self, "multi_teacher_export_sink", None) is not None),
                float(getattr(self, "multi_teacher_oracle_provider", None) is not None),
            )
            <= 0
            or self._total_steps <= 0
        ):
            return 0.0
        # Optional absolute optimizer-step schedule, independent of the
        # Prefix-Full schedule. Existing runs retain the original ratio path.
        start = getattr(self, "on_policy_start_step", None)
        if start is not None:
            blend = (self._global_step - start + 1) / max(1, self.on_policy_ramp_steps)
            return self.on_policy_mix_ratio_max * min(max(blend, 0.0), 1.0)
        progress = self._global_step / self._total_steps
        if progress < self.prefix_credit_warmup_ratio:
            return 0.0
        ramp = max(self.prefix_credit_ramp_ratio, 1e-8)
        return self.on_policy_mix_ratio_max * min(
            max((progress - self.prefix_credit_warmup_ratio) / ramp, 0.0), 1.0
        )

    def _sampled_prefix_memory_scale(self) -> float:
        """Stage the new head after a baseline-only scratch warmup."""
        if self._total_steps <= 0:
            # Direct model calls outside a scheduled training run use the
            # exported/inference value, which is always the full memory path.
            return 1.0
        progress = self._global_step / self._total_steps
        start = self.carh_sampled_prefix_memory_start_ratio
        if progress < start:
            return 0.0
        ramp = self.carh_sampled_prefix_memory_ramp_ratio
        if ramp <= 0:
            return 1.0
        return min(max((progress - start) / ramp, 0.0), 1.0)

    def _refiner_advantage_phase(self) -> tuple[float, float]:
        """Return (gate, distill) scales for calibration/distill/consolidation."""
        if self.refiner_advantage_mode == "none" or self._total_steps <= 0:
            return 0.0, 0.0
        progress = self._global_step / self._total_steps
        if progress >= 1.0 - self.refiner_advantage_consolidation_ratio:
            return 0.0, 0.0
        distill = float(
            self.refiner_advantage_mode == "distill"
            and progress >= self.refiner_advantage_distill_start_ratio
        )
        return 1.0, distill

    def _sample_on_policy_temperatures(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one rollout temperature per proposal path.

        A temperature is held fixed across all draft depths in a path.  This
        matches serving, where a request's sampling parameters do not change
        halfway through a speculative block.  The categorical draw is local to
        the current rank and follows PyTorch's seeded RNG, so checkpoint resume
        retains the trainer's normal reproducibility contract.
        """
        if len(self.on_policy_rollout_temperatures) == 1:
            return (
                torch.full(
                    (batch_size,),
                    self.on_policy_rollout_temperatures[0],
                    device=device,
                    dtype=torch.float32,
                ),
                torch.zeros(batch_size, device=device, dtype=torch.long),
            )
        probabilities = torch.tensor(
            self.on_policy_rollout_temperature_probs,
            device=device,
            dtype=torch.float32,
        )
        choices = torch.multinomial(probabilities, batch_size, replacement=True)
        configured = torch.tensor(
            self.on_policy_rollout_temperatures,
            device=device,
            dtype=torch.float32,
        )
        return configured[choices], choices

    def _on_policy_flatness_power(self) -> float:
        if self.on_policy_flatness_power_max <= 0:
            return 0.0
        if self.on_policy_flatness_warmup_ratio <= 0 or self._total_steps <= 0:
            return self.on_policy_flatness_power_max
        progress = self._global_step / self._total_steps
        return self.on_policy_flatness_power_max * min(
            max(progress / self.on_policy_flatness_warmup_ratio, 0.0), 1.0
        )

    def _dialogue_anchor_features(
        self,
        active: torch.Tensor,
        anchor_positions: torch.Tensor,
        loss_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> dict:
        """Describe single-turn response occupancy at every candidate anchor."""
        response_prefix = loss_mask.gt(0.5).long().cumsum(dim=-1)
        response_total = loss_mask.gt(0.5).sum(dim=-1, keepdim=True).clamp_min(1)
        ordinal = response_prefix.gather(1, anchor_positions)
        fraction = (ordinal.float() - 0.5).clamp_min(0.0) / response_total.float()
        # Opening 10%, early 20%, middle 40%, late 20%, closing 10%.
        phase = (
            fraction.ge(0.10).long()
            + fraction.ge(0.30).long()
            + fraction.ge(0.70).long()
            + fraction.ge(0.90).long()
        )

        lookup = self.dialogue_token_class_lookup
        depth_offsets = torch.arange(
            self.block_size, device=anchor_positions.device
        ).view(1, 1, -1)
        current_positions = (anchor_positions.unsqueeze(-1) + depth_offsets).clamp_max(
            input_ids.size(1) - 1
        )
        next_positions = (current_positions + 1).clamp_max(input_ids.size(1) - 1)
        expanded_ids = input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1)
        current_ids = expanded_ids.gather(2, current_positions).clamp(
            min=0, max=lookup.numel() - 1
        )
        next_ids = expanded_ids.gather(2, next_positions).clamp(
            min=0, max=lookup.numel() - 1
        )
        current_class = lookup[current_ids]
        next_class = lookup[next_ids]
        sequence_ids = input_ids.clamp(min=0, max=lookup.numel() - 1)
        sequence_class = lookup[sequence_ids]
        supervised = loss_mask.gt(0.5)
        han_count = (sequence_class.eq(1) & supervised).sum(dim=-1).float()
        latin_count = (sequence_class.eq(2) & supervised).sum(dim=-1).float()
        han_ratio = han_count / (han_count + latin_count).clamp_min(1.0)
        han_ratio = han_ratio.unsqueeze(-1).expand(-1, anchor_positions.size(1))
        depth_active = active.unsqueeze(-1)
        depth_boundary = (
            current_class.ge(4) | next_class.ge(4)
        ) & depth_active
        depth_script_transition = (
            (
                (current_class.eq(1) & next_class.eq(2))
                | (current_class.eq(2) & next_class.eq(1))
            )
            & depth_active
        )
        return {
            "phase": phase,
            "fraction": fraction,
            "han_ratio": han_ratio,
            "boundary": depth_boundary[..., 0],
            "script_transition": depth_script_transition[..., 0],
            "depth_boundary": depth_boundary,
            "depth_script_transition": depth_script_transition,
        }

    def _dialogue_phase_weights(
        self, active: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        """Allocate configured mass to phases, then uniformly within a phase."""
        masses = active.new_tensor(self.dialogue_phase_masses, dtype=torch.float32)
        phase_counts = torch.stack(
            [(active & phase.eq(index)).sum(dim=-1) for index in range(5)], dim=-1
        ).float()
        per_phase = masses.unsqueeze(0) / phase_counts.clamp_min(1.0)
        weights = per_phase.gather(1, phase) * active.float()
        weight_sum = weights.sum(dim=-1, keepdim=True)
        fallback = active.float() / active.sum(
            dim=-1, keepdim=True
        ).clamp_min(1).float()
        return torch.where(
            weight_sum > 0,
            weights / weight_sum.clamp_min(1e-12),
            fallback,
        )

    def _dialogue_credit_multiplier(
        self,
        active: torch.Tensor,
        dialogue_features: dict,
        *,
        use_phase: bool,
        use_boundary: bool,
        reference_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return detached block weights with baseline mass preserved per row."""
        active_float = active.float()
        if use_phase:
            weights = self._dialogue_phase_weights(
                active, dialogue_features["phase"]
            ) * active_float.sum(dim=-1, keepdim=True)
        else:
            weights = active_float
        if use_boundary:
            weights = weights * torch.where(
                dialogue_features["boundary"],
                weights.new_tensor(self.dialogue_boundary_boost),
                weights.new_tensor(1.0),
            )
            weights = weights * torch.where(
                dialogue_features["script_transition"],
                weights.new_tensor(self.dialogue_script_transition_boost),
                weights.new_tensor(1.0),
            )
        reference = (
            active_float
            if reference_weight is None
            else reference_weight.detach().float() * active_float
        )
        target_mass = reference.sum(dim=-1, keepdim=True)
        current_mass = (weights * reference).sum(dim=-1, keepdim=True)
        multiplier = weights * target_mass / current_mass.clamp_min(1e-12)
        return torch.where(active, multiplier, torch.zeros_like(multiplier)).detach()

    def _select_on_policy_blocks(
        self,
        active: torch.Tensor,
        block_hazard: Optional[torch.Tensor],
        block_flatness: Optional[torch.Tensor],
        dialogue_features: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select one valid proposal block per row without first-block bias."""
        active_rows = active.any(dim=-1)
        safe_active = active.clone()
        safe_active[~active_rows, 0] = True
        uniform = safe_active.float()
        uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if self.on_policy_anchor_sampling == "first":
            return safe_active.float().argmax(dim=-1), uniform
        if self.on_policy_anchor_sampling == "uniform":
            weights = uniform
        else:
            if block_hazard is None:
                hazard = torch.ones_like(uniform)
            else:
                hazard = block_hazard.detach().float().clamp(0.0, 1.0)
            weights = (hazard + 1e-4).pow(self.on_policy_hazard_power)
            if self.on_policy_anchor_sampling in {
                "hazard-flatness",
                "frontier-value",
                "dialogue-frontier",
            }:
                if block_flatness is None:
                    flatness = torch.ones_like(uniform)
                else:
                    flatness = block_flatness.detach().float().clamp(0.0, 1.0)
                learnability = (
                    flatness
                    if self.on_policy_anchor_sampling == "hazard-flatness"
                    else 1.0 - flatness
                )
                weights = weights * (learnability + 1e-4).pow(
                    self._on_policy_flatness_power()
                )
            exploration_base = uniform
            if self.on_policy_anchor_sampling == "dialogue-frontier":
                if dialogue_features is None:
                    raise ValueError(
                        "dialogue-frontier sampling requires dialogue anchor features"
                    )
                phase_base = self._dialogue_phase_weights(
                    safe_active, dialogue_features["phase"]
                )
                boundary_multiplier = torch.where(
                    dialogue_features["boundary"],
                    weights.new_tensor(self.dialogue_boundary_boost),
                    weights.new_tensor(1.0),
                )
                script_multiplier = torch.where(
                    dialogue_features["script_transition"],
                    weights.new_tensor(self.dialogue_script_transition_boost),
                    weights.new_tensor(1.0),
                )
                weights = (
                    weights * phase_base * boundary_multiplier * script_multiplier
                )
                exploration_base = phase_base
            weights = weights * safe_active.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            exploration = self.on_policy_uniform_exploration
            weights = (1.0 - exploration) * weights + exploration * exploration_base
        return torch.multinomial(weights, 1).squeeze(-1), weights

    def _distributional_opsc_blend(self) -> float:
        """Blend stable per-depth overlap into prefix-survival optimization."""
        ratio = self.on_policy_distributional_cold_start_ratio
        if ratio <= 0 or self._total_steps <= 0:
            return 1.0
        progress = self._global_step / self._total_steps
        return min(max(progress / ratio, 0.0), 1.0)

    @torch.no_grad()
    def _temperature_hazard_credit_weights(
        self,
        first_rejection: torch.Tensor,
        alive_before: torch.Tensor,
        valid: torch.Tensor,
        temperature_choices: torch.Tensor,
        active_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Return detached per-row/depth credit from verifier rejection hazard.

        Counts are synchronized before the EMA update so every data-parallel
        rank applies the same weights.  Each temperature is normalized by its
        mean observed depth hazard: this reallocates credit without changing
        the average auxiliary-loss scale.
        """
        if not self.on_policy_temperature_hazard_credit:
            return valid.new_ones(valid.shape, dtype=torch.float32)

        n_temperatures = len(self.on_policy_rollout_temperatures)
        numerator = torch.zeros(
            n_temperatures,
            self.block_size,
            device=valid.device,
            dtype=torch.float32,
        )
        denominator = torch.zeros_like(numerator)
        row_mask = active_rows.unsqueeze(-1)
        numerator.index_add_(
            0,
            temperature_choices,
            (first_rejection & row_mask).float(),
        )
        denominator.index_add_(
            0,
            temperature_choices,
            (alive_before & valid.bool() & row_mask).float(),
        )
        if dist.is_initialized():
            dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)

        observed = numerator / denominator.clamp_min(1.0)
        available = denominator.gt(0)
        initialized = self.on_policy_temperature_hazard_initialized
        ema_buffer = self.on_policy_temperature_hazard_ema
        # FSDP mixed precision casts floating-point buffers to buffer_dtype
        # (BF16 in the training scripts). Keep the EMA arithmetic and indexed
        # updates in FP32, then copy the result back with an explicit cast.
        # Without this, the first post-warmup observation attempts to assign an
        # FP32 statistic into a BF16 buffer and torch.index_put_ raises.
        ema = ema_buffer.float()
        first_observation = available & ~initialized
        ema[first_observation] = observed[first_observation]
        continuing = available & initialized
        decay = self.on_policy_hazard_ema_decay
        ema[continuing] = (
            decay * ema[continuing] + (1.0 - decay) * observed[continuing]
        )
        initialized.logical_or_(available)
        ema_buffer.copy_(ema.to(dtype=ema_buffer.dtype))

        configured_probs = torch.tensor(
            self.on_policy_rollout_temperature_probs,
            device=valid.device,
            dtype=torch.float32,
        ).unsqueeze(-1)
        fallback = (ema * configured_probs).sum(dim=0)
        selected = ema[temperature_choices]
        selected_initialized = initialized[temperature_choices]
        selected = torch.where(
            selected_initialized,
            selected,
            fallback.unsqueeze(0),
        )
        valid_float = valid.float()
        mean_hazard = (selected * valid_float).sum(dim=-1, keepdim=True) / (
            valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        )
        return (selected / mean_hazard.clamp_min(1e-4)).clamp(
            self.on_policy_hazard_weight_min,
            self.on_policy_hazard_weight_max,
        )

    @torch.no_grad()
    def _marginal_survival_credit_weights(
        self,
        *,
        conditional_acceptance: torch.Tensor,
        valid: torch.Tensor,
        active_rows: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Return detached reachability-times-continuation value credit.

        For conditional acceptance estimates ``a_d``, the derivative of the
        expected accepted length with respect to ``a_d`` is the probability of
        reaching depth ``d`` multiplied by the value of the continuation that
        becomes available when that position survives.  The result is clipped
        and mass-normalized so it reallocates strict-v2's existing auxiliary
        budget instead of silently increasing it.
        """
        one = valid.new_ones(valid.shape, dtype=torch.float32)
        zero = valid.new_zeros((), dtype=torch.float32)
        if not self.on_policy_marginal_value_credit:
            return one, {"on_policy_marginal_value_enabled": zero}
        if conditional_acceptance.shape != valid.shape:
            raise ValueError(
                "conditional_acceptance and valid must have the same shape"
            )
        if active_rows.shape != valid.shape[:1]:
            raise ValueError("active_rows must have shape [batch]")

        mask = valid.bool() & active_rows.unsqueeze(-1)
        acceptance = torch.where(
            mask,
            conditional_acceptance.float().clamp(0.0, 1.0),
            torch.ones_like(one),
        )
        reach_before = torch.cat(
            [
                torch.ones_like(acceptance[:, :1]),
                torch.cumprod(acceptance[:, :-1], dim=-1),
            ],
            dim=-1,
        )
        continuation = torch.ones_like(acceptance)
        for depth in range(acceptance.size(-1) - 2, -1, -1):
            continuation[:, depth] = (
                1.0
                + acceptance[:, depth + 1] * continuation[:, depth + 1]
            )
        raw = reach_before * continuation
        active = mask.float()
        active_mass = active.sum().clamp_min(1.0)
        raw_mean = (raw * active).sum() / active_mass
        bounded = (raw / raw_mean.clamp_min(1e-6)).clamp(
            self.on_policy_marginal_value_weight_min,
            self.on_policy_marginal_value_weight_max,
        )
        bounded_mean = (bounded * active).sum() / active_mass
        weights = bounded / bounded_mean.clamp_min(1e-6)
        weights = torch.where(mask, weights, one)
        diagnostics = {
            "on_policy_marginal_value_enabled": zero.new_tensor(1.0),
            "on_policy_marginal_value_credit_mean": (
                (weights * active).sum() / active_mass
            ).detach(),
            "on_policy_marginal_value_credit_min": (
                torch.where(mask, weights, torch.full_like(weights, float("inf")))
                .amin()
                .detach()
            ),
            "on_policy_marginal_value_credit_max": (
                torch.where(mask, weights, torch.zeros_like(weights)).amax().detach()
            ),
            "on_policy_marginal_value_reach_mean": (
                (reach_before * active).sum() / active_mass
            ).detach(),
            "on_policy_marginal_value_continuation_mean": (
                (continuation * active).sum() / active_mass
            ).detach(),
            "on_policy_marginal_value_acceptance_mean": (
                (acceptance * active).sum() / active_mass
            ).detach(),
        }
        return weights.detach(), diagnostics

    @torch.no_grad()
    def _language_hazard_credit_weights(
        self,
        first_rejection: torch.Tensor,
        alive_before: torch.Tensor,
        valid: torch.Tensor,
        active_rows: torch.Tensor,
        han_ratio: torch.Tensor,
        predecessor_ids: torch.Tensor,
        proposal_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Allocate fixed credit mass to Chinese high-hazard verifier states.

        The EMA is indexed by response language, realized proposal-transition
        class and depth.  Weight normalization is global across data-parallel
        workers, so enabling the router reallocates strict-v2 credit instead of
        silently increasing the auxiliary-loss scale.
        """
        one = valid.new_ones(valid.shape, dtype=torch.float32)
        zero = valid.new_zeros((), dtype=torch.float32)
        if not self.on_policy_language_hazard_credit:
            return one, {"on_policy_language_hazard_enabled": zero}

        lookup = self.dialogue_token_class_lookup
        max_token_id = lookup.numel() - 1
        previous_class = lookup[predecessor_ids.clamp(0, max_token_id)]
        proposal_class = lookup[proposal_ids.clamp(0, max_token_id)]
        boundary = previous_class.ge(4) | proposal_class.ge(4)
        script_transition = (
            (previous_class.eq(1) & proposal_class.eq(2))
            | (previous_class.eq(2) & proposal_class.eq(1))
        )
        transition_group = torch.where(
            script_transition,
            torch.full_like(previous_class, 2, dtype=torch.long),
            boundary.long(),
        )
        language_group = han_ratio.ge(
            self.on_policy_language_han_threshold
        ).long()

        depth_count = valid.size(-1)
        depth_index = torch.arange(depth_count, device=valid.device).view(1, -1)
        state_index = (
            (language_group.unsqueeze(-1) * 3 + transition_group) * self.block_size
            + depth_index
        )
        row_mask = active_rows.unsqueeze(-1)
        numerator = torch.zeros(
            2 * 3 * self.block_size, device=valid.device, dtype=torch.float32
        )
        denominator = torch.zeros_like(numerator)
        numerator.scatter_add_(
            0,
            state_index.reshape(-1),
            (first_rejection & row_mask).float().reshape(-1),
        )
        denominator.scatter_add_(
            0,
            state_index.reshape(-1),
            (alive_before & valid.bool() & row_mask).float().reshape(-1),
        )
        if dist.is_initialized():
            dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        numerator = numerator.view(2, 3, self.block_size)
        denominator = denominator.view(2, 3, self.block_size)
        observed = numerator / denominator.clamp_min(1.0)
        available = denominator.gt(0)

        ema_buffer = self.on_policy_language_hazard_ema
        initialized = self.on_policy_language_hazard_initialized
        ema = ema_buffer.float()
        first_observation = available & ~initialized
        ema[first_observation] = observed[first_observation]
        continuing = available & initialized
        decay = self.on_policy_language_hazard_ema_decay
        ema[continuing] = (
            decay * ema[continuing] + (1.0 - decay) * observed[continuing]
        )
        initialized.logical_or_(available)
        ema_buffer.copy_(ema.to(dtype=ema_buffer.dtype))

        global_numerator = numerator.sum(dim=(0, 1))
        global_denominator = denominator.sum(dim=(0, 1))
        global_observed = global_numerator / global_denominator.clamp_min(1.0)
        global_available = global_denominator.gt(0)
        global_buffer = self.on_policy_language_global_hazard_ema
        global_initialized = self.on_policy_language_global_hazard_initialized
        global_ema = global_buffer.float()
        first_global = global_available & ~global_initialized
        global_ema[first_global] = global_observed[first_global]
        continuing_global = global_available & global_initialized
        global_ema[continuing_global] = (
            decay * global_ema[continuing_global]
            + (1.0 - decay) * global_observed[continuing_global]
        )
        global_initialized.logical_or_(global_available)
        global_buffer.copy_(global_ema.to(dtype=global_buffer.dtype))

        flat_ema = ema.reshape(-1)
        flat_initialized = initialized.reshape(-1)
        selected = flat_ema[state_index]
        selected_initialized = flat_initialized[state_index]
        fallback = global_ema[:depth_count].unsqueeze(0).expand_as(selected)
        selected = torch.where(selected_initialized, selected, fallback)
        hazard_ratio = torch.where(
            fallback.gt(1e-4),
            selected / fallback.clamp_min(1e-4),
            torch.ones_like(selected),
        )
        raw_weight = 1.0 + han_ratio.unsqueeze(-1) * (hazard_ratio - 1.0)
        raw_weight = raw_weight.clamp(
            self.on_policy_language_hazard_weight_min,
            self.on_policy_language_hazard_weight_max,
        )

        reference = (valid.bool() & row_mask).float()
        target_mass = reference.sum()
        weighted_mass = (raw_weight * reference).sum()
        if dist.is_initialized():
            dist.all_reduce(target_mass, op=dist.ReduceOp.SUM)
            dist.all_reduce(weighted_mass, op=dist.ReduceOp.SUM)
        credit = raw_weight * target_mass / weighted_mass.clamp_min(1e-6)
        credit = torch.where(reference.bool(), credit, torch.ones_like(credit))

        active_count = active_rows.sum().clamp_min(1)
        diagnostics = {
            "on_policy_language_hazard_enabled": zero.new_ones(()),
            "on_policy_language_han_ratio": (
                (han_ratio * active_rows.float()).sum() / active_count
            ).detach(),
            "on_policy_language_chinese_row_rate": (
                (language_group.bool() & active_rows).sum().float() / active_count
            ).detach(),
            "on_policy_language_hazard_credit_mean": (
                (credit * reference).sum() / reference.sum().clamp_min(1.0)
            ).detach(),
            "on_policy_language_boundary_rate": (
                (boundary & reference.bool()).sum().float()
                / reference.sum().clamp_min(1.0)
            ).detach(),
            "on_policy_language_script_transition_rate": (
                (script_transition & reference.bool()).sum().float()
                / reference.sum().clamp_min(1.0)
            ).detach(),
        }
        return credit.detach(), diagnostics

    @torch.no_grad()
    def _temperature_pareto_credit_weights(
        self,
        survived: torch.Tensor,
        valid: torch.Tensor,
        temperature_choices: torch.Tensor,
        active_rows: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Upweight rollout temperatures with the largest survival shortfall.

        The statistic is synchronized before its EMA update so all data-parallel
        ranks optimize the same temperature-level objective.  We normalize the
        selected weights under the configured sampling distribution, preserving
        the average auxiliary-loss scale while preventing an already-strong
        greedy group from dominating short continuation runs.
        """
        n_temperatures = len(self.on_policy_rollout_temperatures)
        if not self.on_policy_pareto_credit:
            weights = valid.new_ones(active_rows.shape, dtype=torch.float32)
            return weights, {
                "on_policy_pareto_enabled": valid.new_zeros((), dtype=torch.float32)
            }

        valid_count = valid.float().sum(dim=-1).clamp_min(1.0)
        accepted_fraction = (
            survived.float().sum(dim=-1) / valid_count
        )
        shortfall = (1.0 - accepted_fraction) * active_rows.float()
        numerator = torch.zeros(
            n_temperatures, device=valid.device, dtype=torch.float32
        )
        denominator = torch.zeros_like(numerator)
        numerator.index_add_(0, temperature_choices, shortfall)
        denominator.index_add_(0, temperature_choices, active_rows.float())
        if dist.is_initialized():
            dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)

        observed = numerator / denominator.clamp_min(1.0)
        available = denominator.gt(0)
        initialized = self.on_policy_temperature_shortfall_initialized
        ema_buffer = self.on_policy_temperature_shortfall_ema
        ema = ema_buffer.float()
        first_observation = available & ~initialized
        ema[first_observation] = observed[first_observation]
        continuing = available & initialized
        decay = self.on_policy_pareto_ema_decay
        ema[continuing] = (
            decay * ema[continuing] + (1.0 - decay) * observed[continuing]
        )
        initialized.logical_or_(available)
        ema_buffer.copy_(ema.to(dtype=ema_buffer.dtype))

        configured_probs = torch.tensor(
            self.on_policy_rollout_temperature_probs,
            device=valid.device,
            dtype=torch.float32,
        )
        centered = ema - (configured_probs * ema).sum()
        group_weights = torch.exp(
            centered / self.on_policy_pareto_temperature
        ).clamp(
            self.on_policy_pareto_weight_min,
            self.on_policy_pareto_weight_max,
        )
        group_weights = group_weights / (
            configured_probs * group_weights
        ).sum().clamp_min(1e-6)
        selected = group_weights[temperature_choices]
        selected = torch.where(active_rows, selected, torch.zeros_like(selected))

        diagnostics = {
            "on_policy_pareto_enabled": valid.new_ones((), dtype=torch.float32),
            "on_policy_pareto_weight_mean": (
                selected.sum() / active_rows.sum().clamp_min(1)
            ).detach(),
        }
        for index, temperature in enumerate(self.on_policy_rollout_temperatures):
            label = str(temperature).replace(".", "p")
            diagnostics[f"on_policy_pareto_shortfall_t{label}"] = ema[index].detach()
            diagnostics[f"on_policy_pareto_weight_t{label}"] = (
                group_weights[index].detach()
            )
        return selected, diagnostics

    def _carh_gate_calibration_scale(self) -> float:
        """Delay gate labels until the warm-start residual has stabilized."""
        if max(self.carh_gate_calibration_alpha, self.carh_gate_noop_alpha) <= 0:
            return 0.0
        if self._total_steps <= 0:
            return 0.0
        progress = self._global_step / self._total_steps
        warmup = self.carh_gate_calibration_warmup_ratio
        if progress < warmup:
            return 0.0
        remaining = max(1.0 - warmup, 1e-8)
        return min(max((progress - warmup) / min(0.10, remaining), 0.0), 1.0)

    @staticmethod
    def _adaptive_distributional_mask(
        target_topk_probs: torch.Tensor,
        min_top_k: int,
        mass_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the smallest prefix of sorted target candidates reaching mass.

        A zero threshold preserves the legacy fixed Top-k objective.  When the
        configured maximum cannot reach the requested mass, every available
        candidate is retained and ``cap_limited`` records the truncation.
        """
        max_top_k = target_topk_probs.size(-1)
        if mass_threshold <= 0:
            keep = torch.ones_like(target_topk_probs, dtype=torch.bool)
            effective_k = torch.full(
                target_topk_probs.shape[:-1],
                max_top_k,
                dtype=torch.long,
                device=target_topk_probs.device,
            )
            retained_mass = target_topk_probs.sum(dim=-1)
            cap_limited = torch.zeros_like(effective_k, dtype=torch.bool)
            return keep, effective_k, retained_mass, cap_limited

        cumulative = target_topk_probs.cumsum(dim=-1)
        reached = cumulative.ge(float(mass_threshold))
        reached_any = reached.any(dim=-1)
        first_reached = reached.to(torch.int64).argmax(dim=-1) + 1
        effective_k = torch.where(
            reached_any,
            first_reached,
            torch.full_like(first_reached, max_top_k),
        ).clamp(min=max(1, int(min_top_k)), max=max_top_k)
        ranks = torch.arange(
            1, max_top_k + 1, device=target_topk_probs.device
        ).view(*((1,) * (target_topk_probs.ndim - 1)), max_top_k)
        keep = ranks.le(effective_k.unsqueeze(-1))
        retained_mass = (target_topk_probs * keep).sum(dim=-1)
        return keep, effective_k, retained_mass, ~reached_any

    def _sample_on_policy_tokens(
        self, logits: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        """Sample proposal tokens with a per-row temperature schedule.

        Temperature zero is exactly greedy and therefore preserves the legacy
        OPSC rollout. Positive temperatures sample from the full corrected CARH
        distribution.  Sampling is detached because proposal ids define the
        on-policy state; gradients flow through the target-scored margin loss.
        """
        detached = logits.detach().float()
        result = detached.argmax(dim=-1)
        stochastic = temperatures.gt(0)
        if stochastic.any():
            log_probs = processed_log_probs(
                detached[stochastic],
                temperatures[stochastic],
                top_k=self.on_policy_deployment_top_k,
                top_p=self.on_policy_deployment_top_p,
            )
            probabilities = log_probs.exp()
            sampled = torch.multinomial(
                probabilities.reshape(-1, probabilities.size(-1)), 1
            ).squeeze(-1).view(probabilities.shape[:-1])
            result = result.clone()
            result[stochastic] = sampled
        return result

    def _branch_value_objective(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        anchors: torch.Tensor,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        proposals: torch.Tensor,
        rollout_logits: torch.Tensor,
        first_rejection: torch.Tensor,
        active_rows: torch.Tensor,
        ce_num: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Target-score sibling continuations at the real first rejection.

        The branch teacher is training-only.  It never injects the gold token:
        siblings are strict serving Top-M candidates from the student's logits.
        """
        zero = rollout_logits.new_zeros((), dtype=torch.float32)
        if self.branch_value_mode == "none" or self.on_policy_scorer is None:
            return zero, {}
        has_boundary = first_rejection.any(dim=-1) & active_rows
        if not has_boundary.any():
            return zero, {"branch_value_scored_rows": zero}

        batch_size, depth_count, _ = rollout_logits.shape
        top_m = self.branch_value_top_m
        boundary_depth = first_rejection.float().argmax(dim=-1)
        rows = torch.arange(batch_size, device=rollout_logits.device)
        boundary_logits = rollout_logits[rows, boundary_depth]
        candidate_ids = boundary_logits.detach().topk(top_m, dim=-1).indices

        with torch.no_grad():
            branch_proposals = proposals[:, None, :].expand(
                -1, top_m, -1
            ).clone()
            predecessor = input_ids[rows, anchors][:, None].expand(-1, top_m)
            for depth in range(depth_count):
                before = boundary_depth.gt(depth).unsqueeze(-1)
                at = boundary_depth.eq(depth).unsqueeze(-1)
                main_token = proposals[:, depth].unsqueeze(-1).expand(-1, top_m)
                if depth == 0:
                    generated = main_token
                else:
                    branch_hidden = hidden_states[:, depth, :][:, None, :].expand(
                        -1, top_m, -1
                    )
                    branch_base = base_logits[:, depth, :][:, None, :].expand(
                        -1, top_m, -1
                    )
                    if self.draft_model.recall_correction is not None:
                        depth_ids = torch.full_like(predecessor, depth)
                        corrected_hidden = self.draft_model.apply_recall_correction(
                            branch_hidden, predecessor, depth_ids
                        )
                        branch_base = self.lm_head(
                            corrected_hidden.reshape(-1, corrected_hidden.size(-1))
                        ).view(batch_size, top_m, -1)
                    residual = self.draft_model.markov_head.compute_step_bias(
                        predecessor,
                        branch_hidden,
                        depth_idx=depth,
                        **_carh_predecessor_context_kwargs(
                            self.draft_model.markov_head,
                            hidden_states[:, None, :, :].expand(-1, top_m, -1, -1),
                            depth,
                        ),
                    )
                    generated = (branch_base + residual).argmax(dim=-1)
                selected = torch.where(
                    before,
                    main_token,
                    torch.where(at, candidate_ids, generated),
                )
                branch_proposals[:, :, depth] = selected
                predecessor = selected

            verifier_output = self.on_policy_scorer(
                input_ids,
                attention_mask,
                anchors,
                branch_proposals,
                has_boundary,
            )
            verifier_ids = (
                verifier_output[0]
                if isinstance(verifier_output, tuple)
                else verifier_output
            ).to(branch_proposals.device)
            matches = branch_proposals.eq(verifier_ids)
            offsets = torch.arange(depth_count, device=matches.device).view(1, 1, -1)
            in_horizon = offsets.ge(boundary_depth[:, None, None]) & offsets.lt(
                boundary_depth[:, None, None] + self.branch_value_horizon
            )
            branch_alive = torch.ones(
                batch_size, top_m, device=matches.device, dtype=torch.bool
            )
            values = torch.zeros(
                batch_size, top_m, device=matches.device, dtype=torch.float32
            )
            for depth in range(depth_count):
                active_depth = boundary_depth.le(depth)
                branch_alive = branch_alive & (
                    ~active_depth.unsqueeze(-1) | matches[:, :, depth]
                )
                values = values + (
                    branch_alive & in_horizon[:, :, depth]
                ).float()

        student_scores = boundary_logits.gather(-1, candidate_ids).float()
        teacher_probs = torch.softmax(
            values / self.branch_value_temperature, dim=-1
        )
        per_row = -(
            teacher_probs * torch.log_softmax(student_scores, dim=-1)
        ).sum(dim=-1)
        progress = (
            self._global_step / self._total_steps if self._total_steps > 0 else 0.0
        )
        active_phase = (
            progress >= self.branch_value_warmup_ratio
            and progress < 1.0 - self.branch_value_consolidation_ratio
        )
        phase_scale = float(active_phase)
        raw = (
            phase_scale
            * self.branch_value_alpha
            * per_row[has_boundary].sum()
        )
        if self.branch_value_mode == "distill" and active_phase:
            objective, budget_scale = self._budget_auxiliary_loss(
                raw, self.branch_value_loss_budget, ce_num
            )
        else:
            objective = zero
            budget_scale = zero

        greedy_value = values[:, 0]
        best_value, best_index = values.max(dim=-1)
        denom = has_boundary.sum().clamp_min(1)
        diagnostics = {
            "branch_value_scored_rows": has_boundary.sum().detach(),
            "branch_value_oracle_gain": (
                (best_value - greedy_value)[has_boundary].sum() / denom
            ).detach(),
            "branch_value_recovery_rate": (
                best_value[has_boundary].gt(greedy_value[has_boundary]).sum().float()
                / denom
            ).detach(),
            "branch_value_greedy_optimal_rate": (
                best_index[has_boundary].eq(0).sum().float() / denom
            ).detach(),
            "branch_value_teacher_entropy": (
                -(
                    teacher_probs.clamp_min(1e-12).log() * teacher_probs
                ).sum(dim=-1)[has_boundary].mean()
            ).detach(),
            "branch_value_budget_scale": budget_scale.detach(),
            "branch_value_phase_scale": zero.new_tensor(phase_scale),
        }
        return objective, diagnostics

    def _hazard_adaptive_parallel_refinement(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        rollout_temperatures: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Run one or two shared-weight proposal refinement rounds.

        Intermediate proposal selection is deliberately detached.  Only the
        final shared LM-head projection retains a vocabulary-sized autograd
        graph, keeping two-round training close to one-round memory use.
        """
        refiner = getattr(self.draft_model, "parallel_refiner", None)
        if refiner is None:
            if rollout_temperatures is None:
                proposals = logits.detach().argmax(dim=-1)
            else:
                proposals = self._sample_on_policy_tokens(
                    logits.detach(), rollout_temperatures
                )
            return logits, proposals, None, None

        def sample(current_logits: torch.Tensor) -> torch.Tensor:
            if rollout_temperatures is None:
                return current_logits.argmax(dim=-1)
            return self._sample_on_policy_tokens(
                current_logits, rollout_temperatures
            )

        original_logits = logits
        proposals = sample(original_logits.detach())
        accumulated_delta = torch.zeros_like(hidden_states)
        hazard_logits = None
        gate = None
        for iteration_idx in range(refiner.max_steps):
            predecessor_ids = torch.cat(
                [anchor_token_ids.unsqueeze(-1), proposals[..., :-1]], dim=-1
            )
            delta, hazard_logits, gate = self.draft_model.apply_parallel_refiner(
                hidden_states,
                predecessor_ids,
                iteration_idx,
                rollout_temperatures=rollout_temperatures,
            )
            accumulated_delta = accumulated_delta + delta
            if iteration_idx + 1 < refiner.max_steps:
                with torch.no_grad():
                    intermediate_residual = self.lm_head(
                        accumulated_delta.detach().reshape(
                            -1, accumulated_delta.size(-1)
                        )
                    ).view_as(original_logits)
                    proposals = sample(
                        original_logits.detach() + intermediate_residual
                    )

        residual_logits = self.lm_head(
            accumulated_delta.reshape(-1, accumulated_delta.size(-1))
        ).view_as(original_logits)
        refined_logits = original_logits + residual_logits
        proposals = sample(refined_logits.detach())
        return refined_logits, proposals, hazard_logits, gate

    @staticmethod
    def _continuation_survival_value(accepted: torch.Tensor) -> torch.Tensor:
        """Number of consecutively accepted tokens starting at every depth."""
        values = torch.zeros_like(accepted, dtype=torch.float32)
        running = torch.zeros_like(values[:, 0])
        for depth in range(accepted.size(-1) - 1, -1, -1):
            running = accepted[:, depth].float() * (1.0 + running)
            values[:, depth] = running
        return values

    def _refiner_advantage_objective(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        base_accepted: torch.Tensor,
        teacher_accepted: torch.Tensor,
        valid: torch.Tensor,
        advantage_class_logits: Optional[torch.Tensor],
        advantage_value_prediction: Optional[torch.Tensor],
        rollout_temperatures: torch.Tensor,
        ce_num: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Learn only from HAPR corrections with positive verifier advantage."""
        zero = student_logits.new_zeros((), dtype=torch.float32)
        gate_scale, distill_scale = self._refiner_advantage_phase()
        if advantage_class_logits is None or gate_scale <= 0:
            return zero, {
                "adv_teacher_phase_gate": zero.new_tensor(gate_scale),
                "adv_teacher_phase_distill": zero.new_tensor(distill_scale),
            }

        with torch.no_grad():
            base_survived = torch.cumprod(
                base_accepted.to(torch.int32), dim=-1
            ).bool()
            teacher_survived = torch.cumprod(
                teacher_accepted.to(torch.int32), dim=-1
            ).bool()
            alive_before = torch.cat(
                [
                    torch.ones_like(base_survived[:, :1]),
                    base_survived[:, :-1],
                ],
                dim=-1,
            ) & valid.bool()
            base_value = self._continuation_survival_value(base_accepted)
            teacher_value = self._continuation_survival_value(teacher_accepted)
            advantage = (teacher_value - base_value).clamp(
                min=-self.refiner_advantage_value_clip,
                max=self.refiner_advantage_value_clip,
            )
            positive = alive_before & advantage.gt(
                self.refiner_advantage_threshold
            )
            nonpositive = alive_before & ~positive
            base_length = base_survived.sum(dim=-1).float()
            teacher_length = teacher_survived.sum(dim=-1).float()

        alive_weight = alive_before.float()
        positive_target = positive.float()
        gate_bce = F.binary_cross_entropy_with_logits(
            advantage_class_logits.float(), positive_target, reduction="none"
        )
        pos_mass = positive_target.sum()
        neg_mass = nonpositive.sum().float()
        total_mass = pos_mass + neg_mass
        pos_scale = torch.where(
            pos_mass > 0,
            0.5 * total_mass / pos_mass.clamp_min(1),
            torch.zeros_like(pos_mass),
        )
        neg_scale = torch.where(
            neg_mass > 0,
            0.5 * total_mass / neg_mass.clamp_min(1),
            torch.zeros_like(neg_mass),
        )
        balanced = positive_target * pos_scale + nonpositive.float() * neg_scale
        gate_raw = self.refiner_advantage_gate_alpha * (gate_bce * balanced).sum()
        regression_raw = zero
        if advantage_value_prediction is not None:
            regression = F.smooth_l1_loss(
                advantage_value_prediction.float(), advantage, reduction="none"
            )
            regression_raw = self.refiner_advantage_regression_alpha * (
                regression * alive_weight
            ).sum()

        distill_raw = zero
        gate_weight = torch.zeros_like(advantage, dtype=torch.float32)
        if distill_scale > 0 and positive.any():
            top_k = min(
                self.refiner_advantage_distill_top_k,
                teacher_logits.size(-1),
            )
            teacher_values, teacher_ids = torch.topk(
                teacher_logits.detach().float(), k=top_k, dim=-1
            )
            student_values = student_logits.float().gather(-1, teacher_ids)
            temperature = self.refiner_advantage_distill_temperature
            teacher_prob = torch.softmax(teacher_values / temperature, dim=-1)
            student_log_prob = torch.log_softmax(
                student_values / temperature, dim=-1
            )
            kl = F.kl_div(
                student_log_prob, teacher_prob, reduction="none"
            ).sum(dim=-1) * (temperature * temperature)
            gain_weight = advantage.clamp_min(0.0)
            gate_probability = torch.sigmoid(
                advantage_class_logits.detach().float()
            )
            gate_floor = self.refiner_advantage_gate_distill_min_probability
            if gate_floor > 0:
                gate_weight = (
                    (gate_probability - gate_floor) / (1.0 - gate_floor)
                ).clamp(min=0.0, max=1.0)
            else:
                gate_weight = torch.ones_like(gate_probability)
            distill_raw = (
                self.refiner_advantage_distill_alpha
                * distill_scale
                * (kl * positive.float() * gain_weight * gate_weight).sum()
            )

        # For states without positive teacher evidence, make CARH's verifier
        # target margin harder to reduce.  This is a useful proximal constraint
        # even though the anchor distribution is the current student itself.
        # On states without positive teacher evidence, preserve a confident
        # student decision instead of pulling it toward the refiner.
        top2 = torch.topk(student_logits.float(), k=2, dim=-1).values
        current_margin = top2[..., 0] - top2[..., 1]
        preservation_penalty = F.softplus(
            (
                self.parallel_refiner_margin_floor
                - current_margin
            )
            / self.parallel_refiner_temperature
        )
        preservation_raw = (
            self.refiner_advantage_preservation_alpha
            * distill_scale
            * (preservation_penalty * nonpositive.float()).sum()
        )
        # Greedy preservation is deliberately verifier-confirmed: protect an
        # already accepted T=0 decision, including the important cases where
        # the frozen refiner tries to replace that correct token. This prevents
        # high-temperature distillation from flattening stable greedy margins.
        student_ids = student_logits.detach().argmax(dim=-1)
        teacher_ids = teacher_logits.detach().argmax(dim=-1)
        greedy_rows = rollout_temperatures.float().le(1e-6).unsqueeze(-1)
        greedy_preservation_mask = (
            alive_before
            & base_accepted.bool()
            & greedy_rows
        )
        greedy_preservation_penalty = F.softplus(
            (
                self.refiner_advantage_greedy_margin_floor
                - current_margin
            )
            / self.parallel_refiner_temperature
        )
        greedy_preservation_raw = (
            self.refiner_advantage_greedy_preservation_alpha
            * distill_scale
            * (
                greedy_preservation_penalty
                * greedy_preservation_mask.float()
            ).sum()
        )
        raw = (
            gate_scale * (gate_raw + regression_raw)
            + distill_raw
            + preservation_raw
            + greedy_preservation_raw
        )
        objective, budget_scale = self._budget_auxiliary_loss(
            self._effective_on_policy_mix() * raw,
            self.refiner_advantage_loss_budget,
            ce_num,
        )
        with torch.no_grad():
            prediction = torch.sigmoid(advantage_class_logits.float()).ge(0.5)
            stochastic_rows = ~greedy_rows
            diagnostics = {
                "adv_teacher_phase_gate": zero.new_tensor(gate_scale),
                "adv_teacher_phase_distill": zero.new_tensor(distill_scale),
                "adv_teacher_positive_rate": positive.sum().float()
                / alive_before.sum().clamp_min(1),
                "adv_teacher_mean_positive_gain": (
                    (advantage * positive.float()).sum()
                    / positive.sum().clamp_min(1)
                ),
                "adv_teacher_paired_accept_gain": (
                    teacher_length - base_length
                ).mean(),
                "adv_teacher_oracle_accept_gain": (
                    torch.maximum(teacher_length, base_length) - base_length
                ).mean(),
                "adv_teacher_recovery_rate": teacher_length.gt(base_length).float().mean(),
                "adv_teacher_harmful_rate": teacher_length.lt(base_length).float().mean(),
                "adv_gate_precision": (
                    (prediction & positive).sum().float()
                    / (prediction & alive_before).sum().clamp_min(1)
                ),
                "adv_gate_recall": (
                    (prediction & positive).sum().float()
                    / positive.sum().clamp_min(1)
                ),
                "adv_teacher_positive_rate_t0": (
                    (positive & greedy_rows).sum().float()
                    / (alive_before & greedy_rows).sum().clamp_min(1)
                ),
                "adv_teacher_positive_rate_stochastic": (
                    (positive & stochastic_rows).sum().float()
                    / (alive_before & stochastic_rows).sum().clamp_min(1)
                ),
                "adv_gate_precision_t0": (
                    (prediction & positive & greedy_rows).sum().float()
                    / (prediction & alive_before & greedy_rows).sum().clamp_min(1)
                ),
                "adv_gate_recall_t0": (
                    (prediction & positive & greedy_rows).sum().float()
                    / (positive & greedy_rows).sum().clamp_min(1)
                ),
                "adv_gate_precision_stochastic": (
                    (prediction & positive & stochastic_rows).sum().float()
                    / (prediction & alive_before & stochastic_rows).sum().clamp_min(1)
                ),
                "adv_gate_recall_stochastic": (
                    (prediction & positive & stochastic_rows).sum().float()
                    / (positive & stochastic_rows).sum().clamp_min(1)
                ),
                "student_distill_coverage": positive.sum().float()
                / alive_before.sum().clamp_min(1),
                "student_effective_distill_coverage": (
                    (positive.float() * gate_weight).sum()
                    / alive_before.sum().clamp_min(1)
                    if distill_scale > 0
                    else zero
                ),
                "student_recovery_accuracy": (
                    (student_ids.eq(teacher_ids) & positive).sum().float()
                    / positive.sum().clamp_min(1)
                ),
                "student_regression_rate": (
                    (student_ids.ne(teacher_ids) & nonpositive).sum().float()
                    / nonpositive.sum().clamp_min(1)
                ),
                "adv_gate_loss": gate_raw.detach(),
                "adv_value_loss": regression_raw.detach(),
                "adv_distill_loss": distill_raw.detach(),
                "adv_preservation_loss": preservation_raw.detach(),
                "adv_greedy_preservation_loss": greedy_preservation_raw.detach(),
                "adv_greedy_preservation_coverage": (
                    greedy_preservation_mask.sum().float()
                    / (alive_before & greedy_rows).sum().clamp_min(1)
                ),
                "adv_loss_budget_scale": budget_scale.detach(),
            }
        return objective, diagnostics

    @torch.no_grad()
    def _repair_value_weights(self, *, input_ids, attention_mask, anchors, base,
                              hidden, proposals, verifier_ids, first_rejection,
                              valid, rollout_temperatures):
        """Detached T=0 correction value, NOT predicted finite-update gain.

        Re-run the serving CARH recurrence after an oracle boundary correction;
        target-verify that new path. Only first-rejection repair is reweighted.
        Fixed parallel backbone states match this draft's serving recurrence.
        """
        weights = base.new_ones(proposals.shape, dtype=torch.float32)
        if not self.on_policy_repair_value:
            return weights, {}
        boundary = first_rejection.float().argmax(-1)
        eligible = (first_rejection.any(-1) & rollout_temperatures.eq(0)
                    & boundary.lt(self.on_policy_first_rejection_max_depth))
        if not _distributed_any(eligible):
            return weights, {"repair_value_rows": weights.new_zeros(())}
        rows = torch.arange(base.size(0), device=base.device)
        previous = input_ids[rows, anchors]
        tokens = []
        horizon = self.on_policy_repair_value_horizon
        for depth in range(proposals.size(1)):
            step_base = base[:, depth]
            if self.draft_model.recall_correction is not None:
                step_base = self.lm_head(self.draft_model.apply_recall_correction(
                    hidden[:, depth], previous, torch.full_like(previous, depth)))
            logits = step_base + self.draft_model.markov_head.compute_step_bias(
                previous, hidden[:, depth], depth_idx=depth,
                **_carh_predecessor_context_kwargs(
                    self.draft_model.markov_head, hidden, depth
                ),
            )
            suffix = eligible & boundary.lt(depth) & (boundary + horizon).ge(depth)
            token = torch.where(suffix, logits.argmax(-1), proposals[:, depth])
            token = torch.where(eligible & boundary.eq(depth), verifier_ids[:, depth], token)
            tokens.append(token)
            previous = token
        corrected = torch.stack(tokens, -1)
        output = self.on_policy_scorer(input_ids, attention_mask, anchors, corrected, eligible)
        labels = (output[0] if isinstance(output, tuple) else output).to(base.device)
        accepted = corrected.eq(labels) & valid.bool()
        length = accepted.int().cumprod(-1).sum(-1)
        # Includes the repaired boundary; a failed consistency check gets value 0.
        value = (length - boundary).clamp(min=0, max=horizon + 1).float()
        stats = torch.stack([(value * eligible).sum(), eligible.sum().float()])
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        mean = stats[0] / stats[1].clamp_min(1)
        # Half uniform credit, half value credit; cap outliers before a final
        # global normalization. Mean eligible weight is 1, range at most [1/3, 3].
        normalized = torch.where(mean > 0, 0.5 + 0.5 * value / mean.clamp_min(1e-6),
                                 torch.ones_like(value)).clamp(min=0.5, max=1.5)
        weight_sum = (normalized * eligible).sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(weight_sum)
        normalized = normalized / (weight_sum / stats[1].clamp_min(1)).clamp_min(1e-6)
        weights = torch.where(first_rejection & eligible[:, None], normalized[:, None], weights)
        count = eligible.sum().clamp_min(1)
        return weights.detach(), {
            "repair_value_rows": eligible.sum().float(),
            "repair_value_mean": (value * eligible).sum() / count,
            "repair_value_suffix_mean": ((value - 1).clamp_min(0) * eligible).sum() / count,
            "repair_value_invalid_correction": (eligible & value.eq(0)).sum().float(),
        }

    def _reset_replay_scale(self) -> float:
        if self.on_policy_reset_replay_alpha <= 0 or self._total_steps <= 0:
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.on_policy_reset_replay_start_ratio:
            return 0.0
        if self.on_policy_reset_replay_ramp_ratio <= 0:
            return 1.0
        return min(
            max(
                (progress - self.on_policy_reset_replay_start_ratio)
                / self.on_policy_reset_replay_ramp_ratio,
                0.0,
            ),
            1.0,
        )

    def _rollout_preference_objective(
        self,
        *,
        primary_proposals: torch.Tensor,
        alternative_proposals: torch.Tensor,
        primary_logits: torch.Tensor,
        primary_verifier_ids: torch.Tensor,
        alternative_verifier_ids: torch.Tensor,
        valid: torch.Tensor,
        active_rows: torch.Tensor,
        rollout_temperatures: torch.Tensor,
        ce_num: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Prefer the first branch decision with higher verified continuation.

        Two stochastic CARH rollouts share a state until their first differing
        token.  The target scores both complete paths, but the preference loss
        updates only that common-state branch decision.  This avoids assigning
        likelihood credit to incomparable post-divergence hidden states.
        """
        zero = primary_logits.new_zeros((), dtype=torch.float32)
        if self.on_policy_preference_alpha <= 0:
            return zero, {}

        primary_accepted = primary_proposals.eq(primary_verifier_ids) & valid.bool()
        alternative_accepted = alternative_proposals.eq(
            alternative_verifier_ids
        ) & valid.bool()
        primary_value = torch.cumprod(
            primary_accepted.to(torch.int32), dim=-1
        ).sum(dim=-1).float()
        alternative_value = torch.cumprod(
            alternative_accepted.to(torch.int32), dim=-1
        ).sum(dim=-1).float()
        value_delta = alternative_value - primary_value
        value_gap = value_delta.abs()

        differs = primary_proposals.ne(alternative_proposals) & valid.bool()
        has_divergence = differs.any(dim=-1)
        divergence_depth = differs.float().argmax(dim=-1)
        preference_rows = (
            active_rows
            & rollout_temperatures.gt(0)
            & has_divergence
            & value_gap.ge(self.on_policy_preference_value_gap)
        )
        if not preference_rows.any():
            graph_zero = primary_logits.sum() * 0.0
            return graph_zero, {
                "rollout_preference_coverage": zero,
                "rollout_preference_mean_value_gap": zero,
                "rollout_preference_loss": zero,
            }

        rows = torch.arange(primary_logits.size(0), device=primary_logits.device)
        decision_logits = primary_logits[rows, divergence_depth]
        alternative_wins = value_delta.gt(0)
        preferred_tokens = torch.where(
            alternative_wins,
            alternative_proposals[rows, divergence_depth],
            primary_proposals[rows, divergence_depth],
        )
        rejected_tokens = torch.where(
            alternative_wins,
            primary_proposals[rows, divergence_depth],
            alternative_proposals[rows, divergence_depth],
        )
        preferred_logit = decision_logits.gather(
            -1, preferred_tokens.unsqueeze(-1)
        ).squeeze(-1)
        rejected_logit = decision_logits.gather(
            -1, rejected_tokens.unsqueeze(-1)
        ).squeeze(-1)
        preference_margin = (
            preferred_logit - rejected_logit
        ) / self.on_policy_preference_temperature
        gap_weight = value_gap.clamp(max=float(self.block_size)).detach()
        raw = self.on_policy_preference_alpha * (
            F.softplus(-preference_margin)
            * gap_weight
            * preference_rows.float()
        ).sum()
        objective, budget_scale = self._budget_auxiliary_loss(
            self._effective_on_policy_mix() * raw,
            self.on_policy_preference_loss_budget,
            ce_num,
        )
        denominator = active_rows.sum().clamp_min(1)
        preference_count = preference_rows.sum().clamp_min(1)
        diagnostics = {
            "rollout_preference_coverage": (
                preference_rows.sum().float() / denominator
            ).detach(),
            "rollout_preference_mean_value_gap": (
                (value_gap * preference_rows.float()).sum() / preference_count
            ).detach(),
            "rollout_preference_alternative_win_rate": (
                (alternative_wins & preference_rows).sum().float()
                / preference_count
            ).detach(),
            "rollout_preference_loss": raw.detach(),
            "rollout_preference_budget_scale": budget_scale.detach(),
        }
        return objective, diagnostics

    def _reset_replay_objective(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        anchors: torch.Tensor,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        proposals: torch.Tensor,
        verifier_ids: torch.Tensor,
        first_rejection: torch.Tensor,
        valid: torch.Tensor,
        active_rows: torch.Tensor,
        rollout_temperatures: torch.Tensor,
        ce_num: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Replay a short suffix after replacing the first rejection by target.

        This is a training-only counterfactual path.  The accepted draft prefix
        is kept, the verifier token repairs the first rejected slot, and CARH
        generates the following ``horizon`` positions from that corrected
        predecessor.  A second serial verifier call then credits only the
        continuation reachable after the repair.
        """
        zero = base_logits.new_zeros((), dtype=torch.float32)
        phase_scale = self._reset_replay_scale()
        if phase_scale <= 0 or self.on_policy_scorer is None:
            return zero, {"reset_replay_phase_scale": zero.new_tensor(phase_scale)}

        batch_size, depth_count, _ = base_logits.shape
        boundary_depth = first_rejection.float().argmax(dim=-1)
        has_boundary = first_rejection.any(dim=-1) & active_rows
        replay_rows = (
            has_boundary
            & boundary_depth.lt(self.on_policy_reset_replay_max_rejection_depth)
            & boundary_depth.add(1).lt(depth_count)
        )
        # Keep the extra verifier/FSDP path rank-synchronous.  Individual
        # ranks can legitimately have zero eligible rejection frontiers; if
        # any rank has work, all ranks still execute the scorer and the
        # zero-row ranks contribute a graph-connected zero loss.
        if not _distributed_any(replay_rows):
            return zero, {
                "reset_replay_phase_scale": zero.new_tensor(phase_scale),
                "reset_replay_scored_rows": zero,
            }

        predecessor = input_ids[
            torch.arange(batch_size, device=input_ids.device), anchors
        ]
        replay_tokens = []
        replay_step_logits = []
        for depth_idx in range(depth_count):
            step_base = base_logits[:, depth_idx, :]
            if self.draft_model.recall_correction is not None:
                depth_ids = torch.full_like(predecessor, depth_idx)
                corrected_hidden = self.draft_model.apply_recall_correction(
                    hidden_states[:, depth_idx, :], predecessor, depth_ids
                )
                step_base = self.lm_head(corrected_hidden)
            residual = self.draft_model.markov_head.compute_step_bias(
                predecessor,
                hidden_states[:, depth_idx, :],
                depth_idx=depth_idx,
                **_carh_predecessor_context_kwargs(
                    self.draft_model.markov_head, hidden_states, depth_idx
                ),
            )
            step_logits = step_base + residual
            sampled = self._sample_on_policy_tokens(
                step_logits, rollout_temperatures
            )
            before = boundary_depth.gt(depth_idx)
            at_boundary = boundary_depth.eq(depth_idx) & replay_rows
            in_replay_suffix = (
                boundary_depth.lt(depth_idx)
                & boundary_depth.add(self.on_policy_reset_replay_horizon).ge(
                    depth_idx
                )
                & replay_rows
            )
            token = torch.where(
                in_replay_suffix,
                sampled,
                proposals[:, depth_idx],
            )
            token = torch.where(
                at_boundary,
                verifier_ids[:, depth_idx],
                token,
            )
            # ``before`` is documented explicitly even though the fallback is
            # already the original proposal; it guards future path changes.
            token = torch.where(before, proposals[:, depth_idx], token)
            replay_tokens.append(token)
            replay_step_logits.append(step_logits)
            predecessor = token.detach()

        replay_proposals = torch.stack(replay_tokens, dim=-1)
        replay_logits = torch.stack(replay_step_logits, dim=1).float()
        # Compare regenerated continuation against the cheapest meaningful
        # counterfactual: repair the rejected boundary token but retain the
        # original suffix.  Both paths are verified together, so reset replay
        # receives credit only when regeneration adds continuation value.
        baseline_reset_proposals = proposals.clone()
        replay_row_ids = torch.arange(batch_size, device=proposals.device)
        baseline_reset_proposals[
            replay_row_ids, boundary_depth
        ] = verifier_ids[replay_row_ids, boundary_depth]
        candidate_proposals = torch.stack(
            [baseline_reset_proposals, replay_proposals], dim=1
        )
        with torch.no_grad():
            replay_verifier_output = self.on_policy_scorer(
                input_ids,
                attention_mask,
                anchors,
                candidate_proposals,
                replay_rows,
            )
            candidate_verifier_ids = (
                replay_verifier_output[0]
                if isinstance(replay_verifier_output, tuple)
                else replay_verifier_output
            ).to(replay_proposals.device)
            candidate_accepted = candidate_proposals.eq(
                candidate_verifier_ids
            ) & valid[:, None, :].bool()
            depth = torch.arange(depth_count, device=input_ids.device).view(1, -1)
            suffix = (
                depth.gt(boundary_depth.unsqueeze(-1))
                & depth.le(
                    boundary_depth.add(
                        self.on_policy_reset_replay_horizon
                    ).unsqueeze(-1)
                )
                & replay_rows.unsqueeze(-1)
                & valid.bool()
            )
            candidate_alive = torch.ones(
                batch_size,
                2,
                device=input_ids.device,
                dtype=torch.bool,
            )
            candidate_survived = torch.zeros_like(candidate_accepted)
            for depth_idx in range(depth_count):
                in_window = suffix[:, depth_idx].unsqueeze(-1)
                candidate_alive = torch.where(
                    in_window,
                    candidate_alive & candidate_accepted[:, :, depth_idx],
                    candidate_alive,
                )
                candidate_survived[:, :, depth_idx] = (
                    candidate_alive & in_window
                )
            candidate_values = candidate_survived.float().sum(dim=-1)
            reset_advantage = candidate_values[:, 1] - candidate_values[:, 0]
            positive_rows = replay_rows & reset_advantage.gt(
                self.on_policy_reset_replay_advantage_threshold
            )
            harmful_rows = replay_rows & reset_advantage.lt(0)
            credit_mask = (
                candidate_survived[:, 1]
                & suffix
                & positive_rows.unsqueeze(-1)
            )
            advantage_weight = reset_advantage.clamp(
                min=0.0,
                max=self.on_policy_reset_replay_value_clip,
            ).unsqueeze(-1)

        replay_verifier_ids = candidate_verifier_ids[:, 1]

        target_logit = replay_logits.gather(
            -1, replay_verifier_ids.unsqueeze(-1)
        ).squeeze(-1)
        top2_values, top2_ids = torch.topk(replay_logits, k=2, dim=-1)
        best_other = torch.where(
            top2_ids[..., 0].eq(replay_verifier_ids),
            top2_values[..., 1],
            top2_values[..., 0],
        )
        margin = target_logit - best_other
        penalty = F.softplus(
            (self.on_policy_margin_floor - margin) / self.on_policy_temperature
        )
        raw = (
            self.on_policy_reset_replay_alpha
            * phase_scale
            * (penalty * credit_mask.float() * advantage_weight).sum()
        )
        objective, budget_scale = self._budget_auxiliary_loss(
            self._effective_on_policy_mix() * raw,
            self.on_policy_reset_replay_loss_budget,
            ce_num,
        )
        denominator = replay_rows.sum().clamp_min(1)
        diagnostics = {
            "reset_replay_phase_scale": zero.new_tensor(phase_scale),
            "reset_replay_scored_rows": replay_rows.sum().detach(),
            "reset_replay_credit_count": credit_mask.sum().detach(),
            "reset_replay_mean_survival": (
                (candidate_survived[:, 1] & suffix).sum().float() / denominator
            ).detach(),
            "reset_replay_positive_rate": (
                positive_rows.sum().float() / denominator
            ).detach(),
            "reset_replay_harmful_rate": (
                harmful_rows.sum().float() / denominator
            ).detach(),
            "reset_replay_mean_advantage": (
                (reset_advantage * replay_rows.float()).sum() / denominator
            ).detach(),
            "reset_replay_mean_positive_advantage": (
                (reset_advantage.clamp_min(0.0) * positive_rows.float()).sum()
                / positive_rows.sum().clamp_min(1)
            ).detach(),
            "reset_replay_loss": raw.detach(),
            "reset_replay_budget_scale": budget_scale.detach(),
        }
        return objective, diagnostics

    def _multi_teacher_oracle_objective(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_ids: Optional[torch.Tensor],
        anchors: torch.Tensor,
        proposals: torch.Tensor,
        logits: torch.Tensor,
        survived: torch.Tensor,
        valid: torch.Tensor,
        active_rows: torch.Tensor,
        rollout_temperatures: torch.Tensor,
        target_topk_ids: Optional[torch.Tensor],
        target_topk_probs: Optional[torch.Tensor],
        ce_num: torch.Tensor,
        mix: float,
    ) -> Tuple[torch.Tensor, dict]:
        """Re-score a cached oracle trajectory and distill positive advantage.

        Distillation is restricted to the shared student/teacher prefix and to
        states that remain reachable under the teacher.  This avoids comparing
        distributions from two different autoregressive states after their
        proposal paths diverge.
        """
        zero = logits.new_zeros((), dtype=torch.float32)
        provider = self.multi_teacher_oracle_provider
        if provider is None or sample_ids is None:
            return zero, {"multi_teacher_oracle_cache_hit_rate": zero}
        cached = provider(sample_ids, anchors, rollout_temperatures, logits.device)
        hit = cached["hit"].bool() & active_rows

        depth_count = logits.size(1)
        teacher_proposals = cached["proposals"][:, :depth_count]
        hard_student_value = (survived & valid.bool()).sum(dim=-1).float()
        student_value = hard_student_value
        if target_topk_ids is not None and target_topk_probs is not None:
            student_log_probs = processed_log_probs(
                logits,
                rollout_temperatures,
                top_k=self.on_policy_deployment_top_k,
                top_p=self.on_policy_deployment_top_p,
            )
            target_matches = target_topk_ids.eq(proposals.unsqueeze(-1))
            target_proposal_prob = (
                target_topk_probs * target_matches.float()
            ).sum(dim=-1)
            student_proposal_prob = student_log_probs.gather(
                -1, proposals.unsqueeze(-1)
            ).squeeze(-1).exp()
            student_acceptance_probability = torch.minimum(
                torch.ones_like(target_proposal_prob),
                target_proposal_prob / student_proposal_prob.clamp_min(1e-12),
            )
            student_soft_value = (
                torch.cumprod(
                    torch.where(
                        valid.bool(),
                        student_acceptance_probability,
                        torch.ones_like(student_acceptance_probability),
                    ),
                    dim=-1,
                )
                * valid
            ).sum(dim=-1)
            student_value = torch.where(
                rollout_temperatures.gt(0), student_soft_value, hard_student_value
            )
        teacher_value = cached["cached_value"]
        advantage = teacher_value - student_value
        positive = hit & advantage.gt(self.multi_teacher_oracle_advantage_threshold)

        prefix_equal = teacher_proposals.eq(proposals)
        shared_before = torch.cat(
            [
                torch.ones_like(prefix_equal[:, :1]),
                torch.cumprod(prefix_equal[:, :-1].to(torch.int32), dim=-1).bool(),
            ],
            dim=-1,
        )
        teacher_acceptance_probability = cached["acceptance_probability"][
            :, :depth_count
        ]
        teacher_occupancy_before = torch.cat(
            [
                torch.ones_like(teacher_acceptance_probability[:, :1]),
                torch.cumprod(teacher_acceptance_probability[:, :-1], dim=-1),
            ],
            dim=-1,
        )
        distill_mask = (
            positive.unsqueeze(-1)
            & shared_before
            & valid.bool()
        )
        teacher_ids = cached["topk_ids"][:, :depth_count]
        teacher_probs = cached["topk_probs"][:, :depth_count].float()
        teacher_probs = teacher_probs / teacher_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        student_selected_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, teacher_ids
        )
        kl = (
            teacher_probs
            * (teacher_probs.clamp_min(1e-8).log() - student_selected_log_probs)
        ).sum(dim=-1)
        gain_weight = advantage.clamp(
            min=0.0, max=self.multi_teacher_oracle_value_clip
        ).unsqueeze(-1)
        raw = self.multi_teacher_oracle_alpha * (
            kl
            * distill_mask.float()
            * teacher_occupancy_before.detach()
            * gain_weight
        ).sum()
        if self.multi_teacher_oracle_mode != "distill":
            raw = zero
        objective, budget_scale = self._budget_auxiliary_loss(
            mix * raw, self.multi_teacher_oracle_loss_budget, ce_num
        )

        hit_count = hit.sum().clamp_min(1)
        positive_count = positive.sum().clamp_min(1)
        diagnostics = {
            "multi_teacher_oracle_cache_hit_rate": (
                hit.sum().float() / active_rows.sum().clamp_min(1)
            ).detach(),
            "multi_teacher_oracle_positive_rate": (
                positive.sum().float() / hit_count
            ).detach(),
            "multi_teacher_oracle_mean_gain": (
                (advantage * hit.float()).sum() / hit_count
            ).detach(),
            "multi_teacher_oracle_mean_positive_gain": (
                (advantage * positive.float()).sum() / positive_count
            ).detach(),
            "multi_teacher_oracle_cached_gain": (
                ((cached["cached_value"] - cached["baseline_value"]) * hit.float()).sum()
                / hit_count
            ).detach(),
            "multi_teacher_oracle_shared_prefix_coverage": (
                distill_mask.sum().float()
                / (positive.unsqueeze(-1) & valid.bool()).sum().clamp_min(1)
            ).detach(),
            "multi_teacher_oracle_distill_count": distill_mask.sum().detach(),
            "multi_teacher_oracle_raw_loss": raw.detach(),
            "multi_teacher_oracle_budget_scale": budget_scale.detach(),
        }
        provider_owner = getattr(provider, "__self__", None)
        teacher_names = getattr(provider_owner, "teacher_names", ())
        for teacher_index in range(len(teacher_names)):
            diagnostics[f"multi_teacher_oracle_teacher_{teacher_index}_rate"] = (
                (cached["teacher_index"].eq(teacher_index) & hit).sum().float()
                / hit_count
            ).detach()
        return objective, diagnostics

    def _on_policy_survival_objective(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        eval_mask: torch.Tensor,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        ce_num: torch.Tensor,
        sample_ids: Optional[torch.Tensor] = None,
        source_ids: Optional[torch.Tensor] = None,
        block_hazard: Optional[torch.Tensor] = None,
        block_flatness: Optional[torch.Tensor] = None,
        active_block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Credit real drafter proposals with target-scored prefix survival.

        One valid anchor per sequence is scored to bound verifier overhead.  The
        proposal path is generated with the exact CARH serving recurrence and,
        when enabled, the fixed-count parallel refinement pass.  The frozen
        target then evaluates that same final path, so accepted-prefix,
        first-rejection and full-survival states are mutually exclusive and no
        teacher-forced suffix receives survival credit.
        """
        zero = base_logits.new_zeros((), dtype=torch.float32)
        mix = self._effective_on_policy_mix()
        if (
            mix <= 0
            or self.on_policy_scorer is None
            or self._global_step % self.on_policy_interval != 0
        ):
            return zero, {"on_policy_mix": zero.new_tensor(mix)}

        active_block_size = int(active_block_size or self.block_size)
        active = block_keep_mask & eval_mask[..., 0].bool()
        active_rows = active.any(dim=-1)
        # All ranks must either enter or skip the verifier/FSDP path together.
        # `_select_on_policy_blocks` installs a harmless block-0 placeholder
        # for a rank with no local active rows.
        if not _distributed_any(active_rows):
            return zero, {"on_policy_mix": zero.new_tensor(mix)}
        dialogue_features = None
        if (
            self.on_policy_anchor_sampling == "dialogue-frontier"
            or self.dialogue_occupancy_diagnostics
            or self.on_policy_language_hazard_credit
        ):
            dialogue_features = self._dialogue_anchor_features(
                active, anchor_positions, loss_mask, input_ids
            )
        block_index, sampling_weights = self._select_on_policy_blocks(
            active, block_hazard, block_flatness, dialogue_features
        )
        row = torch.arange(input_ids.size(0), device=input_ids.device)
        anchors = anchor_positions[row, block_index]
        base = base_logits[row, block_index, :active_block_size]
        hidden = hidden_states[row, block_index, :active_block_size]
        valid = (
            eval_mask[row, block_index, :active_block_size]
            * active_rows.unsqueeze(-1)
        )

        predecessor = input_ids[row, anchors]
        markov_head = self.draft_model.markov_head
        memory_enabled = getattr(markov_head, "sampled_prefix_memory", None) is not None
        memory_state = (
            markov_head.init_sampled_prefix_state(predecessor)
            if memory_enabled else None
        )
        rollout_temperatures, temperature_choices = self._sample_on_policy_temperatures(
            input_ids.size(0), input_ids.device
        )
        proposal_steps = []
        rollout_logits = []
        gate_calibration_logits = []
        gate_calibration_latents = []
        gate_calibration_scale = self._carh_gate_calibration_scale()
        for depth in range(active_block_size):
            if memory_enabled and depth >= 2:
                memory_state = markov_head.advance_sampled_prefix_state(
                    memory_state, proposal_steps[depth - 2]
                )
            memory_kwargs = (
                {"sampled_prefix_state": memory_state} if memory_enabled else {}
            )
            step_base = base[:, depth, :]
            if self.draft_model.recall_correction is not None:
                depth_ids = torch.full_like(predecessor, depth)
                corrected_hidden = self.draft_model.apply_recall_correction(
                    hidden[:, depth, :], predecessor, depth_ids
                )
                step_base = self.lm_head(corrected_hidden)
            if gate_calibration_scale > 0:
                residual, _, latent = (
                    self.draft_model.markov_head.compute_step_bias_gate_and_latent(
                        predecessor, hidden[:, depth, :], depth_idx=depth,
                        **_carh_predecessor_context_kwargs(
                            self.draft_model.markov_head, hidden, depth
                        ),
                        **memory_kwargs,
                    )
                )
                # Re-evaluate the scalar projection on a detached latent so the
                # calibration auxiliary updates gate_proj only.  The normal
                # proposal loss still trains the complete CARH path.
                gate_logit = self.draft_model.markov_head.gate_proj(
                    latent.detach()
                ).squeeze(-1)
                gate_calibration_logits.append(gate_logit)
                gate_calibration_latents.append(latent.detach())
            else:
                residual = self.draft_model.markov_head.compute_step_bias(
                    predecessor, hidden[:, depth, :], depth_idx=depth,
                    **_carh_predecessor_context_kwargs(
                        self.draft_model.markov_head, hidden, depth
                    ),
                    **memory_kwargs,
                )
            step = step_base + residual
            proposal = self._sample_on_policy_tokens(step, rollout_temperatures)
            rollout_logits.append(step)
            proposal_steps.append(proposal)
            predecessor = proposal
        proposals = torch.stack(proposal_steps, dim=-1)
        logits = torch.stack(rollout_logits, dim=1).float()
        preference_proposals = None
        if self.on_policy_preference_alpha > 0:
            preference_predecessor = input_ids[row, anchors]
            preference_steps = []
            for depth in range(active_block_size):
                preference_base = base[:, depth, :]
                if self.draft_model.recall_correction is not None:
                    depth_ids = torch.full_like(preference_predecessor, depth)
                    corrected_hidden = self.draft_model.apply_recall_correction(
                        hidden[:, depth, :], preference_predecessor, depth_ids
                    )
                    preference_base = self.lm_head(corrected_hidden)
                preference_residual = (
                    self.draft_model.markov_head.compute_step_bias(
                        preference_predecessor,
                        hidden[:, depth, :],
                        depth_idx=depth,
                        **_carh_predecessor_context_kwargs(
                            self.draft_model.markov_head, hidden, depth
                        ),
                    )
                )
                preference_step = preference_base + preference_residual
                preference_token = self._sample_on_policy_tokens(
                    preference_step, rollout_temperatures
                )
                preference_steps.append(preference_token)
                preference_predecessor = preference_token
            preference_proposals = torch.stack(preference_steps, dim=-1)
        pre_refiner_logits = logits
        pre_refiner_proposals = proposals
        refiner_hazard_logits = None
        refiner_gate = None
        advantage_class_logits = None
        advantage_value_prediction = None
        advantage_gate_scale, _ = self._refiner_advantage_phase()
        use_advantage_teacher = (
            self.refiner_advantage_mode != "none" and advantage_gate_scale > 0
        )
        if use_advantage_teacher and preference_proposals is not None:
            raise RuntimeError(
                "verifier-ranked multi-rollout preference and refiner "
                "advantage teacher cannot be enabled in the same run"
            )
        if (
            getattr(self.draft_model, "parallel_refiner", None) is not None
            and (
                use_advantage_teacher
                or self.parallel_refiner_hazard_alpha > 0
                or self.parallel_refiner_recovery_alpha > 0
                or self.parallel_refiner_preservation_alpha > 0
            )
        ):
            (
                refined_logits,
                refined_proposals,
                refiner_hazard_logits,
                refiner_gate,
            ) = self._hazard_adaptive_parallel_refinement(
                logits=(logits.detach() if use_advantage_teacher else logits),
                hidden_states=(hidden.detach() if use_advantage_teacher else hidden),
                anchor_token_ids=input_ids[row, anchors],
                rollout_temperatures=rollout_temperatures,
            )
            refiner = self.draft_model.parallel_refiner
            advantage_class_logits = getattr(
                refiner, "_last_advantage_class_logits", None
            )
            advantage_value_prediction = getattr(
                refiner, "_last_advantage_value", None
            )
            if not use_advantage_teacher:
                logits, proposals = refined_logits, refined_proposals
        logits = logits.float()

        with torch.no_grad():
            if use_advantage_teacher:
                scorer_proposals = torch.stack(
                    [pre_refiner_proposals, refined_proposals], dim=1
                )
            elif preference_proposals is not None:
                scorer_proposals = torch.stack(
                    [proposals, preference_proposals], dim=1
                )
            else:
                scorer_proposals = proposals
            scorer_args = (
                input_ids,
                attention_mask,
                anchors,
                scorer_proposals,
                active_rows,
            )
            if (
                self.on_policy_rejection_aligned_alpha > 0
                or self.on_policy_clipped_rkl_alpha > 0
                or self.on_policy_target_margin_alpha > 0
                or self.on_policy_marginal_value_credit
            ):
                distribution_temperatures = rollout_temperatures.clamp_min(
                    self.on_policy_target_distribution_temperature_floor
                )
                verifier_output = self.on_policy_scorer(
                    *scorer_args, distribution_temperatures
                )
            else:
                verifier_output = self.on_policy_scorer(*scorer_args)
            if isinstance(verifier_output, tuple):
                verifier_ids, target_topk_ids, target_topk_probs = verifier_output
                verifier_ids = verifier_ids.to(input_ids.device)
                target_topk_ids = target_topk_ids.to(input_ids.device)
                target_topk_probs = target_topk_probs.to(input_ids.device).float()
            else:
                verifier_ids = verifier_output.to(input_ids.device)
                target_topk_ids = None
                target_topk_probs = None
            teacher_verifier_ids = None
            preference_verifier_ids = None
            if use_advantage_teacher:
                teacher_verifier_ids = verifier_ids[:, 1]
                verifier_ids = verifier_ids[:, 0]
                if target_topk_ids is not None:
                    target_topk_ids = target_topk_ids[:, 0]
                    target_topk_probs = target_topk_probs[:, 0]
                proposals = pre_refiner_proposals
                logits = pre_refiner_logits.float()
            elif preference_proposals is not None:
                preference_verifier_ids = verifier_ids[:, 1]
                verifier_ids = verifier_ids[:, 0]
                if target_topk_ids is not None:
                    target_topk_ids = target_topk_ids[:, 0]
                    target_topk_probs = target_topk_probs[:, 0]
            accepted = proposals.eq(verifier_ids) & valid.bool()
            survived = torch.cumprod(accepted.to(torch.int32), dim=-1).bool()
            alive_before = torch.cat(
                [torch.ones_like(survived[:, :1]), survived[:, :-1]], dim=-1
            ) & valid.bool()
            first_rejection = alive_before & ~accepted & valid.bool()
            full_survival = survived[:, -1] & active_rows
            confidence_target = torch.zeros_like(eval_mask)
            confidence_mask = torch.zeros_like(eval_mask)
            confidence_target[
                row, block_index, :active_block_size
            ] = accepted.float()
            confidence_mask[
                row, block_index, :active_block_size
            ] = alive_before.float()
            self._last_on_policy_confidence = (
                confidence_target,
                confidence_mask,
            )

            if self.multi_teacher_export_sink is not None:
                self.multi_teacher_export_sink.record(
                    sample_ids=sample_ids,
                    anchors=anchors,
                    temperatures=rollout_temperatures,
                    proposals=proposals,
                    logits=logits,
                    verifier_ids=verifier_ids,
                    valid=valid,
                    active_rows=active_rows,
                    acceptance_probability=(
                        torch.where(
                            rollout_temperatures.gt(0).unsqueeze(-1),
                            torch.minimum(
                                torch.ones_like(target_topk_probs[..., 0]),
                                (
                                    target_topk_probs
                                    * target_topk_ids.eq(
                                        proposals.unsqueeze(-1)
                                    ).float()
                                ).sum(dim=-1)
                                / processed_log_probs(
                                    logits,
                                    rollout_temperatures,
                                    top_k=self.on_policy_deployment_top_k,
                                    top_p=self.on_policy_deployment_top_p,
                                ).gather(
                                    -1, proposals.unsqueeze(-1)
                                ).squeeze(-1).exp().clamp_min(1e-12),
                            ),
                            accepted.float(),
                        )
                        if target_topk_ids is not None
                        else accepted.float()
                    ),
                )

        preference_objective = zero
        preference_diagnostics = {}
        if preference_proposals is not None:
            preference_objective, preference_diagnostics = (
                self._rollout_preference_objective(
                    primary_proposals=proposals,
                    alternative_proposals=preference_proposals,
                    primary_logits=logits,
                    primary_verifier_ids=verifier_ids,
                    alternative_verifier_ids=preference_verifier_ids,
                    valid=valid,
                    active_rows=active_rows,
                    rollout_temperatures=rollout_temperatures,
                    ce_num=ce_num,
                )
            )

        multi_teacher_objective, multi_teacher_diagnostics = (
            self._multi_teacher_oracle_objective(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sample_ids=sample_ids,
                anchors=anchors,
                proposals=proposals,
                logits=logits,
                survived=survived,
                valid=valid,
                active_rows=active_rows,
                rollout_temperatures=rollout_temperatures,
                target_topk_ids=target_topk_ids,
                target_topk_probs=target_topk_probs,
                ce_num=ce_num,
                mix=mix,
            )
        )

        advantage_objective = zero
        advantage_diagnostics = {}
        if use_advantage_teacher:
            teacher_accepted = (
                refined_proposals.eq(teacher_verifier_ids) & valid.bool()
            )
            advantage_objective, advantage_diagnostics = (
                self._refiner_advantage_objective(
                    student_logits=pre_refiner_logits,
                    teacher_logits=refined_logits,
                    base_accepted=accepted,
                    teacher_accepted=teacher_accepted,
                    valid=valid,
                    advantage_class_logits=advantage_class_logits,
                    advantage_value_prediction=advantage_value_prediction,
                    rollout_temperatures=rollout_temperatures,
                    ce_num=ce_num,
                )
            )

        pareto_row_weight, pareto_diagnostics = (
            self._temperature_pareto_credit_weights(
                survived=survived,
                valid=valid,
                temperature_choices=temperature_choices,
                active_rows=active_rows,
            )
        )
        pareto_credit = pareto_row_weight.unsqueeze(-1)

        target_logit = logits.gather(-1, verifier_ids.unsqueeze(-1)).squeeze(-1)
        top2_values, top2_ids = torch.topk(logits, k=2, dim=-1)
        best_other = torch.where(
            top2_ids[..., 0].eq(verifier_ids), top2_values[..., 1], top2_values[..., 0]
        )
        margin = target_logit - best_other
        penalty = F.softplus(
            (self.on_policy_margin_floor - margin) / self.on_policy_temperature
        )
        # Depth values below are one-indexed semantically: tensor index 0 is
        # draft depth 1.  A boundary max of zero preserves the legacy behavior
        # where accepted-prefix protection is not depth-restricted.
        depth = torch.arange(active_block_size, device=input_ids.device)
        shallow_fr = first_rejection & depth.lt(
            self.on_policy_first_rejection_max_depth
        ).view(1, -1)
        # Accepted-prefix protection uses only states before a rejection. Full
        # chains are handled by the separate full-survival term, never twice.
        prefix_protect = survived & ~full_survival.unsqueeze(-1)
        full_protect = valid.bool() & full_survival.unsqueeze(-1)
        if self.on_policy_boundary_max_depth > 0:
            boundary_depth = depth.lt(
                self.on_policy_boundary_max_depth
            ).view(1, -1)
        else:
            boundary_depth = torch.ones_like(valid, dtype=torch.bool)
        boundary_protect = (prefix_protect | shallow_fr) & boundary_depth
        deep_depth = depth.ge(self.on_policy_deep_start_depth - 1).view(1, -1)
        deep_protect = (prefix_protect | full_protect) & deep_depth
        hazard_credit = self._temperature_hazard_credit_weights(
            first_rejection=first_rejection,
            alive_before=alive_before,
            valid=valid,
            temperature_choices=temperature_choices,
            active_rows=active_rows,
        )
        selected_han_ratio = (
            dialogue_features["han_ratio"][row, block_index]
            if dialogue_features is not None
            else valid.new_zeros(valid.size(0), dtype=torch.float32)
        )
        rollout_predecessors = torch.cat(
            [input_ids[row, anchors].unsqueeze(-1), proposals[:, :-1]], dim=-1
        )
        language_credit, language_hazard_diagnostics = (
            self._language_hazard_credit_weights(
                first_rejection=first_rejection,
                alive_before=alive_before,
                valid=valid,
                active_rows=active_rows,
                han_ratio=selected_han_ratio,
                predecessor_ids=rollout_predecessors,
                proposal_ids=proposals,
            )
        )
        greedy_acceptance_surrogate = torch.sigmoid(
            margin / self.on_policy_marginal_value_temperature
        )
        conditional_acceptance = greedy_acceptance_surrogate
        if target_topk_ids is not None and target_topk_probs is not None:
            proposal_target_probability = (
                target_topk_probs
                * target_topk_ids.eq(proposals.unsqueeze(-1)).float()
            ).sum(dim=-1)
            proposal_draft_probability = processed_log_probs(
                logits,
                rollout_temperatures,
                top_k=self.on_policy_deployment_top_k,
                top_p=self.on_policy_deployment_top_p,
            ).gather(-1, proposals.unsqueeze(-1)).squeeze(-1).exp()
            rejection_acceptance = torch.minimum(
                torch.ones_like(proposal_target_probability),
                proposal_target_probability
                / proposal_draft_probability.clamp_min(1e-12),
            )
            conditional_acceptance = torch.where(
                rollout_temperatures.gt(1e-6).unsqueeze(-1),
                rejection_acceptance,
                greedy_acceptance_surrogate,
            )
        marginal_credit, marginal_value_diagnostics = (
            self._marginal_survival_credit_weights(
                conditional_acceptance=conditional_acceptance,
                valid=valid,
                active_rows=active_rows,
            )
        )
        survival_credit = (
            hazard_credit * language_credit * pareto_credit * marginal_credit
        )
        greedy_rows = rollout_temperatures.le(1e-6).unsqueeze(-1)
        target_margin_frontier_raw = zero
        target_margin_frontier_count = zero
        target_margin_mean = zero
        target_margin_required_mean = zero
        if self.on_policy_target_margin_alpha > 0:
            if target_topk_probs is None or target_topk_probs.size(-1) < 2:
                raise RuntimeError(
                    "target-margin frontier requires at least two target candidates"
                )
            eps = torch.finfo(torch.float32).eps
            target_margin = (
                target_topk_probs[..., 0].clamp_min(eps).log()
                - target_topk_probs[..., 1].clamp_min(eps).log()
            ).detach()
            required_margin = (
                self.on_policy_target_margin_scale * target_margin
                + self.on_policy_target_margin_offset
            ).clamp(
                min=self.on_policy_target_margin_min,
                max=self.on_policy_target_margin_max,
            )
            target_margin_mask = (
                first_rejection
                & greedy_rows
                & depth.lt(self.on_policy_target_margin_max_depth).view(1, -1)
            )
            target_margin_penalty = F.softplus(
                (required_margin - margin) / self.on_policy_temperature
            )
            target_margin_frontier_raw = self.on_policy_target_margin_alpha * (
                target_margin_penalty
                * target_margin_mask.float()
                * survival_credit
            ).sum()
            target_margin_frontier_count = target_margin_mask.sum().float()
            target_margin_mean = (
                target_margin * target_margin_mask.float()
            ).sum() / target_margin_frontier_count.clamp_min(1.0)
            target_margin_required_mean = (
                required_margin * target_margin_mask.float()
            ).sum() / target_margin_frontier_count.clamp_min(1.0)
        sr_route = (
            greedy_rows.float()
            if self.on_policy_temperature_exclusive_routing
            else torch.ones_like(survival_credit)
        )
        routed_survival_credit = survival_credit * sr_route
        repair_weights, repair_value_diagnostics = self._repair_value_weights(
            input_ids=input_ids, attention_mask=attention_mask, anchors=anchors,
            base=base, hidden=hidden, proposals=proposals, verifier_ids=verifier_ids,
            first_rejection=first_rejection, valid=valid,
            rollout_temperatures=rollout_temperatures,
        )
        routed_survival_credit = routed_survival_credit * repair_weights
        strict_early = torch.zeros_like(first_rejection)
        strict_middle = torch.zeros_like(first_rejection)
        strict_deep = torch.zeros_like(first_rejection)
        if self.on_policy_credit_partition == "strict-v2":
            # Real verifier states own disjoint depth/state regions:
            #   d<=early: first-rejection margin repair (FRC semantics),
            #   early<d<=middle: full CE at the rejected frontier (Prefix-Full),
            #   d>=deep: margin preservation on genuinely survived tokens (DSG).
            # No teacher-forced suffix or post-rejection token receives credit.
            early_depth = depth.lt(
                self.on_policy_first_rejection_max_depth
            ).view(1, -1)
            middle_depth = (
                depth.ge(self.on_policy_first_rejection_max_depth)
                & depth.lt(self.on_policy_middle_max_depth)
            ).view(1, -1)
            strict_early = first_rejection & early_depth
            strict_middle = first_rejection & middle_depth
            strict_deep = survived & deep_depth
            middle_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                verifier_ids.reshape(-1),
                reduction="none",
            ).view_as(valid)
            sr_raw = self.on_policy_survival_alpha * (
                penalty * strict_early.float() * routed_survival_credit
            ).sum()
            sr_raw = sr_raw + self.on_policy_middle_alpha * (
                middle_ce * strict_middle.float() * routed_survival_credit
            ).sum()
            sr_raw = sr_raw + self.on_policy_deep_alpha * (
                penalty * strict_deep.float() * routed_survival_credit
            ).sum()
        else:
            sr_raw = self.on_policy_survival_alpha * (
                penalty * boundary_protect.float() * routed_survival_credit
            ).sum()
            sr_raw = sr_raw + self.on_policy_deep_alpha * (
                penalty * deep_protect.float() * routed_survival_credit
            ).sum()
            # Retain the old all-depth full-survival term for checkpoint/config
            # compatibility.  Strict-v2 deliberately excludes this overlap.
            sr_raw = sr_raw + self.on_policy_full_alpha * (
                penalty * full_protect.float() * routed_survival_credit
            ).sum()
        raw = sr_raw + target_margin_frontier_raw

        # Pareto consolidation is deliberately orthogonal to strict-v2's
        # exclusive repair regions.  Every genuinely survived state receives a
        # small margin-floor preservation term, while CARH is prevented from
        # reducing the verifier-target margin below the unary draft backbone.
        # Both masks are detached target-scored states and add no serving path.
        preservation_mask = survived & valid.bool()
        preservation_raw = self.on_policy_preservation_alpha * (
            penalty * preservation_mask.float() * survival_credit
        ).sum()

        greedy_preservation_mask = survived & valid.bool() & greedy_rows
        greedy_preservation_penalty = F.softplus(
            (
                self.on_policy_greedy_preservation_margin_floor
                - margin
            )
            / self.on_policy_greedy_preservation_temperature
        )
        greedy_preservation_raw = self.on_policy_greedy_preservation_alpha * (
            greedy_preservation_penalty
            * greedy_preservation_mask.float()
            * survival_credit
        ).sum()

        base_fp32_for_regression = base.detach().float()
        base_target = base_fp32_for_regression.gather(
            -1, verifier_ids.unsqueeze(-1)
        ).squeeze(-1)
        base_top2, base_top2_ids = torch.topk(
            base_fp32_for_regression, k=2, dim=-1
        )
        base_other = torch.where(
            base_top2_ids[..., 0].eq(verifier_ids),
            base_top2[..., 1],
            base_top2[..., 0],
        )
        base_margin = base_target - base_other
        regression_mask = (
            alive_before
            & valid.bool()
            & base_margin.gt(margin.detach())
        ).detach()
        regression_penalty = F.softplus(
            (
                base_margin
                + self.on_policy_regression_margin
                - margin
            )
            / self.on_policy_temperature
        )
        regression_raw = self.on_policy_regression_alpha * (
            regression_penalty
            * regression_mask.float()
            * survival_credit
        ).sum()
        raw = raw + preservation_raw + regression_raw + greedy_preservation_raw
        refiner_objective = zero
        refiner_diagnostics = {}
        if refiner_hazard_logits is not None:
            alive_weight = alive_before.float() * valid
            hazard_target = first_rejection.float()
            hazard_bce = F.binary_cross_entropy_with_logits(
                refiner_hazard_logits.float(),
                hazard_target,
                reduction="none",
            )
            positive_weight = first_rejection.float() * valid
            negative_weight = (alive_before & accepted).float() * valid
            positive_mass = positive_weight.sum()
            negative_mass = negative_weight.sum()
            hazard_mass = positive_mass + negative_mass
            positive_scale = torch.where(
                positive_mass > 0,
                0.5 * hazard_mass / positive_mass.clamp_min(1),
                torch.zeros_like(positive_mass),
            )
            negative_scale = torch.where(
                negative_mass > 0,
                0.5 * hazard_mass / negative_mass.clamp_min(1),
                torch.zeros_like(negative_mass),
            )
            balanced_hazard_weight = (
                positive_weight * positive_scale
                + negative_weight * negative_scale
            )
            hazard_raw = self.parallel_refiner_hazard_alpha * (
                hazard_bce * balanced_hazard_weight
            ).sum()

            with torch.no_grad():
                pre_ids = pre_refiner_logits.argmax(dim=-1)
                pre_target = pre_refiner_logits.gather(
                    -1, verifier_ids.unsqueeze(-1)
                ).squeeze(-1)
                pre_top2, pre_top2_ids = torch.topk(
                    pre_refiner_logits, k=2, dim=-1
                )
                pre_other = torch.where(
                    pre_top2_ids[..., 0].eq(verifier_ids),
                    pre_top2[..., 1],
                    pre_top2[..., 0],
                )
                pre_margin = pre_target - pre_other
                recoverable_frontier = (
                    alive_before
                    & pre_ids.ne(verifier_ids)
                    & valid.bool()
                )
                preserved_frontier = (
                    alive_before
                    & pre_ids.eq(verifier_ids)
                    & valid.bool()
                )
            recovery_penalty = F.softplus(
                (
                    self.parallel_refiner_margin_floor - margin
                )
                / self.parallel_refiner_temperature
            )
            recovery_raw = self.parallel_refiner_recovery_alpha * (
                recovery_penalty
                * recoverable_frontier.float()
                * survival_credit
            ).sum()
            preservation_penalty = F.softplus(
                (
                    pre_margin
                    - margin
                )
                / self.parallel_refiner_temperature
            )
            preservation_raw_refiner = (
                self.parallel_refiner_preservation_alpha
                * (
                    preservation_penalty
                    * preserved_frontier.float()
                    * survival_credit
                ).sum()
            )
            refiner_raw = hazard_raw + recovery_raw + preservation_raw_refiner
            refiner_objective, refiner_budget_scale = self._budget_auxiliary_loss(
                mix * refiner_raw,
                self.parallel_refiner_loss_budget,
                ce_num,
            )
            with torch.no_grad():
                refiner_diagnostics = {
                    "parallel_refiner_hazard_loss": hazard_raw.detach(),
                    "parallel_refiner_recovery_loss": recovery_raw.detach(),
                    "parallel_refiner_preservation_loss": (
                        preservation_raw_refiner.detach()
                    ),
                    "parallel_refiner_budget_scale": (
                        refiner_budget_scale.detach()
                    ),
                    "parallel_refiner_first_rejection_hazard": (
                        (
                            torch.sigmoid(refiner_hazard_logits.float())
                            * first_rejection.float()
                        ).sum()
                        / first_rejection.sum().clamp_min(1)
                    ),
                    "parallel_refiner_survived_hazard": (
                        (
                            torch.sigmoid(refiner_hazard_logits.float())
                            * survived.float()
                        ).sum()
                        / survived.sum().clamp_min(1)
                    ),
                    "parallel_refiner_recoverable_count": (
                        recoverable_frontier.sum().detach()
                    ),
                    "parallel_refiner_preserved_count": (
                        preserved_frontier.sum().detach()
                    ),
                    "parallel_refiner_gate_mean_on_alive": (
                        (refiner_gate.float() * alive_weight).sum()
                        / alive_weight.sum().clamp_min(1)
                    ),
                }
        gate_calibration_diagnostics = {}
        if gate_calibration_scale > 0:
            calibration_logits = torch.stack(gate_calibration_logits, dim=-1)
            calibration_latents = torch.stack(gate_calibration_latents, dim=1)
            with torch.no_grad():
                ungated_residual = self.draft_model.markov_head.project_bias(
                    calibration_latents
                ).float()
                ungated_logits = base.detach().float() + ungated_residual
                base_fp32 = base.detach().float()
                if target_topk_ids is not None and target_topk_probs is not None:
                    target_mass = target_topk_probs.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-6)
                    target_weights = target_topk_probs / target_mass
                    base_selected = base_fp32.gather(-1, target_topk_ids)
                    ungated_selected = ungated_logits.gather(
                        -1, target_topk_ids
                    )
                    log_normalizer_shift = (
                        torch.logsumexp(ungated_logits, dim=-1)
                        - torch.logsumexp(base_fp32, dim=-1)
                    ).unsqueeze(-1)
                    gate_utility_gain = (
                        target_weights
                        * (
                            ungated_selected
                            - base_selected
                            - log_normalizer_shift
                        )
                    ).sum(dim=-1)
                else:
                    base_gold = base_fp32.gather(
                        -1, verifier_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    base_top2, base_top2_ids = torch.topk(
                        base_fp32, k=2, dim=-1
                    )
                    base_other = torch.where(
                        base_top2_ids[..., 0].eq(verifier_ids),
                        base_top2[..., 1],
                        base_top2[..., 0],
                    )
                    ungated_gold = ungated_logits.gather(
                        -1, verifier_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    ungated_top2, ungated_top2_ids = torch.topk(
                        ungated_logits, k=2, dim=-1
                    )
                    ungated_other = torch.where(
                        ungated_top2_ids[..., 0].eq(verifier_ids),
                        ungated_top2[..., 1],
                        ungated_top2[..., 0],
                    )
                    gate_utility_gain = (ungated_gold - ungated_other) - (
                        base_gold - base_other
                    )
                helpful_gate = (
                    gate_utility_gain.gt(self.carh_gate_margin_threshold)
                    & alive_before
                )
                noop_gate = alive_before & ~helpful_gate
                harmful_gate = (
                    gate_utility_gain.lt(-self.carh_gate_margin_threshold)
                    & alive_before
                )

            gate_bce = F.binary_cross_entropy_with_logits(
                calibration_logits.float(),
                helpful_gate.float(),
                reduction="none",
            )
            positive_weight = valid * helpful_gate.float()
            negative_weight = valid * noop_gate.float()
            positive_mass = positive_weight.sum()
            negative_mass = negative_weight.sum()
            total_mass = positive_mass + negative_mass
            positive_scale = torch.where(
                positive_mass > 0,
                0.5 * total_mass / positive_mass.clamp_min(1e-6),
                torch.zeros_like(positive_mass),
            )
            negative_scale = torch.where(
                negative_mass > 0,
                0.5 * total_mass / negative_mass.clamp_min(1e-6),
                torch.zeros_like(negative_mass),
            )
            balanced_weight = (
                positive_weight * positive_scale
                + negative_weight * negative_scale
            )
            gate_calibration_raw = (gate_bce * balanced_weight).sum()
            gate_noop_raw = (
                torch.sigmoid(calibration_logits.float()).square()
                * negative_weight
            ).sum()
            raw = raw + gate_calibration_scale * (
                self.carh_gate_calibration_alpha * gate_calibration_raw
                + self.carh_gate_noop_alpha * gate_noop_raw
            )

            with torch.no_grad():
                predicted_open = calibration_logits.ge(0) & alive_before
                helpful_count = helpful_gate.sum().clamp_min(1)
                open_count = predicted_open.sum().clamp_min(1)
                noop_count = noop_gate.sum().clamp_min(1)
                gate_values = torch.sigmoid(calibration_logits.float())
                gate_calibration_diagnostics = {
                    "carh_gate_calibration_scale": zero.new_tensor(
                        gate_calibration_scale
                    ),
                    "carh_gate_helpful_rate": (
                        helpful_gate.sum().float()
                        / alive_before.sum().clamp_min(1)
                    ),
                    "carh_gate_harmful_rate": (
                        harmful_gate.sum().float()
                        / alive_before.sum().clamp_min(1)
                    ),
                    "carh_gate_precision": (
                        (predicted_open & helpful_gate).sum().float() / open_count
                    ),
                    "carh_gate_recall": (
                        (predicted_open & helpful_gate).sum().float()
                        / helpful_count
                    ),
                    "carh_gate_helpful_mean": (
                        (gate_values * helpful_gate.float()).sum() / helpful_count
                    ),
                    "carh_gate_noop_mean": (
                        (gate_values * noop_gate.float()).sum() / noop_count
                    ),
                    "carh_gate_utility_gain": (
                        (gate_utility_gain * alive_before.float()).sum()
                        / alive_before.sum().clamp_min(1)
                    ),
                }
                for depth_idx in range(active_block_size):
                    depth_alive = alive_before[:, depth_idx]
                    depth_count = depth_alive.sum().clamp_min(1)
                    gate_calibration_diagnostics.update(
                        {
                            f"carh_gate_helpful_rate_depth_{depth_idx + 1}": (
                                helpful_gate[:, depth_idx].sum().float()
                                / depth_count
                            ),
                            f"carh_gate_open_mean_depth_{depth_idx + 1}": (
                                (
                                    gate_values[:, depth_idx]
                                    * depth_alive.float()
                                ).sum()
                                / depth_count
                            ),
                        }
                    )
        distributional_overlap = None
        distributional_survival = None
        distributional_effective_k = None
        distributional_retained_mass = None
        distributional_cap_limited = None
        rejection_acceptance_probability = None
        rejection_occupancy_before = None
        rejection_expected_length = None
        rejection_alignment_mask = None
        rejection_alignment_raw = zero
        mixed_kl_forward = None
        mixed_kl_reverse = None
        mixed_kl_accepted_mask = None
        mixed_kl_rejected_mask = None
        mixed_kl_rejected_weights = None
        mixed_kl_raw = zero
        clipped_rkl_raw = zero
        clipped_rkl_count = zero
        clipped_rkl_unclipped_mean = zero
        clipped_rkl_capped_mean = zero
        distributional_blend = self._distributional_opsc_blend()
        if self.on_policy_distributional_top_k > 0:
            if target_topk_ids is None or target_topk_probs is None:
                raise RuntimeError(
                    "distributional OPSC requires a Top-k proposal scorer"
                )
            (
                adaptive_keep,
                distributional_effective_k,
                distributional_retained_mass,
                distributional_cap_limited,
            ) = self._adaptive_distributional_mask(
                target_topk_probs,
                self.on_policy_distributional_min_top_k,
                self.on_policy_distributional_mass_threshold,
            )
            masked_target_probs = target_topk_probs * adaptive_keep
            if (
                self.on_policy_rejection_aligned_alpha > 0
                or self.on_policy_clipped_rkl_alpha > 0
            ):
                draft_log_probs = processed_log_probs(
                    logits,
                    rollout_temperatures,
                    top_k=self.on_policy_deployment_top_k,
                    top_p=self.on_policy_deployment_top_p,
                )
            else:
                draft_log_probs = torch.log_softmax(logits, dim=-1)
            draft_topk_probs = torch.exp(
                draft_log_probs.gather(-1, target_topk_ids)
            ) * adaptive_keep

            # Draft-OPD objective port on the *same* DSpark proposal states.
            # The target API intentionally returns only Top-k probabilities.
            # Rather than renormalizing that truncated support (which changes
            # the divergence), aggregate everything outside the retained Top-k
            # into one tail event.  KL contracts under this coarse graining, so
            # these are finite, conservative lower bounds to the full-vocab KLs.
            if (
                self.on_policy_mixed_kl_alpha > 0
                or self.on_policy_clipped_rkl_alpha > 0
            ):
                eps = torch.finfo(torch.float32).eps
                target_retained = masked_target_probs.clamp_min(0.0)
                draft_retained = draft_topk_probs.clamp_min(0.0)
                target_tail = (
                    1.0 - target_retained.sum(dim=-1, keepdim=True)
                ).clamp_min(0.0)
                draft_tail = (
                    1.0 - draft_retained.sum(dim=-1, keepdim=True)
                ).clamp_min(0.0)
                target_bins = torch.cat([target_retained, target_tail], dim=-1)
                draft_bins = torch.cat([draft_retained, draft_tail], dim=-1)
                # Normalize away fp32 round-off when a retained sum is a few
                # ulps above one, then guard only log underflow.
                target_bins = target_bins / target_bins.sum(
                    dim=-1, keepdim=True
                ).clamp_min(eps)
                draft_bins = draft_bins / draft_bins.sum(
                    dim=-1, keepdim=True
                ).clamp_min(eps)
                target_log_bins = target_bins.clamp_min(eps).log()
                draft_log_bins = draft_bins.clamp_min(eps).log()
                mixed_kl_forward = (
                    target_bins * (target_log_bins - draft_log_bins)
                ).sum(dim=-1)
                mixed_kl_reverse = (
                    draft_bins * (draft_log_bins - target_log_bins)
                ).sum(dim=-1)

                if self.on_policy_clipped_rkl_alpha > 0:
                    rkl_mask = alive_before & valid.bool()
                    temperature_weights = zero.new_tensor(
                        self.on_policy_clipped_rkl_temperature_weights
                    )[temperature_choices].unsqueeze(-1)
                    rkl_weight = (
                        rkl_mask.float() * temperature_weights * pareto_credit
                    )
                    # Cap each state's contribution without killing its gradient:
                    # the detached scale preserves direction while preventing a
                    # few high-divergence states from dominating the batch.
                    clip_scale = (
                        self.on_policy_clipped_rkl_clip
                        / mixed_kl_reverse.detach().clamp_min(1e-12)
                    ).clamp(max=1.0)
                    clipped_reverse = mixed_kl_reverse * clip_scale
                    clipped_rkl_count = rkl_weight.sum()
                    clipped_rkl_mean = (
                        clipped_reverse * rkl_weight
                    ).sum() / clipped_rkl_count.clamp_min(1.0)
                    clipped_rkl_raw = (
                        clipped_rkl_mean * valid.sum().detach()
                    )
                    raw = raw + (
                        self.on_policy_clipped_rkl_alpha * clipped_rkl_raw
                    )
                    clipped_rkl_unclipped_mean = (
                        mixed_kl_reverse.detach() * rkl_weight
                    ).sum() / clipped_rkl_count.clamp_min(1.0)
                    clipped_rkl_capped_mean = (
                        clipped_reverse.detach() * rkl_weight
                    ).sum() / clipped_rkl_count.clamp_min(1.0)

                mixed_kl_accepted_mask = survived & valid.bool()
                mixed_kl_rejected_mask = ~survived & valid.bool()
                position_weight = torch.pow(
                    zero.new_tensor(self.on_policy_mixed_kl_rejection_decay),
                    depth.float(),
                ).view(1, -1)
                mixed_kl_rejected_weights = (
                    mixed_kl_rejected_mask.float() * position_weight
                )
                accepted_count = mixed_kl_accepted_mask.sum().float()
                rejected_effective_count = mixed_kl_rejected_weights.sum()
                accepted_mean = (
                    mixed_kl_forward * mixed_kl_accepted_mask.float()
                ).sum() / accepted_count.clamp_min(1.0)
                rejected_mean = (
                    mixed_kl_reverse * mixed_kl_rejected_weights
                ).sum() / rejected_effective_count.clamp_min(1.0)
                accepted_active = accepted_count.gt(0).float()
                rejected_active = rejected_effective_count.gt(0).float()
                stream_weight = (
                    self.on_policy_mixed_kl_accepted_weight * accepted_active
                    + self.on_policy_mixed_kl_rejected_weight * rejected_active
                )
                mixed_kl_mean = (
                    self.on_policy_mixed_kl_accepted_weight
                    * accepted_active
                    * accepted_mean
                    + self.on_policy_mixed_kl_rejected_weight
                    * rejected_active
                    * rejected_mean
                ) / stream_weight.clamp_min(1e-6)
                # `_on_policy_survival_objective` returns numerator-like losses;
                # restore selected-state mass after Draft-OPD's stream means so
                # the common DSpark denominator and loss budget remain valid.
                if self.on_policy_mixed_kl_alpha > 0:
                    mixed_kl_raw = mixed_kl_mean * valid.sum().detach()
                    raw = raw + self.on_policy_mixed_kl_alpha * mixed_kl_raw
            # This is a conservative lower bound on full-vocabulary overlap:
            # target tail mass is not credited unless it enters target Top-k.
            distributional_overlap = torch.minimum(
                draft_topk_probs, masked_target_probs
            ).sum(dim=-1).clamp(min=1e-6, max=1.0)
            overlap_for_survival = torch.where(
                valid.bool(),
                distributional_overlap,
                torch.ones_like(distributional_overlap),
            )
            distributional_survival = torch.cumprod(
                overlap_for_survival,
                dim=-1,
            )
            overlap_penalty = -torch.log(distributional_overlap)
            survival_penalty = 1.0 - distributional_survival
            distributional_penalty = (
                (1.0 - distributional_blend) * overlap_penalty
                + distributional_blend * survival_penalty
            )
            if self.on_policy_distributional_mass_threshold > 0:
                mass_confidence = (
                    distributional_retained_mass
                    / self.on_policy_distributional_mass_threshold
                ).clamp(max=1.0)
                distributional_penalty = (
                    distributional_penalty * mass_confidence
                )
            boundary_limit = (
                self.on_policy_boundary_max_depth
                if self.on_policy_boundary_max_depth > 0
                else self.on_policy_first_rejection_max_depth
            )
            distributional_boundary = valid.bool() & depth.lt(
                boundary_limit
            ).view(1, -1)
            distributional_deep = valid.bool() & deep_depth
            raw = raw + self.on_policy_distributional_alpha * (
                distributional_penalty
                * distributional_boundary.float()
                * language_credit
                * pareto_credit
            ).sum()
            raw = raw + self.on_policy_distributional_deep_alpha * (
                distributional_penalty
                * distributional_deep.float()
                * language_credit
                * pareto_credit
            ).sum()
            if self.on_policy_rejection_aligned_alpha > 0:
                with torch.no_grad():
                    proposal_matches = target_topk_ids.eq(
                        proposals.unsqueeze(-1)
                    ) & adaptive_keep
                    target_proposal_prob = (
                        masked_target_probs * proposal_matches.float()
                    ).sum(dim=-1)
                    draft_proposal_prob = draft_log_probs.gather(
                        -1, proposals.unsqueeze(-1)
                    ).squeeze(-1).exp()
                    rejection_acceptance_probability = torch.minimum(
                        torch.ones_like(target_proposal_prob),
                        target_proposal_prob
                        / draft_proposal_prob.clamp_min(1e-12),
                    )
                    acceptance_for_occupancy = torch.where(
                        valid.bool(),
                        rejection_acceptance_probability,
                        torch.ones_like(rejection_acceptance_probability),
                    )
                    rejection_occupancy_before = torch.cat(
                        [
                            torch.ones_like(acceptance_for_occupancy[:, :1]),
                            torch.cumprod(
                                acceptance_for_occupancy, dim=-1
                            )[:, :-1],
                        ],
                        dim=-1,
                    )
                    rejection_alignment_mask = valid.bool() & (
                        rollout_temperatures.gt(0).unsqueeze(-1)
                    )

                overlap_term = -torch.log(distributional_overlap)
                survival_term = 1.0 - distributional_overlap
                blend = self.on_policy_rejection_aligned_survival_blend
                rejection_penalty = (
                    (1.0 - blend) * overlap_term + blend * survival_term
                )
                mass_confidence = torch.ones_like(distributional_retained_mass)
                if self.on_policy_distributional_mass_threshold > 0:
                    mass_confidence = (
                        distributional_retained_mass
                        / self.on_policy_distributional_mass_threshold
                    ).clamp(max=1.0)
                rejection_alignment_raw = (
                    rejection_penalty
                    * rejection_occupancy_before.detach()
                    * rejection_alignment_mask.float()
                    * mass_confidence
                    * language_credit
                    * pareto_credit
                ).sum()
                raw = raw + (
                    self.on_policy_rejection_aligned_alpha
                    * rejection_alignment_raw
                )
                rejection_expected_length = (
                    rejection_occupancy_before
                    * distributional_overlap.detach()
                    * rejection_alignment_mask.float()
                ).sum(dim=-1)
        reset_replay_objective, reset_replay_diagnostics = (
            self._reset_replay_objective(
                input_ids=input_ids,
                attention_mask=attention_mask,
                anchors=anchors,
                base_logits=base,
                hidden_states=hidden,
                proposals=proposals,
                verifier_ids=verifier_ids,
                first_rejection=first_rejection,
                valid=valid,
                active_rows=active_rows,
                rollout_temperatures=rollout_temperatures,
                ce_num=ce_num,
            )
        )
        weighted = mix * raw
        objective, budget_scale = self._budget_auxiliary_loss(
            weighted, self.on_policy_loss_budget, ce_num
        )
        branch_objective, branch_diagnostics = self._branch_value_objective(
            input_ids=input_ids,
            attention_mask=attention_mask,
            anchors=anchors,
            base_logits=base,
            hidden_states=hidden,
            proposals=proposals,
            rollout_logits=logits,
            first_rejection=first_rejection,
            active_rows=active_rows,
            ce_num=ce_num,
        )
        objective = (
            objective
            + branch_objective
            + refiner_objective
            + advantage_objective
            + reset_replay_objective
            + multi_teacher_objective
            + preference_objective
        )
        denom = active_rows.sum().clamp_min(1)
        diagnostics = {
            "on_policy_mix": zero.new_tensor(mix),
            "on_policy_scored_blocks": active_rows.sum().detach(),
            "on_policy_first_rejection_count": first_rejection.sum().detach(),
            "on_policy_shallow_repair_count": shallow_fr.sum().detach(),
            "on_policy_boundary_credit_count": (
                (
                    strict_early
                    if self.on_policy_credit_partition == "strict-v2"
                    else boundary_protect
                ).sum().detach()
            ),
            "on_policy_middle_credit_count": strict_middle.sum().detach(),
            "on_policy_deep_credit_count": (
                (
                    strict_deep
                    if self.on_policy_credit_partition == "strict-v2"
                    else deep_protect
                ).sum().detach()
            ),
            "on_policy_credit_strict_v2": zero.new_tensor(
                float(self.on_policy_credit_partition == "strict-v2")
            ),
            "on_policy_credit_mask_overlap_rate": (
                (
                    (strict_early & strict_middle)
                    | (strict_early & strict_deep)
                    | (strict_middle & strict_deep)
                ).sum().float()
                / valid.sum().clamp_min(1)
            ).detach(),
            "on_policy_hazard_credit_mean": (
                (hazard_credit * valid).sum() / valid.sum().clamp_min(1)
            ).detach(),
            "on_policy_boundary_max_depth": zero.new_tensor(
                self.on_policy_boundary_max_depth
            ),
            "on_policy_deep_start_depth": zero.new_tensor(
                self.on_policy_deep_start_depth
            ),
            "on_policy_full_survival_rate": full_survival.sum().float() / denom,
            "on_policy_mean_survival": survived.sum().float() / denom,
            "on_policy_budget_scale": budget_scale.detach(),
            "on_policy_preservation_count": preservation_mask.sum().detach(),
            "on_policy_preservation_loss": preservation_raw.detach(),
            "on_policy_regression_count": regression_mask.sum().detach(),
            "on_policy_regression_loss": regression_raw.detach(),
            "on_policy_greedy_preservation_count": (
                greedy_preservation_mask.sum().detach()
            ),
            "on_policy_greedy_preservation_loss": (
                greedy_preservation_raw.detach()
            ),
            "on_policy_rollout_temperature_mean": (
                rollout_temperatures[active_rows].mean()
            ).detach(),
            "on_policy_temperature_exclusive_routing": zero.new_tensor(
                float(self.on_policy_temperature_exclusive_routing)
            ),
            "on_policy_sr_routed_row_count": (
                (active_rows & rollout_temperatures.le(1e-6)).sum().detach()
                if self.on_policy_temperature_exclusive_routing
                else active_rows.sum().detach()
            ),
            "on_policy_pq_routed_row_count": (
                (active_rows & rollout_temperatures.gt(1e-6)).sum().detach()
            ),
            "on_policy_sr_routed_raw_loss": sr_raw.detach(),
            "on_policy_selected_block_index_mean": (
                block_index[active_rows].float().mean()
            ).detach(),
            "on_policy_sampling_entropy": (
                -(sampling_weights.clamp_min(1e-12).log() * sampling_weights)
                .sum(dim=-1)[active_rows]
                .mean()
            ).detach(),
            "on_policy_flatness_power": zero.new_tensor(
                self._on_policy_flatness_power()
            ),
            "on_policy_distributional_blend": zero.new_tensor(
                distributional_blend
            ),
            "rejection_aligned_raw_loss": rejection_alignment_raw.detach(),
            "rejection_aligned_survival_blend": zero.new_tensor(
                self.on_policy_rejection_aligned_survival_blend
            ),
            "mixed_kl_raw_loss": mixed_kl_raw.detach(),
            "mixed_kl_rejection_decay": zero.new_tensor(
                self.on_policy_mixed_kl_rejection_decay
            ),
            "clipped_rkl_raw_loss": clipped_rkl_raw.detach(),
            "clipped_rkl_state_mass": clipped_rkl_count.detach(),
            "clipped_rkl_unclipped_mean": clipped_rkl_unclipped_mean.detach(),
            "clipped_rkl_capped_mean": clipped_rkl_capped_mean.detach(),
            "clipped_rkl_clip": zero.new_tensor(self.on_policy_clipped_rkl_clip),
            "target_margin_frontier_loss": target_margin_frontier_raw.detach(),
            "target_margin_frontier_count": target_margin_frontier_count.detach(),
            "target_margin_teacher_mean": target_margin_mean.detach(),
            "target_margin_required_mean": target_margin_required_mean.detach(),
        }
        diagnostics.update(pareto_diagnostics)
        diagnostics.update(marginal_value_diagnostics)
        diagnostics.update(language_hazard_diagnostics)
        diagnostics.update(refiner_diagnostics)
        diagnostics.update(advantage_diagnostics)
        diagnostics.update(reset_replay_diagnostics)
        diagnostics.update(repair_value_diagnostics)
        diagnostics.update(preference_diagnostics)
        diagnostics.update(multi_teacher_diagnostics)
        diagnostics.update(self.draft_model.parallel_refiner_diagnostics())
        if self.on_policy_temperature_hazard_credit:
            for temperature_index, temperature in enumerate(
                self.on_policy_rollout_temperatures
            ):
                temperature_label = str(temperature).replace(".", "p")
                for depth_index in range(self.block_size):
                    diagnostics[
                        f"on_policy_hazard_t{temperature_label}_depth_{depth_index + 1}"
                    ] = self.on_policy_temperature_hazard_ema[
                        temperature_index, depth_index
                    ].detach()
        diagnostics.update(gate_calibration_diagnostics)
        diagnostics.update(branch_diagnostics)
        if self.dialogue_occupancy_diagnostics and dialogue_features is not None:
            selected_phase = dialogue_features["phase"][row, block_index]
            selected_boundary = dialogue_features["boundary"][row, block_index]
            selected_transition = dialogue_features["script_transition"][
                row, block_index
            ]
            selected_fraction = dialogue_features["fraction"][row, block_index]
            rejected_rows = first_rejection.any(dim=-1)
            accepted_length = survived.sum(dim=-1).float()
            diagnostics.update(
                {
                    "dialogue_selected_response_fraction": (
                        (selected_fraction * active_rows).sum() / denom
                    ),
                    "dialogue_selected_boundary_rate": (
                        (selected_boundary & active_rows).sum().float() / denom
                    ),
                    "dialogue_selected_script_transition_rate": (
                        (selected_transition & active_rows).sum().float() / denom
                    ),
                }
            )
            candidate_count = active.sum().clamp_min(1)
            for phase_index in range(5):
                phase_candidates = active & dialogue_features["phase"].eq(
                    phase_index
                )
                phase_rows = active_rows & selected_phase.eq(phase_index)
                phase_count = phase_rows.sum().clamp_min(1)
                diagnostics.update(
                    {
                        f"dialogue_candidate_phase_{phase_index + 1}_rate": (
                            phase_candidates.sum().float() / candidate_count
                        ).detach(),
                        f"dialogue_selected_phase_{phase_index + 1}_rate": (
                            phase_rows.sum().float() / denom
                        ).detach(),
                        f"dialogue_phase_{phase_index + 1}_mean_survival": (
                            (accepted_length * phase_rows).sum() / phase_count
                        ).detach(),
                        f"dialogue_phase_{phase_index + 1}_rejection_rate": (
                            (rejected_rows & phase_rows).sum().float() / phase_count
                        ).detach(),
                    }
                )
            if source_ids is not None:
                for source_index in range(self.dialogue_num_sources):
                    source_rows = active_rows & source_ids.eq(source_index)
                    source_count = source_rows.sum().clamp_min(1)
                    diagnostics.update(
                        {
                            f"dialogue_source_{source_index}_selected_rate": (
                                source_rows.sum().float() / denom
                            ).detach(),
                            f"dialogue_source_{source_index}_mean_survival": (
                                (accepted_length * source_rows).sum() / source_count
                            ).detach(),
                            f"dialogue_source_{source_index}_rejection_rate": (
                                (rejected_rows & source_rows).sum().float()
                                / source_count
                            ).detach(),
                        }
                    )
            for temperature_index, temperature in enumerate(
                self.on_policy_rollout_temperatures
            ):
                temperature_rows = active_rows & temperature_choices.eq(
                    temperature_index
                )
                temperature_count = temperature_rows.sum().clamp_min(1)
                temperature_label = str(temperature).replace(".", "p")
                diagnostics.update(
                    {
                        f"dialogue_t{temperature_label}_selected_rate": (
                            temperature_rows.sum().float() / denom
                        ).detach(),
                        f"dialogue_t{temperature_label}_mean_survival": (
                            (accepted_length * temperature_rows).sum()
                            / temperature_count
                        ).detach(),
                        f"dialogue_t{temperature_label}_rejection_rate": (
                            (rejected_rows & temperature_rows).sum().float()
                            / temperature_count
                        ).detach(),
                    }
                )
                for phase_index in range(5):
                    phase_temperature_rows = temperature_rows & selected_phase.eq(
                        phase_index
                    )
                    phase_temperature_count = phase_temperature_rows.sum().clamp_min(1)
                    metric_name = (
                        f"dialogue_t{temperature_label}_phase_"
                        f"{phase_index + 1}_mean_survival"
                    )
                    diagnostics[metric_name] = (
                        (accepted_length * phase_temperature_rows).sum()
                        / phase_temperature_count
                    ).detach()
        if block_hazard is not None:
            diagnostics["on_policy_selected_hazard"] = (
                block_hazard[row, block_index][active_rows].float().mean().detach()
            )
        if block_flatness is not None:
            diagnostics["on_policy_selected_flatness"] = (
                block_flatness[row, block_index][active_rows]
                .float()
                .mean()
                .detach()
            )
        if distributional_overlap is not None:
            target_topk_mass = target_topk_probs.sum(dim=-1)
            diagnostics["on_policy_target_topk_mass"] = (
                (target_topk_mass * valid).sum() / valid.sum().clamp_min(1)
            ).detach()
            diagnostics["on_policy_adaptive_retained_mass"] = (
                (distributional_retained_mass * valid).sum()
                / valid.sum().clamp_min(1)
            ).detach()
            diagnostics["on_policy_adaptive_effective_top_k"] = (
                (distributional_effective_k.float() * valid).sum()
                / valid.sum().clamp_min(1)
            ).detach()
            diagnostics["on_policy_adaptive_cap_limited_rate"] = (
                (distributional_cap_limited.float() * valid).sum()
                / valid.sum().clamp_min(1)
            ).detach()
            diagnostics["overlap_lower_bound_gap"] = (
                ((1.0 - distributional_retained_mass) * valid).sum()
                / valid.sum().clamp_min(1)
            ).detach()
            if mixed_kl_forward is not None:
                accepted_count = mixed_kl_accepted_mask.sum().clamp_min(1)
                rejected_count = mixed_kl_rejected_mask.sum().clamp_min(1)
                rejected_effective_count = mixed_kl_rejected_weights.sum().clamp_min(
                    1e-6
                )
                diagnostics["mixed_kl_forward_accepted"] = (
                    (
                        mixed_kl_forward
                        * mixed_kl_accepted_mask.float()
                    ).sum()
                    / accepted_count
                ).detach()
                diagnostics["mixed_kl_reverse_rejected"] = (
                    (
                        mixed_kl_reverse
                        * mixed_kl_rejected_weights
                    ).sum()
                    / rejected_effective_count
                ).detach()
                diagnostics["mixed_kl_accepted_count"] = (
                    mixed_kl_accepted_mask.sum().detach()
                )
                diagnostics["mixed_kl_rejected_count"] = (
                    mixed_kl_rejected_mask.sum().detach()
                )
                diagnostics["mixed_kl_rejected_unweighted_mean"] = (
                    (
                        mixed_kl_reverse
                        * mixed_kl_rejected_mask.float()
                    ).sum()
                    / rejected_count
                ).detach()
                diagnostics["mixed_kl_tail_mass_target"] = (
                    (target_tail.squeeze(-1) * valid).sum()
                    / valid.sum().clamp_min(1)
                ).detach()
                diagnostics["mixed_kl_tail_mass_draft"] = (
                    (draft_tail.squeeze(-1) * valid).sum()
                    / valid.sum().clamp_min(1)
                ).detach()
            if rejection_acceptance_probability is not None:
                stochastic_valid = rejection_alignment_mask.float()
                stochastic_count = stochastic_valid.sum().clamp_min(1)
                # Greedy rows have near-unit retained mass. Report stochastic
                # coverage separately so that they cannot hide tail truncation.
                diagnostics["ra_stochastic_token_count"] = stochastic_valid.sum().detach()
                diagnostics["ra_stochastic_retained_mass"] = (
                    (distributional_retained_mass * stochastic_valid).sum()
                    / stochastic_count
                ).detach()
                diagnostics["ra_stochastic_cap_limited_rate"] = (
                    (distributional_cap_limited.float() * stochastic_valid).sum()
                    / stochastic_count
                ).detach()
                diagnostics["ra_stochastic_proposal_outside_retained_rate"] = (
                    (~proposal_matches.any(dim=-1)).float().mul(stochastic_valid).sum()
                    / stochastic_count
                ).detach()
                diagnostics["sampled_acceptance_probability"] = (
                    (rejection_acceptance_probability * stochastic_valid).sum()
                    / stochastic_count
                ).detach()
                diagnostics["hard_vs_soft_credit_disagreement"] = (
                    (
                        (
                            accepted.float()
                            - rejection_acceptance_probability
                        ).abs()
                        * stochastic_valid
                    ).sum()
                    / stochastic_count
                ).detach()
                stochastic_rows = (
                    active_rows & rollout_temperatures.gt(0)
                )
                diagnostics["rejection_aligned_expected_accept_length"] = (
                    rejection_expected_length[stochastic_rows].sum()
                    / stochastic_rows.sum().clamp_min(1)
                ).detach()
                diagnostics["rejection_aligned_occupancy_mean"] = (
                    (rejection_occupancy_before * stochastic_valid).sum()
                    / stochastic_count
                ).detach()
            for depth_idx in range(active_block_size):
                depth_valid = valid[:, depth_idx].sum().clamp_min(1)
                diagnostics[
                    f"on_policy_adaptive_effective_top_k_{depth_idx + 1}"
                ] = (
                    (
                        distributional_effective_k[:, depth_idx].float()
                        * valid[:, depth_idx]
                    ).sum()
                    / depth_valid
                ).detach()
                diagnostics[
                    f"on_policy_adaptive_retained_mass_{depth_idx + 1}"
                ] = (
                    (
                        distributional_retained_mass[:, depth_idx]
                        * valid[:, depth_idx]
                    ).sum()
                    / depth_valid
                ).detach()
                diagnostics[f"on_policy_overlap_{depth_idx + 1}"] = (
                    (distributional_overlap[:, depth_idx] * valid[:, depth_idx]).sum()
                    / depth_valid
                ).detach()
                diagnostics[f"on_policy_distributional_survival_{depth_idx + 1}"] = (
                    (
                        distributional_survival[:, depth_idx]
                        * valid[:, depth_idx]
                    ).sum()
                    / depth_valid
                ).detach()
            if rejection_acceptance_probability is not None:
                for temperature_index, configured_temperature in enumerate(
                    self.on_policy_rollout_temperatures
                ):
                    if configured_temperature <= 0:
                        continue
                    temperature_rows = (
                        temperature_choices.eq(temperature_index) & active_rows
                    )
                    temperature_valid = (
                        valid.bool() & temperature_rows.unsqueeze(-1)
                    ).float()
                    temperature_count = temperature_valid.sum().clamp_min(1)
                    label = str(configured_temperature).replace(".", "p")
                    diagnostics[f"rejection_overlap_t{label}"] = (
                        (distributional_overlap * temperature_valid).sum()
                        / temperature_count
                    ).detach()
        for idx in range(self.block_size):
            diagnostics[f"on_policy_survival_{idx + 1}"] = (
                survived[:, idx].sum().float() / denom
                if idx < active_block_size
                else zero.detach()
            )
        for temperature_idx, configured_temperature in enumerate(
            self.on_policy_rollout_temperatures
        ):
            temperature_rows = (
                temperature_choices.eq(temperature_idx) & active_rows
            )
            temperature_count = temperature_rows.sum()
            temperature_denom = temperature_count.clamp_min(1)
            prefix = f"on_policy_rollout_temperature_{temperature_idx}"
            diagnostics[f"{prefix}_value"] = zero.new_tensor(
                configured_temperature
            )
            diagnostics[f"{prefix}_fraction"] = (
                temperature_count.float() / denom
            )
            diagnostics[f"{prefix}_mean_survival"] = (
                survived[temperature_rows].sum().float() / temperature_denom
            )
            diagnostics[f"{prefix}_full_survival_rate"] = (
                full_survival[temperature_rows].sum().float() / temperature_denom
            )
            for depth_idx in range(self.block_size):
                diagnostics[f"{prefix}_survival_{depth_idx + 1}"] = (
                    survived[temperature_rows, depth_idx].sum().float()
                    / temperature_denom
                    if depth_idx < active_block_size
                    else zero.detach()
                )
        return objective, diagnostics

    def _effective_pace_blend(self) -> float:
        """Blend fixed depth prior into PACE credit with warmup and ramp.

        Defaults preserve the original full-replacement experiments.  When a
        schedule is requested, the training script supplies optimizer-step
        progress through ``set_training_progress`` inherited from DFlash.
        """
        if self.pace_mode == "none" or self.pace_blend_max <= 0:
            return 0.0
        if (
            self.pace_warmup_ratio == 0
            and self.pace_ramp_ratio == 0
            and self.pace_decay_start_ratio >= 1.0
        ):
            return self.pace_blend_max
        if self._total_steps <= 0:
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.pace_warmup_ratio:
            return 0.0
        if self.pace_ramp_ratio > 0:
            ramp_end = self.pace_warmup_ratio + self.pace_ramp_ratio
            if progress < ramp_end:
                return self.pace_blend_max * (
                    progress - self.pace_warmup_ratio
                ) / self.pace_ramp_ratio
        if progress < self.pace_decay_start_ratio:
            return self.pace_blend_max
        if self.pace_decay_start_ratio >= 1.0:
            return self.pace_blend_max
        decay_progress = (progress - self.pace_decay_start_ratio) / (
            1.0 - self.pace_decay_start_ratio
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.pace_blend_final + (
            self.pace_blend_max - self.pace_blend_final
        ) * cosine

    def _offline_acceptance_e2e_blend(self) -> float:
        """Return the AngelSpec cold-start to end-to-end objective blend.

        The stable D-PACE-weighted LK objective is used first.  It is then
        linearly replaced by the multiplicative expected-accepted-length
        surrogate.  The schedule is expressed in optimizer-step units through
        ``set_training_progress``, so gradient accumulation does not shorten
        either phase.
        """
        if self.offline_acceptance_objective == "none":
            return 0.0
        if self._total_steps <= 0:
            return 0.0
        progress = min(max(self._global_step / self._total_steps, 0.0), 1.0)
        cold_end = self.offline_acceptance_cold_start_ratio
        if progress <= cold_end:
            return 0.0
        transition = self.offline_acceptance_transition_ratio
        if transition <= 0:
            return 1.0
        return min(max((progress - cold_end) / transition, 0.0), 1.0)

    def _pace_weights(
        self,
        quality: torch.Tensor,
        eval_mask: torch.Tensor,
        reference_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Build detached continuation credit and preserve baseline loss mass.

        ``quality`` is a per-position acceptance proxy.  Its smoothed prefix
        product estimates survival through each depth; a reverse cumulative
        sum assigns each position the expected continuation it supports.  The
        result is matched per sample to DSpark's original exponential weight
        mass, so changing credit allocation does not change the effective LR.
        """
        with torch.no_grad():
            valid = eval_mask.bool()
            q = quality.detach().float().clamp(0.0, 1.0)
            smooth_q = self.pace_alpha + (1.0 - self.pace_alpha) * q
            smooth_q = torch.where(valid, smooth_q, torch.ones_like(smooth_q))
            survival = torch.cumprod(smooth_q, dim=-1)
            raw_credit = torch.flip(
                torch.cumsum(
                    torch.flip(survival * eval_mask, dims=[-1]), dim=-1
                ),
                dims=[-1],
            )
            raw_weights = raw_credit * eval_mask
            reduce_dims = tuple(range(1, raw_weights.dim()))
            reference_mass = reference_weights.sum(
                dim=reduce_dims, keepdim=True
            )
            raw_mass = raw_weights.sum(dim=reduce_dims, keepdim=True)
            mass_scale = reference_mass / raw_mass.clamp_min(1e-6)
            mass_scale = torch.where(
                raw_mass > 0, mass_scale, torch.zeros_like(mass_scale)
            )
            weights = raw_weights * mass_scale
            valid_count = eval_mask.sum().clamp_min(1.0)
            diagnostics = {
                "pace_quality_mean": (q * eval_mask).sum() / valid_count,
                "pace_survival_mean": (survival * eval_mask).sum()
                / valid_count,
                "pace_weight_mass_ratio": weights.sum()
                / reference_weights.sum().clamp_min(1e-6),
                "pace_mass_scale_mean": mass_scale.sum()
                / mass_scale.gt(0).float().sum().clamp_min(1.0),
            }
        return weights, diagnostics

    def _rollout_quality(
        self,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the lightweight Markov chain with self-generated predecessors.

        DSpark's transformer logits are block-parallel.  Only the low-rank
        Markov bias depends on the previous token, so an inference-aligned
        rollout requires ``block_size`` inexpensive head applications rather
        than another transformer forward.  The rollout is a detached credit
        estimator; the auxiliary objective reuses the already-computed CE, so
        no second vocabulary-sized backward graph is retained.
        """
        if self.draft_model.markov_head is None:
            raise ValueError("rollout-residual PACE requires a Markov head")

        step_quality = []
        prev_ids = anchor_token_ids
        markov_head = self.draft_model.markov_head
        memory_enabled = getattr(markov_head, "sampled_prefix_memory", None) is not None
        memory_state = (
            markov_head.init_sampled_prefix_state(prev_ids)
            if memory_enabled else None
        )
        sampled_steps = []
        with torch.no_grad():
            for depth in range(self.block_size):
                if memory_enabled and depth >= 2:
                    memory_state = markov_head.advance_sampled_prefix_state(
                        memory_state, sampled_steps[depth - 2]
                    )
                logits_d = base_logits[:, :, depth, :] + (
                    markov_head.compute_step_bias(
                        prev_ids,
                        hidden_states[:, :, depth, :],
                        depth_idx=depth,
                        **_carh_predecessor_context_kwargs(
                            markov_head, hidden_states, depth
                        ),
                        **({"sampled_prefix_state": memory_state} if memory_enabled else {}),
                    )
                )
                log_q = F.log_softmax(logits_d.float(), dim=-1).gather(
                    -1, target_ids[:, :, depth].unsqueeze(-1)
                )
                step_quality.append(log_q.squeeze(-1).exp())
                prev_ids = logits_d.argmax(dim=-1)
                sampled_steps.append(prev_ids)

        return torch.stack(step_quality, dim=2)

    def _first_rejection_mask(
        self,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Locate the first inference-time rejection in each sampled block."""
        if self.draft_model.markov_head is None:
            raise ValueError("prefix credit requires a causal Markov/CARH head")
        first_rejection = torch.zeros_like(eval_mask, dtype=torch.bool)
        alive = eval_mask[..., 0].bool()
        prev_ids = anchor_token_ids
        markov_head = self.draft_model.markov_head
        memory_enabled = getattr(markov_head, "sampled_prefix_memory", None) is not None
        memory_state = (
            markov_head.init_sampled_prefix_state(prev_ids)
            if memory_enabled else None
        )
        sampled_steps = []
        correct_steps = []
        with torch.no_grad():
            for depth in range(self.block_size):
                if memory_enabled and depth >= 2:
                    memory_state = markov_head.advance_sampled_prefix_state(
                        memory_state, sampled_steps[depth - 2]
                    )
                bias = markov_head.compute_step_bias(
                    prev_ids,
                    hidden_states[:, :, depth, :],
                    depth_idx=depth,
                    **_carh_predecessor_context_kwargs(
                        markov_head, hidden_states, depth
                    ),
                    **({"sampled_prefix_state": memory_state} if memory_enabled else {}),
                )
                pred_ids = (base_logits[:, :, depth, :] + bias).argmax(dim=-1)
                valid = eval_mask[:, :, depth].bool()
                correct = pred_ids.eq(target_ids[:, :, depth]) & valid
                first_rejection[:, :, depth] = alive & valid & ~correct
                # Inactive Elastic-Horizon suffix positions are neutral: they
                # neither reject the active prefix nor turn a fully surviving
                # K=7 view into a failed K=10 view in diagnostics.
                correct_steps.append(correct | ~valid)
                alive = alive & (correct | ~valid)
                prev_ids = pred_ids
                sampled_steps.append(pred_ids)

            correct_stack = torch.stack(correct_steps, dim=-1)
            survival = torch.cumprod(correct_stack.to(torch.int32), dim=-1).float()
            valid_blocks = eval_mask[..., 0].sum().clamp_min(1.0)
            valid_depth = eval_mask.sum(dim=-1).long()
            last_valid_depth = (valid_depth - 1).clamp_min(0)
            full_survival = survival.gather(
                -1, last_valid_depth.unsqueeze(-1)
            ).squeeze(-1)
            depth_values = torch.arange(
                1,
                self.block_size + 1,
                device=base_logits.device,
                dtype=torch.float32,
            )
            rejected_blocks = first_rejection.any(dim=-1)
            first_depth = (first_rejection.float() * depth_values).sum(dim=-1)
            first_depth = torch.where(
                rejected_blocks,
                first_depth,
                valid_depth.to(first_depth.dtype) + 1.0,
            )
            diagnostics = {
                "prefix_first_rejection_depth": (
                    first_depth * eval_mask[..., 0]
                ).sum()
                / valid_blocks,
                "prefix_full_accept_rate": (
                    full_survival * eval_mask[..., 0]
                ).sum()
                / valid_blocks,
            }
            for depth in range(self.block_size):
                diagnostics[f"prefix_survival_{depth + 1}"] = (
                    survival[..., depth] * eval_mask[..., depth]
                ).sum() / valid_blocks
        return first_rejection.float(), diagnostics

    def _rollout_residual_weights(
        self,
        quality: torch.Tensor,
        eval_mask: torch.Tensor,
        reference_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Bound PACE's log-credit advantage over DSpark's depth prior."""
        pace_weights, diagnostics = self._pace_weights(
            quality, eval_mask, reference_weights
        )
        with torch.no_grad():
            valid = eval_mask.bool()
            eps = torch.finfo(torch.float32).tiny
            log_ratio = torch.log(
                pace_weights.float().clamp_min(eps)
                / reference_weights.float().clamp_min(eps)
            )
            valid_count = eval_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            centered = log_ratio - (
                log_ratio * eval_mask
            ).sum(dim=-1, keepdim=True) / valid_count
            modifier = torch.exp(self.pace_residual_beta * centered).clamp(
                self.pace_residual_min, self.pace_residual_max
            )
            modifier = torch.where(valid, modifier, torch.zeros_like(modifier))
            residual_weights = reference_weights * modifier
            global_valid = eval_mask.sum().clamp_min(1.0)
            diagnostics.update(
                {
                    "pace_residual_modifier_mean": (
                        modifier * eval_mask
                    ).sum()
                    / global_valid,
                    "pace_residual_modifier_min": torch.where(
                        valid, modifier, torch.ones_like(modifier)
                    ).amin(),
                    "pace_residual_modifier_max": modifier.amax(),
                }
            )
        return residual_weights, diagnostics

    def _decay_weights(self, device: torch.device) -> torch.Tensor:
        """exp(-k/gamma) over within-block position k (DeepSpec convention).

        Every slot 0..B-1 is a real prediction in DSpark (unlike DFlash, where
        slot 0 is the masked anchor), so slot 0 (the first predicted token) gets
        weight 1.0 and later slots decay.
        """
        k = torch.arange(self.block_size, device=device).view(1, 1, -1)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            return torch.exp(-k.float() / self.loss_decay_gamma)
        return torch.ones_like(k, dtype=torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        source_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """DSpark training forward.

        ``hidden_states`` is the fused multi-layer context feature
        ``[B, S, len(target_layer_ids)*hidden]`` (the draft model applies its
        ``fc``/``hidden_norm`` internally). ``last_hidden_states`` is the target
        model's final hidden state ``[B, S, hidden]`` (needed by distributional
        L1, confidence, and offline acceptance-aligned objectives).

        Returns ``(loss, accuracy, loss_per_position, acc_per_position,
        count_per_position, loss_components)``. ``loss`` is the configured
        combined objective; ``loss_components`` contains detached diagnostic
        scalars for logging.
        """
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        memory_head = getattr(self.draft_model, "markov_head", None)
        if getattr(memory_head, "sampled_prefix_memory", None) is not None:
            memory_head.sampled_prefix_memory_scale = (
                self._sampled_prefix_memory_scale()
            )
        active_block_size = self._elastic_active_horizon()
        paired_projective = self._elastic_pair_enabled()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # ---- DFlash backbone (identical construction to OnlineDFlashModel.forward) ----
        if getattr(self, "tv_acceptance_enabled", False):
            from specforge.core.tv_acceptance import sample_tv_anchors

            anchor_positions, block_keep_mask = sample_tv_anchors(
                loss_mask, attention_mask, self.num_anchors,
            )
        else:
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
        n_blocks = anchor_positions.shape[1]

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
                active_block_size=active_block_size,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
                active_block_size=active_block_size,
            )

        # ---- Labels + eval mask (DSpark / DeepSpec convention) ----
        # Slot j predicts the token at anchor+j+1 (the real anchor token seeds
        # slot 0). All block_size slots are supervised — there is no masked anchor
        # slot, unlike SpecForge DFlash which drops slot 0.
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, nb, bs]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )  # [B, nb, bs]

        # eval mask = contiguous supervised prefix per block (DeepSpec
        # build_eval_mask): block kept, label in-bounds, target token supervised,
        # then cumprod so a gap truncates the rest of the block.
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        eval_bool = (
            block_keep_mask.unsqueeze(-1) & valid_label_mask & (target_loss_mask > 0.5)
        )
        eval_bool = eval_bool.to(torch.int32).cumprod(dim=-1).bool()
        active_depth_mask = torch.arange(
            self.block_size, device=device
        ).lt(active_block_size).view(1, 1, -1)
        eval_bool = eval_bool & active_depth_mask
        eval_mask = eval_bool.float()  # [B, nb, bs]

        decay_weight_mask = eval_mask * self._decay_weights(device)
        local_den = decay_weight_mask.sum()

        # Training-only predictive auxiliaries. Labels enter losses only, never
        # the backbone's feature stream (its within-block attention is bidirectional).
        predictive_scale = auxiliary_scale(
            self._global_step, self._total_steps,
            self.predictive_aux_warmup_ratio, self.predictive_aux_ramp_ratio,
        )
        predictive_weights = predecessor_source_weights(eval_mask, self._decay_weights(device))
        predictive_den = predictive_weights.sum()
        semantic_kwargs = {}
        if self.conv_source_semantic_alpha > 0:
            semantic_kwargs = dict(
                source_semantic_codes=F.embedding(target_ids, self.source_semantic_codebook).reshape(
                    bsz, n_blocks * self.block_size, -1),
                source_semantic_weights=predictive_weights.reshape(bsz, -1),
            )
        draft_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
            **semantic_kwargs,
        )
        semantic_num = local_den.new_zeros(())
        if semantic_kwargs:
            draft_hidden, semantic_num = draft_hidden
        hidden_4d = draft_hidden.view(bsz, n_blocks, self.block_size, -1)
        hidden_4d = self.draft_model.apply_prefix_state_mixer(hidden_4d)
        hidden_before_summary = hidden_4d
        hidden_4d = self.draft_model.apply_block_summary(hidden_4d)
        if getattr(self, "tv_acceptance_enabled", False):
            from specforge.core.tv_acceptance import candidate_tv_forward

            return candidate_tv_forward(
                self, input_ids, attention_mask, anchor_positions, hidden_4d,
                eval_bool,
            )
        reference_num = reference_cosine_num = reference_mse_num = reference_ratio_num = local_den.new_zeros(())
        if self.carh_reference_calibration_alpha > 0:
            (reference_num, reference_cosine_num, reference_mse_num,
             reference_ratio_num) = reference_calibration_terms(
                self.draft_model.markov_head, hidden_4d, target_ids, predictive_weights,
            )

        # ---- Markov-biased draft logits ----
        # prev token for slot j is the ground-truth token immediately before the
        # one slot j predicts: slot 0's prev is the real anchor token, slot j's is
        # target_ids[j-1]. Matches DeepSpec prev_token_ids.
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)  # [B, nb]
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
        )
        unary_logits_4d = self.lm_head(
            hidden_4d.reshape(-1, hidden_4d.size(-1))
        ).view(bsz, n_blocks, self.block_size, -1)
        hidden_for_logits = hidden_4d
        recall_hidden_delta = None
        recall_miss_mask = None
        if self.draft_model.recall_correction is not None:
            depth_ids = torch.arange(
                self.block_size, device=device, dtype=torch.long
            ).view(1, 1, -1).expand_as(prev_token_ids)
            hidden_for_logits = self.draft_model.apply_recall_correction(
                hidden_4d, prev_token_ids, depth_ids
            )
            recall_hidden_delta = hidden_for_logits - hidden_4d
            # The main DSpark objective must see the exact serving value while
            # updating only the unary backbone.  Build the corrected logits
            # without an autograd graph; the dedicated miss-only objective
            # below recomputes only the rows that train the Recall Expert.
            with torch.no_grad():
                recall_logits_4d = self.lm_head(
                    hidden_for_logits.reshape(-1, hidden_for_logits.size(-1))
                ).view(bsz, n_blocks, self.block_size, -1)
            with torch.no_grad():
                recall_candidates = unary_logits_4d.detach().topk(
                    self.recall_partition_top_k, dim=-1
                ).indices
                recall_miss_mask = (
                    eval_bool
                    & ~recall_candidates.eq(target_ids.unsqueeze(-1)).any(dim=-1)
                )
            base_logits_4d = _forward_value_reference_gradient(
                unary_logits_4d, recall_logits_4d
            )
        else:
            base_logits_4d = unary_logits_4d
        base_logits = base_logits_4d.reshape(
            bsz, n_blocks * self.block_size, -1
        )
        vocab_size = base_logits_4d.size(-1)
        logits_4d = base_logits_4d
        carh_candidate_ids = None
        carh_correct_mask = None
        carh_recoverable_mask = None
        carh_recall_miss_mask = None
        carh_residual_bias = None
        carh_correction_latent = None
        recoverability_enabled = self.carh_recoverability_top_k > 0
        if self.draft_model.markov_head is not None:
            if recoverability_enabled:
                (
                    carh_candidate_ids,
                    carh_correct_mask,
                    carh_recoverable_mask,
                    carh_recall_miss_mask,
                ) = self._carh_recoverability_partition(
                    base_logits_4d,
                    target_ids,
                    eval_mask,
                    self.carh_recoverability_top_k,
                )
                # CARH is a ranking corrector, not a second backbone.  Detach
                # the hidden-state input so correction losses update only the
                # causal head; the unary backbone remains responsible for
                # moving recall misses into Top-k.
                (
                    carh_residual_bias,
                    carh_correction_latent,
                ) = self.draft_model.markov_head.compute_block_bias_and_latent(
                    token_ids=prev_token_ids,
                    hidden_states=hidden_4d.detach(),
                )
                recoverable = carh_recoverable_mask.unsqueeze(-1)
                # Preserve the serving value base+residual exactly, while only
                # routing the primary CE/L1 gradient into CARH on recoverable
                # Top-k ranking errors. Correct and recall-miss states train the
                # backbone but see a detached causal residual.
                routed_residual = torch.where(
                    recoverable,
                    carh_residual_bias,
                    carh_residual_bias.detach(),
                )
                logits_4d = base_logits_4d + routed_residual
            else:
                logits_4d = self.draft_model.markov_head.apply_block_logits(
                    base_logits_4d,
                    token_ids=prev_token_ids,
                    hidden_states=hidden_4d,
                )

        # ---- Cross entropy (hard labels) ----
        flat_logits = logits_4d.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        ce_per_token = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        ).view(bsz, n_blocks, self.block_size)
        ce_weight_mask = decay_weight_mask

        # ---- L1 distribution distillation + accept rate ----
        bv_terms = None
        current_bv_beta = bv_beta(self._global_step, self._total_steps, self.bv_anneal_ratio)
        l1_num = base_logits.new_zeros((), dtype=torch.float32)
        offline_lk_num = base_logits.new_zeros((), dtype=torch.float32)
        offline_lk_den = base_logits.new_zeros((), dtype=torch.float32)
        offline_e2e_num = base_logits.new_zeros((), dtype=torch.float32)
        offline_e2e_den = base_logits.new_zeros((), dtype=torch.float32)
        offline_acceptance_mean = base_logits.new_zeros((), dtype=torch.float32)
        offline_expected_length = base_logits.new_zeros((), dtype=torch.float32)
        offline_lk_lambda_mean = base_logits.new_zeros((), dtype=torch.float32)
        accept_rate = None
        block_hazard = None
        block_flatness = None
        carh_pair_diagnostics = {}
        pair_interval = self.carh_predecessor_diagnostics_interval
        pair_active = (
            pair_interval > 0 and self._global_step % pair_interval == 0
            and getattr(self.draft_model.markov_head, "predecessor_count", 1) >= 2
        )
        need_target = (
            self.offline_acceptance_objective != "none"
        ) or (self.l1_loss_alpha > 0) or (self.bv_loss_alpha > 0) or self.vat_enabled or (
            self.draft_model.confidence_head is not None
            and self.confidence_head_alpha > 0
        ) or self.pace_mode in {"overlap", "hybrid"} or paired_projective or pair_active
        if need_target:
            if last_hidden_states is None:
                raise ValueError(
                    "DSpark L1/BV/confidence losses require target last_hidden_states; "
                    "ensure the target model surfaces its final hidden state."
                )
            # target distribution for the token at label_indices = target LM head
            # applied to the target hidden one position earlier (anchor+j).
            tgt_idx = (safe_label_indices - 1).clamp(min=0)  # [B, nb, bs]
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            aligned_hidden = torch.gather(last_hidden_states, 1, gather_idx)
            aligned_target_logits = F.linear(aligned_hidden, self.lm_head.weight).view(
                bsz, n_blocks, self.block_size, vocab_size
            )
            if self.bv_loss_alpha > 0:
                # Corrected CARH proposal, same target prefix/position. The BV
                # path returns T=1 overlap for unchanged confidence supervision.
                bv_terms = bv_loss_terms(
                    logits_4d, aligned_target_logits, target_ids, eval_bool,
                    beta=current_bv_beta, temperature=self.bv_temperature,
                    block_chunk_size=self.bv_block_chunk_size,
                )
                accept_rate = bv_terms.overlap.clamp(0.0, 1.0)
                l1_per_token = 2.0 * (1.0 - accept_rate)
            else:
                draft_logits_fp32 = logits_4d.float()
                target_logits_fp32 = aligned_target_logits.float()
                draft_probs = torch.softmax(draft_logits_fp32, dim=-1)
                target_probs = torch.softmax(target_logits_fp32, dim=-1)
                l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)  # [B, nb, bs]
                if self.l1_loss_alpha > 0:
                    l1_num = (l1_per_token * decay_weight_mask).sum()
                accept_rate = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
            if pair_active:
                from specforge.modeling.draft.carh_diagnostics import predecessor_pair_metrics

                # Bound the extra vocabulary projection to four anchors per
                # sequence. The counterfactual branch never changes training loss.
                with torch.no_grad():
                    original_gate_mean = self.draft_model.markov_head._last_gate_mean
                    pair_bias, _ = self.draft_model.markov_head.compute_block_bias_and_latent(
                        token_ids=prev_token_ids[:, :4],
                        hidden_states=hidden_4d[:, :4].detach(),
                        use_second_predecessor=False,
                        use_third_predecessor=False,
                    )
                    self.draft_model.markov_head._last_gate_mean = original_gate_mean
                    carh_pair_diagnostics = predecessor_pair_metrics(
                        base_logits_4d[:, :4].detach() + pair_bias,
                        logits_4d[:, :4].detach(),
                        aligned_target_logits[:, :4].detach(),
                        eval_mask[:, :4],
                    )

            if self.offline_acceptance_objective == "angel-lk-e2e":
                # This path is entirely teacher-forced: it uses the frozen
                # target distribution at the same offline sequence positions
                # and never samples a drafter rollout.
                temperature = self.offline_acceptance_temperature
                if temperature == 1.0:
                    offline_draft_probs = draft_probs
                    offline_target_probs = target_probs.detach()
                    offline_draft_logits = draft_logits_fp32
                    offline_target_logits = target_logits_fp32.detach()
                    # Reuse the exact full-vocabulary overlap already derived
                    # from L1, avoiding another vocabulary-sized minimum.
                    overlap = accept_rate
                else:
                    offline_draft_logits = draft_logits_fp32 / temperature
                    offline_target_logits = (
                        target_logits_fp32.detach() / temperature
                    )
                    offline_draft_probs = torch.softmax(
                        offline_draft_logits, dim=-1
                    )
                    offline_target_probs = torch.softmax(
                        offline_target_logits, dim=-1
                    )
                    overlap = torch.minimum(
                        offline_draft_probs, offline_target_probs
                    ).sum(dim=-1).clamp(0.0, 1.0)
                tv_per_token = 1.0 - overlap

                # Forward KL without materializing two additional full-vocab
                # log-softmax tensors.  The target entropy is detached because
                # the target model and LM head are frozen teachers.
                target_entropy = (
                    torch.logsumexp(offline_target_logits, dim=-1)
                    - (
                        offline_target_probs * offline_target_logits
                    ).sum(dim=-1)
                ).detach()
                target_cross_entropy = (
                    torch.logsumexp(offline_draft_logits, dim=-1)
                    - (
                        offline_target_probs * offline_draft_logits
                    ).sum(dim=-1)
                )
                kl_per_token = (target_cross_entropy - target_entropy).clamp_min(0.0)
                lk_lambda = torch.exp(
                    -self.offline_acceptance_lk_eta * overlap.detach()
                )
                lk_per_token = (
                    lk_lambda * kl_per_token
                    + (1.0 - lk_lambda) * tv_per_token
                )

                offline_valid = eval_mask.bool()
                target_token_probability = offline_draft_probs.gather(
                    -1, target_ids.unsqueeze(-1)
                ).squeeze(-1)
                smoothed_confidence = (
                    self.offline_acceptance_dpace_rho
                    + (1.0 - self.offline_acceptance_dpace_rho)
                    * target_token_probability.detach()
                )
                smoothed_confidence = torch.where(
                    offline_valid,
                    smoothed_confidence,
                    torch.ones_like(smoothed_confidence),
                )
                prefix_confidence = torch.cumprod(
                    smoothed_confidence, dim=-1
                )
                dpace_weights = torch.flip(
                    torch.cumsum(
                        torch.flip(
                            prefix_confidence * eval_mask, dims=[-1]
                        ),
                        dim=-1,
                    ),
                    dims=[-1],
                ).detach()
                dpace_weights = dpace_weights * eval_mask
                offline_lk_num = (lk_per_token * dpace_weights).sum()
                offline_lk_den = dpace_weights.sum()

                # Prefix products are the probability that a verifier reaches
                # each depth.  Invalid tail positions are multiplicative
                # identities and are removed from the final mean.
                valid_prefix = torch.cumprod(
                    offline_valid.to(torch.int32), dim=-1
                ).to(dtype=overlap.dtype)
                overlap_or_identity = torch.where(
                    offline_valid, overlap, torch.ones_like(overlap)
                )
                prefix_acceptance = (
                    torch.cumprod(overlap_or_identity, dim=-1) * valid_prefix
                )
                offline_e2e_num = (
                    (1.0 - prefix_acceptance) * valid_prefix
                ).sum()
                offline_e2e_den = valid_prefix.sum()

                metric_den = eval_mask.sum().clamp_min(1.0)
                block_den = offline_valid[..., 0].sum().clamp_min(1)
                offline_acceptance_mean = (
                    overlap.detach() * eval_mask
                ).sum() / metric_den
                offline_expected_length = (
                    prefix_acceptance.detach().sum() / block_den
                )
                offline_lk_lambda_mean = (
                    lk_lambda.detach() * eval_mask
                ).sum() / metric_den
            with torch.no_grad():
                overlap_for_survival = torch.where(
                    eval_bool,
                    accept_rate.detach(),
                    torch.ones_like(accept_rate),
                )
                block_hazard = 1.0 - overlap_for_survival.prod(dim=-1)
                # H(p) = logsumexp(z) - E_p[z].  Reusing the existing target
                # probabilities avoids materializing another full-vocabulary
                # log-softmax tensor solely for anchor sampling.
                if bv_terms is not None:
                    target_entropy = bv_terms.target_entropy / math.log(max(vocab_size, 2))
                else:
                    target_entropy = (
                        torch.logsumexp(target_logits_fp32, dim=-1)
                        - (target_probs * target_logits_fp32).sum(dim=-1)
                    ) / math.log(max(vocab_size, 2))
                block_flatness = (
                    target_entropy * eval_mask
                ).sum(dim=-1) / eval_mask.sum(dim=-1).clamp_min(1.0)

        # ---- Verification-Aware Training (VAT) matched baseline ----
        vat_soft_num = base_logits.new_zeros((), dtype=torch.float32)
        vat_diagnostics = {}
        vat_confidence_target = None
        vat_confidence_weight = None
        if self.vat_enabled:
            (
                vat_confidence_target,
                vat_weight_mask,
                vat_diagnostics,
            ) = self._vat_verification_targets(
                logits_4d.detach(), target_logits_fp32.detach(), eval_mask
            )
            # First-rejection-anchored weights replace the fixed depth decay.
            # They apply identically to hard and soft token supervision.
            ce_weight_mask = vat_weight_mask
            vat_confidence_weight = eval_mask
            vat_soft_per_token = -(
                target_probs.detach()
                * F.log_softmax(logits_4d.float(), dim=-1)
            ).sum(dim=-1)
            vat_soft_num = (vat_soft_per_token * vat_weight_mask).sum()
            local_den = vat_weight_mask.sum()

        # ---- DSpark-PACE continuation credit ----
        pace_diagnostics = {}
        l1_weight_mask = decay_weight_mask
        rollout_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        if self.pace_mode == "rollout-residual":
            rollout_quality = self._rollout_quality(
                base_logits_4d, hidden_4d, anchor_token_ids, target_ids
            )
            residual_weights, pace_diagnostics = self._rollout_residual_weights(
                rollout_quality, eval_mask, decay_weight_mask
            )
            pace_blend = self._effective_pace_blend()
            rollout_aux_num = pace_blend * (
                ce_per_token * residual_weights
            ).sum()
            pace_diagnostics["pace_effective_blend"] = torch.tensor(
                pace_blend, device=device, dtype=torch.float32
            )
            pace_diagnostics["pace_rollout_ce"] = (
                (ce_per_token * decay_weight_mask).sum()
                / local_den.clamp_min(1e-6)
            ).detach()
        elif self.pace_mode != "none":
            token_quality = torch.exp(-ce_per_token.detach().float())
            if self.pace_mode == "token":
                pace_quality = token_quality
            elif self.pace_mode == "overlap":
                pace_quality = accept_rate.detach()
            else:
                # Geometric interpolation is additive in log-survival space.
                beta = self.pace_hybrid_beta
                eps = torch.finfo(torch.float32).tiny
                pace_quality = torch.exp(
                    (1.0 - beta) * torch.log(token_quality.clamp_min(eps))
                    + beta * torch.log(accept_rate.detach().float().clamp_min(eps))
                )
            pace_weight_mask, pace_diagnostics = self._pace_weights(
                pace_quality, eval_mask, decay_weight_mask
            )
            pace_blend = self._effective_pace_blend()
            blended_weight_mask = (
                (1.0 - pace_blend) * decay_weight_mask
                + pace_blend * pace_weight_mask
            )
            ce_weight_mask = blended_weight_mask
            if self.pace_apply_to == "ce-l1":
                l1_weight_mask = blended_weight_mask
            pace_diagnostics["pace_effective_blend"] = torch.tensor(
                pace_blend, device=device, dtype=torch.float32
            )

        ce_num = (ce_per_token * ce_weight_mask).sum()
        if self.l1_loss_alpha > 0:
            l1_num = (l1_per_token * l1_weight_mask).sum()

        # ---- Elastic K=10 teacher -> K=7 projective consistency ----
        # Both views share weights, noise, anchors, labels, and target states;
        # only the intra-block visibility horizon differs. The long view is a
        # detached teacher and is trusted only where verifier overlap or the
        # verifier-token margin is measurably better than the short view.
        elastic_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        elastic_diagnostics = {
            "elastic_active_horizon": base_logits.new_tensor(
                float(active_block_size)
            ),
            "elastic_paired_step": base_logits.new_tensor(
                float(paired_projective)
            ),
        }
        if paired_projective:
            pair_block_count = min(
                n_blocks, self.elastic_projective_num_anchors
            )
            paired_anchors = anchor_positions[:, :pair_block_count]
            paired_keep = block_keep_mask[:, :pair_block_count]
            paired_noise = noise_embedding.view(
                bsz, n_blocks, self.block_size, -1
            )[:, :pair_block_count].reshape(
                bsz, pair_block_count * self.block_size, -1
            )
            paired_draft_positions = draft_position_ids.view(
                bsz, n_blocks, self.block_size
            )[:, :pair_block_count].reshape(
                bsz, pair_block_count * self.block_size
            )
            paired_position_ids = torch.cat(
                [context_position_ids, paired_draft_positions], dim=1
            )
            paired_prev_token_ids = prev_token_ids[:, :pair_block_count]
            alternate_horizon = (
                self.elastic_long_horizon
                if active_block_size == self.elastic_short_horizon
                else self.elastic_short_horizon
            )
            if self.attention_backend == "flex_attention":
                alternate_mask = create_dflash_block_mask(
                    anchor_positions=paired_anchors,
                    block_keep_mask=paired_keep,
                    S=seq_len,
                    block_size=self.block_size,
                    device=device,
                    active_block_size=alternate_horizon,
                )
            else:
                alternate_mask = create_dflash_sdpa_mask(
                    anchor_positions=paired_anchors,
                    block_keep_mask=paired_keep,
                    S=seq_len,
                    block_size=self.block_size,
                    device=device,
                    active_block_size=alternate_horizon,
                )
            alternate_hidden = self.draft_model(
                position_ids=paired_position_ids,
                noise_embedding=paired_noise,
                target_hidden=hidden_states,
                attention_mask=alternate_mask,
            ).view(bsz, pair_block_count, self.block_size, -1)
            alternate_hidden = self.draft_model.apply_prefix_state_mixer(
                alternate_hidden
            )
            alternate_hidden = self.draft_model.apply_block_summary(alternate_hidden)
            if self.draft_model.recall_correction is not None:
                alternate_depth_ids = torch.arange(
                    self.block_size, device=device, dtype=torch.long
                ).view(1, 1, -1).expand_as(paired_prev_token_ids)
                alternate_hidden = self.draft_model.apply_recall_correction(
                    alternate_hidden,
                    paired_prev_token_ids,
                    alternate_depth_ids,
                )
            alternate_logits = self.lm_head(
                alternate_hidden.reshape(-1, alternate_hidden.size(-1))
            ).view(bsz, pair_block_count, self.block_size, vocab_size)
            if self.draft_model.markov_head is not None:
                alternate_logits = (
                    self.draft_model.markov_head.apply_block_logits(
                        alternate_logits,
                        token_ids=paired_prev_token_ids,
                        hidden_states=alternate_hidden,
                    )
                )

            if active_block_size == self.elastic_short_horizon:
                short_logits = logits_4d[
                    :, :pair_block_count, : self.elastic_short_horizon
                ]
                long_logits = alternate_logits[
                    :, :, : self.elastic_short_horizon
                ]
            else:
                short_logits = alternate_logits[
                    :, :, : self.elastic_short_horizon
                ]
                long_logits = logits_4d[:, :, : self.elastic_short_horizon]
                long_logits = long_logits[:, :pair_block_count]

            verifier_logits = aligned_target_logits[
                :, :pair_block_count, : self.elastic_short_horizon
            ].detach().float()
            candidate_ids = verifier_logits.topk(
                min(self.elastic_projective_top_k, vocab_size), dim=-1
            ).indices
            verifier_candidates = verifier_logits.gather(-1, candidate_ids)
            short_candidates = short_logits.float().gather(-1, candidate_ids)
            long_candidates = long_logits.float().gather(-1, candidate_ids)
            verifier_probs = torch.softmax(verifier_candidates, dim=-1)
            short_probs = torch.softmax(short_candidates, dim=-1)
            long_probs = torch.softmax(long_candidates, dim=-1)
            short_overlap = torch.minimum(
                verifier_probs, short_probs
            ).sum(dim=-1)
            long_overlap = torch.minimum(
                verifier_probs, long_probs
            ).sum(dim=-1)

            # Greedy verifier token for the margin gate. Distributional
            # overlap above remains the temperature-robust gate.
            verifier_ids = verifier_logits.argmax(dim=-1)

            def target_margin(values, ids):
                target_value = values.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
                top2_values, top2_ids = torch.topk(values, k=2, dim=-1)
                best_other = torch.where(
                    top2_ids[..., 0].eq(ids),
                    top2_values[..., 1],
                    top2_values[..., 0],
                )
                return target_value - best_other

            short_margin = target_margin(short_logits.float(), verifier_ids)
            long_margin = target_margin(long_logits.float(), verifier_ids)
            elastic_gate_temperature = self._elastic_gate_temperature()
            if elastic_gate_temperature == 0.0:
                beneficial = long_margin.gt(
                    short_margin + self.elastic_margin_gap
                ).detach()
            else:
                beneficial = long_overlap.gt(
                    short_overlap + self.elastic_overlap_gap
                ).detach()
            short_valid = eval_bool[
                :, :pair_block_count, : self.elastic_short_horizon
            ]
            survival_weight = torch.cumprod(
                torch.where(
                    short_valid,
                    short_overlap.detach().clamp_min(1e-4),
                    torch.ones_like(short_overlap),
                ),
                dim=-1,
            )
            teacher = long_probs.detach().clamp_min(1e-8)
            student_log = torch.log_softmax(short_candidates, dim=-1)
            projective_kl = (
                teacher * (teacher.log() - student_log)
            ).sum(dim=-1)
            projective_weight = (
                short_valid.float()
                * beneficial.float()
                * survival_weight
                * decay_weight_mask[
                    :, :pair_block_count, : self.elastic_short_horizon
                ]
            )
            raw_projective = self.elastic_projective_alpha * (
                projective_kl * projective_weight
            ).sum()
            elastic_aux_num, elastic_budget_scale = (
                self._budget_auxiliary_loss(
                    raw_projective, self.elastic_loss_budget, ce_num
                )
            )
            valid_count = short_valid.sum().clamp_min(1)
            elastic_diagnostics.update(
                {
                    "elastic_projective_gate_rate": (
                        (beneficial & short_valid).sum().float() / valid_count
                    ).detach(),
                    "elastic_long_overlap_gain": (
                        ((long_overlap - short_overlap) * short_valid.float()).sum()
                        / valid_count
                    ).detach(),
                    "elastic_long_margin_gain": (
                        ((long_margin - short_margin) * short_valid.float()).sum()
                        / valid_count
                    ).detach(),
                    "elastic_projective_budget_scale": (
                        elastic_budget_scale.detach()
                    ),
                    "elastic_projective_anchor_count": (
                        base_logits.new_tensor(float(pair_block_count))
                    ),
                    "elastic_gate_temperature": base_logits.new_tensor(
                        elastic_gate_temperature
                    ),
                }
            )

        # ---- Recall expert: representation repair outside unary Top-k ----
        recall_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        recall_diagnostics = {}
        if recall_hidden_delta is not None and self.recall_correction_alpha > 0:
            miss_flat = recall_miss_mask.reshape(-1)
            miss_indices = miss_flat.nonzero(as_tuple=False).squeeze(-1)
            miss_weight = recall_miss_mask.float() * decay_weight_mask
            if miss_indices.numel() > 0:
                delta_hidden = recall_hidden_delta.reshape(
                    -1, recall_hidden_delta.size(-1)
                ).index_select(0, miss_indices)
                recall_aux_logits = self.lm_head(delta_hidden)
                recall_aux_logits.add_(
                    unary_logits_4d.detach()
                    .reshape(-1, vocab_size)
                    .index_select(0, miss_indices)
                )
                miss_targets = flat_targets.index_select(0, miss_indices)
                recall_ce = F.cross_entropy(
                    recall_aux_logits,
                    miss_targets,
                    reduction="none",
                )
                recall_ce_num = (
                    recall_ce
                    * miss_weight.reshape(-1).index_select(0, miss_indices)
                ).sum()
                corrected_on_miss = recall_aux_logits.detach().argmax(dim=-1)
                corrected_miss_count = corrected_on_miss.eq(miss_targets).sum()
            else:
                recall_ce_num = recall_hidden_delta.new_zeros((), dtype=torch.float32)
                corrected_miss_count = torch.zeros((), device=device, dtype=torch.long)
            preserve_weight = (
                (~recall_miss_mask & eval_bool).float() * decay_weight_mask
            )
            # Recall correction is a hidden-space residual.  Penalizing that
            # residual directly preserves the no-op path without materializing
            # an FP32 [B, blocks, depth, vocab] tensor (6+ GiB for Qwen3-8B).
            residual_energy = recall_hidden_delta.float().square().mean(dim=-1)
            raw_recall = self.recall_correction_alpha * (
                recall_ce_num
                + 0.05 * (residual_energy * preserve_weight).sum()
            )
            recall_aux_num, recall_budget_scale = self._budget_auxiliary_loss(
                raw_recall, self.recall_correction_loss_budget, ce_num
            )
            miss_count = recall_miss_mask.sum().clamp_min(1)
            recall_diagnostics = {
                "recall_miss_rate": (
                    recall_miss_mask.sum().float()
                    / eval_bool.sum().clamp_min(1)
                ).detach(),
                "recall_to_top1_rate": (
                    corrected_miss_count.float() / miss_count
                ).detach(),
                "recall_correction_budget_scale": recall_budget_scale.detach(),
                "recall_correction_gate_mean": (
                    self.draft_model.recall_correction._last_gate_mean
                    if self.draft_model.recall_correction._last_gate_mean is not None
                    else raw_recall.new_zeros(())
                ),
            }

        # ---- Coverage-aware CARH correction ----
        # The unary backbone receives the normal CE/L1 objective on every
        # token. CARH receives a strict Top-k ranking loss only when the target
        # is covered but not already Top-1, a preservation margin on already
        # correct states, and a small no-op penalty on true recall misses.
        carh_recoverability_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        carh_recoverability_diagnostics = {}
        if recoverability_enabled:
            assert carh_candidate_ids is not None
            assert carh_correct_mask is not None
            assert carh_recoverable_mask is not None
            assert carh_recall_miss_mask is not None
            assert carh_residual_bias is not None
            assert carh_correction_latent is not None
            aux_weight = ce_weight_mask.detach()

            candidate_logits = (
                base_logits_4d.detach().gather(-1, carh_candidate_ids)
                + carh_residual_bias.gather(-1, carh_candidate_ids)
            ).float()
            candidate_matches = carh_candidate_ids.eq(target_ids.unsqueeze(-1))
            candidate_target_index = candidate_matches.long().argmax(dim=-1)
            recovery_ce = F.cross_entropy(
                candidate_logits.reshape(
                    -1, self.carh_recoverability_top_k
                ),
                candidate_target_index.reshape(-1),
                reduction="none",
            ).view_as(target_ids)
            recovery_weight = (
                aux_weight * carh_recoverable_mask.float()
            )
            recovery_raw = (recovery_ce * recovery_weight).sum()

            # Determine the exact best competing token from serving logits,
            # then recompute only gold/competitor values with CARH gradients.
            # This avoids casting the full vocabulary tensor to FP32.
            with torch.no_grad():
                top2_ids = torch.topk(logits_4d.detach(), k=2, dim=-1).indices
                best_other_ids = torch.where(
                    top2_ids[..., 0].eq(target_ids),
                    top2_ids[..., 1],
                    top2_ids[..., 0],
                )
            preservation_ids = torch.stack(
                [target_ids, best_other_ids], dim=-1
            )
            preservation_logits = (
                base_logits_4d.detach().gather(-1, preservation_ids)
                + carh_residual_bias.gather(-1, preservation_ids)
            ).float()
            corrected_margin = (
                preservation_logits[..., 0] - preservation_logits[..., 1]
            )
            preservation_weight = aux_weight * carh_correct_mask.float()
            preservation_raw = (
                F.softplus(
                    (
                        self.carh_preservation_margin - corrected_margin
                    )
                    / self.carh_preservation_temperature
                )
                * preservation_weight
            ).sum()

            miss_weight = aux_weight * carh_recall_miss_mask.float()
            residual_energy = (
                carh_correction_latent.float().square().mean(dim=-1)
            )
            noop_raw = (residual_energy * miss_weight).sum()
            weighted_recoverability = (
                self.carh_recovery_loss_alpha * recovery_raw
                + self.carh_preservation_loss_alpha * preservation_raw
                + self.carh_noop_loss_alpha * noop_raw
            )
            (
                carh_recoverability_aux_num,
                recoverability_budget_scale,
            ) = self._budget_auxiliary_loss(
                weighted_recoverability,
                self.carh_recoverability_loss_budget,
                ce_num,
            )

            with torch.no_grad():
                valid_count = eval_mask.sum().clamp_min(1.0)
                corrected_ids = logits_4d.argmax(dim=-1)
                recovery_count = carh_recoverable_mask.sum().clamp_min(1)
                correct_count = carh_correct_mask.sum().clamp_min(1)
                recovered = (
                    carh_recoverable_mask
                    & corrected_ids.eq(target_ids)
                )
                overcorrected = (
                    carh_correct_mask
                    & corrected_ids.ne(target_ids)
                )
                carh_recoverability_diagnostics = {
                    "carh_topk_coverage": (
                        (carh_correct_mask | carh_recoverable_mask).sum().float()
                        / valid_count
                    ),
                    "carh_recoverable_rate": (
                        carh_recoverable_mask.sum().float() / valid_count
                    ),
                    "carh_recall_miss_rate": (
                        carh_recall_miss_mask.sum().float() / valid_count
                    ),
                    "carh_recovery_accuracy": (
                        recovered.sum().float() / recovery_count
                    ),
                    "carh_overcorrection_rate": (
                        overcorrected.sum().float() / correct_count
                    ),
                    "carh_net_recovery_rate": (
                        (recovered.sum() - overcorrected.sum()).float()
                        / valid_count
                    ),
                    "carh_recoverability_budget_scale": (
                        recoverability_budget_scale.detach()
                    ),
                }
                for depth in range(self.block_size):
                    depth_valid = eval_mask[..., depth].sum().clamp_min(1.0)
                    depth_recoverable = carh_recoverable_mask[..., depth]
                    depth_correct = carh_correct_mask[..., depth]
                    carh_recoverability_diagnostics.update(
                        {
                            f"carh_topk_coverage_depth_{depth + 1}": (
                                (
                                    depth_correct | depth_recoverable
                                ).sum().float()
                                / depth_valid
                            ),
                            f"carh_recovery_accuracy_depth_{depth + 1}": (
                                (
                                    depth_recoverable
                                    & corrected_ids[..., depth].eq(
                                        target_ids[..., depth]
                                    )
                                ).sum().float()
                                / depth_recoverable.sum().clamp_min(1)
                            ),
                            f"carh_overcorrection_rate_depth_{depth + 1}": (
                                (
                                    depth_correct
                                    & corrected_ids[..., depth].ne(
                                        target_ids[..., depth]
                                    )
                                ).sum().float()
                                / depth_correct.sum().clamp_min(1)
                            ),
                        }
                    )

        dialogue_credit_features = None
        dialogue_credit_active = block_keep_mask & eval_mask[..., 0].bool()
        if (
            self.dialogue_prefix_phase_weighting
            or self.dialogue_t2cm_boundary_weighting
            or self.dialogue_dsg_adaptive_margin
        ):
            dialogue_credit_features = self._dialogue_anchor_features(
                dialogue_credit_active,
                anchor_positions,
                loss_mask,
                input_ids,
            )

        self._last_on_policy_confidence = None
        on_policy_aux_num, on_policy_diagnostics = (
            self._on_policy_survival_objective(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                eval_mask=eval_mask,
                base_logits=base_logits_4d,
                hidden_states=hidden_4d,
                ce_num=ce_num,
                sample_ids=sample_ids,
                source_ids=source_ids,
                block_hazard=block_hazard,
                block_flatness=block_flatness,
                active_block_size=active_block_size,
            )
        )

        # ---- Inference-aligned first-rejection credit ----
        prefix_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        prefix_diagnostics = {}
        effective_prefix_alpha = self._effective_prefix_credit_alpha()
        need_acceptance_state = (
            self.prefix_credit_mode != "none"
            or self.shallow_frc_alpha > 0
            or self.dfap_alpha > 0
            or self.transition_credit_alpha > 0
            or self.conv_gate_loss_alpha > 0
            or self.transition2_margin_alpha > 0
            or self.prefix_bottleneck_alpha > 0
            or self.deep_survival_guard_alpha > 0
        )
        first_rejection_mask = None
        if need_acceptance_state:
            first_rejection_mask, prefix_diagnostics = self._first_rejection_mask(
                base_logits_4d,
                hidden_4d,
                anchor_token_ids,
                target_ids,
                eval_mask,
            )
            if dialogue_credit_features is not None:
                dialogue_survival = (
                    first_rejection_mask.cumsum(dim=-1).eq(0)
                    * eval_mask
                ).sum(dim=-1).float()
                dialogue_rejected = first_rejection_mask.any(dim=-1)
                dialogue_boundary = (
                    dialogue_credit_features["depth_boundary"]
                    | dialogue_credit_features["depth_script_transition"]
                )
                valid_dialogue_blocks = dialogue_credit_active.sum().clamp_min(1)
                prefix_diagnostics[
                    "dialogue_teacher_forced_boundary_rate"
                ] = (
                    (dialogue_boundary * eval_mask.bool()).sum().float()
                    / eval_mask.sum().clamp_min(1)
                ).detach()
                for phase_index in range(5):
                    phase_blocks = dialogue_credit_active & dialogue_credit_features[
                        "phase"
                    ].eq(phase_index)
                    phase_count = phase_blocks.sum().clamp_min(1)
                    prefix_diagnostics.update(
                        {
                            f"dialogue_tf_phase_{phase_index + 1}_block_rate": (
                                phase_blocks.sum().float() / valid_dialogue_blocks
                            ).detach(),
                            f"dialogue_tf_phase_{phase_index + 1}_mean_survival": (
                                (dialogue_survival * phase_blocks).sum() / phase_count
                            ).detach(),
                            f"dialogue_tf_phase_{phase_index + 1}_rejection_rate": (
                                (dialogue_rejected & phase_blocks).sum().float()
                                / phase_count
                            ).detach(),
                        }
                    )
                if source_ids is not None:
                    for source_index in range(self.dialogue_num_sources):
                        source_blocks = dialogue_credit_active & source_ids.eq(
                            source_index
                        ).unsqueeze(-1)
                        source_count = source_blocks.sum().clamp_min(1)
                        prefix_diagnostics[
                            f"dialogue_tf_source_{source_index}_mean_survival"
                        ] = (
                            (dialogue_survival * source_blocks).sum() / source_count
                        ).detach()
            if effective_prefix_alpha > 0:
                if self.prefix_credit_mode == "full":
                    prefix_ce = ce_per_token
                else:
                    # Recompute the residual path so its parameters always get
                    # the full auxiliary gradient. residual-only uses scale=0;
                    # partial admits a controlled gradient into the backbone.
                    backbone_grad_scale = (
                        0.0
                        if self.prefix_credit_mode == "residual-only"
                        else self.prefix_credit_backbone_grad_scale
                    )
                    residual_bias = self.draft_model.markov_head.compute_block_bias(
                        token_ids=prev_token_ids,
                        hidden_states=self._scale_gradient(
                            hidden_4d, backbone_grad_scale
                        ),
                    )
                    residual_logits = self._scale_gradient(
                        base_logits_4d, backbone_grad_scale
                    ) + residual_bias
                    prefix_ce = F.cross_entropy(
                        residual_logits.reshape(-1, vocab_size),
                        flat_targets,
                        reduction="none",
                    ).view(bsz, n_blocks, self.block_size)
                prefix_mask = first_rejection_mask
                if (
                    self.state_credit_partition == "strict"
                    and self.shallow_frc_alpha > 0
                ):
                    # Strict state partition: shallow frontiers are owned by
                    # FRC, so Prefix-Full starts after the FRC repair window.
                    prefix_depth = torch.arange(
                        self.block_size, device=device
                    ).ge(self.shallow_frc_max_depth).view(1, 1, -1)
                    prefix_mask = prefix_mask * prefix_depth
                if self.dialogue_prefix_phase_weighting:
                    prefix_block_active = prefix_mask.bool().any(dim=-1)
                    prefix_multiplier = self._dialogue_credit_multiplier(
                        prefix_block_active,
                        dialogue_credit_features,
                        use_phase=True,
                        use_boundary=False,
                        reference_weight=prefix_mask.sum(dim=-1),
                    )
                    prefix_mask = prefix_mask * prefix_multiplier.unsqueeze(-1)
                    for phase_index in range(5):
                        phase_mask = dialogue_credit_features["phase"].eq(
                            phase_index
                        )
                        prefix_diagnostics[
                            f"dialogue_prefix_phase_{phase_index + 1}_credit_mass"
                        ] = (
                            prefix_mask * phase_mask.unsqueeze(-1)
                        ).sum().detach()
                prefix_aux_num = effective_prefix_alpha * (
                    prefix_ce * prefix_mask
                ).sum()
                prefix_diagnostics["prefix_optimized_frontier_count"] = (
                    prefix_mask.sum().detach()
                )
            prefix_diagnostics["prefix_effective_alpha"] = torch.tensor(
                effective_prefix_alpha, device=device, dtype=torch.float32
            )
            prefix_diagnostics["prefix_backbone_grad_scale"] = torch.tensor(
                1.0
                if self.prefix_credit_mode == "full"
                else (
                    0.0
                    if self.prefix_credit_mode == "residual-only"
                    else self.prefix_credit_backbone_grad_scale
                ),
                device=device,
                dtype=torch.float32,
            )

        # ---- State-partitioned acceptance credit ----
        # Prefix-Full repairs partial rollouts with CE. Shallow FRC adds only a
        # boundary-margin residual at the first rejection (depths 1..D), while
        # DFAP acts only on already full-accepted chains at deep depths. The
        # selectors are detached, so the three objectives cannot game their
        # assigned acceptance state.
        frc_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        dfap_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        state_scale = self._effective_state_credit_scale()
        if state_scale > 0 and (self.shallow_frc_alpha > 0 or self.dfap_alpha > 0):
            logits_fp32 = logits_4d.float()
            target_logits = logits_fp32.gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            top2_values, top2_ids = torch.topk(logits_fp32, k=2, dim=-1)
            best_other = torch.where(
                top2_ids[..., 0].eq(target_ids),
                top2_values[..., 1],
                top2_values[..., 0],
            )
            target_margin = target_logits - best_other

            if self.shallow_frc_alpha > 0:
                shallow = torch.arange(
                    self.block_size, device=device
                ).lt(self.shallow_frc_max_depth).view(1, 1, -1)
                suffix_capacity = torch.flip(
                    torch.cumsum(torch.flip(eval_mask, dims=[-1]), dim=-1),
                    dims=[-1],
                )
                frc_weight = (
                    first_rejection_mask * shallow * suffix_capacity
                ).detach()
                frc_aux_num = (
                    state_scale
                    * self.shallow_frc_alpha
                    * (
                        F.softplus(
                            (self.shallow_frc_margin - target_margin)
                            / self.shallow_frc_temperature
                        )
                        * frc_weight
                    ).sum()
                )
                prefix_diagnostics["shallow_frc_count"] = (
                    (first_rejection_mask * shallow).sum().detach()
                )
                prefix_diagnostics["shallow_frc_credit_mass"] = (
                    frc_weight.sum().detach()
                )

            if self.dfap_alpha > 0:
                valid_depth = eval_mask.sum(dim=-1)
                full_accept = (
                    eval_mask[..., 0].bool()
                    & ~first_rejection_mask.bool().any(dim=-1)
                    & valid_depth.ge(self.dfap_min_depth)
                ).detach()
                deep = torch.arange(self.block_size, device=device).ge(
                    self.dfap_min_depth - 1
                ).view(1, 1, -1)
                protected = (
                    eval_mask.bool() & deep & full_accept.unsqueeze(-1)
                ).detach()
                protected_weight = ce_weight_mask.detach() * protected.float()
                dfap_aux_num = (
                    state_scale
                    * self.dfap_alpha
                    * (
                        F.softplus(
                            (self.dfap_margin - target_margin)
                            / self.dfap_temperature
                        )
                        * protected_weight
                    ).sum()
                )
                prefix_diagnostics["dfap_protected_blocks"] = (
                    full_accept.float().sum().detach()
                )
                prefix_diagnostics["dfap_protected_tokens"] = (
                    protected.float().sum().detach()
                )
        prefix_diagnostics["state_credit_scale"] = torch.tensor(
            state_scale, device=device, dtype=torch.float32
        )
        prefix_diagnostics["state_credit_strict_partition"] = torch.tensor(
            float(self.state_credit_partition == "strict"),
            device=device,
            dtype=torch.float32,
        )

        # ---- Transition-aligned prefix credit (TAPC) ----
        # Optimize depths 2..D only when the rollout prefix immediately before
        # that depth is still alive.  The predecessor is the model's detached
        # greedy rollout token, matching CARH serving rather than teacher forcing.
        transition_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        gate_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        transition2_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        bottleneck_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        deep_guard_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        if state_scale > 0 and first_rejection_mask is not None:
            rejected_before = torch.cat(
                [
                    torch.zeros_like(first_rejection_mask[..., :1]),
                    first_rejection_mask[..., :-1].cumsum(dim=-1),
                ],
                dim=-1,
            )
            alive_before = rejected_before.eq(0).float() * eval_mask

            # Compute inference-rollout margins once for all transition-aware
            # objectives.  Each step consumes the detached greedy predecessor,
            # matching CARH serving rather than teacher-forced block logits.
            need_rollout_margin = (
                self.transition_credit_alpha > 0
                or self.transition2_margin_alpha > 0
                or self.prefix_bottleneck_alpha > 0
                or self.deep_survival_guard_alpha > 0
            )
            rollout_target_margin = None
            if need_rollout_margin:
                rollout_depth_count = 0
                if self.transition_credit_alpha > 0:
                    rollout_depth_count = max(
                        rollout_depth_count, self.transition_credit_max_depth
                    )
                if self.transition2_margin_alpha > 0:
                    rollout_depth_count = max(rollout_depth_count, 2)
                if self.prefix_bottleneck_alpha > 0:
                    rollout_depth_count = max(
                        rollout_depth_count, self.prefix_bottleneck_max_depth
                    )
                if self.deep_survival_guard_alpha > 0:
                    rollout_depth_count = self.block_size
                predecessor = anchor_token_ids
                markov_head = self.draft_model.markov_head
                memory_enabled = (
                    getattr(markov_head, "sampled_prefix_memory", None) is not None
                )
                memory_state = (
                    markov_head.init_sampled_prefix_state(predecessor)
                    if memory_enabled else None
                )
                sampled_steps = []
                rollout_margins = []
                for depth in range(rollout_depth_count):
                    if memory_enabled and depth >= 2:
                        memory_state = markov_head.advance_sampled_prefix_state(
                            memory_state, sampled_steps[depth - 2]
                        )
                    step_bias = markov_head.compute_step_bias(
                        predecessor,
                        hidden_states=(
                            hidden_4d[..., depth, :].detach()
                            if recoverability_enabled
                            else hidden_4d[..., depth, :]
                        ),
                        depth_idx=depth,
                        **_carh_predecessor_context_kwargs(
                            markov_head,
                            hidden_4d.detach() if recoverability_enabled else hidden_4d,
                            depth,
                        ),
                        **({"sampled_prefix_state": memory_state} if memory_enabled else {}),
                    )
                    if recoverability_enabled:
                        route = carh_recoverable_mask[..., depth].unsqueeze(-1)
                        step_bias = torch.where(
                            route, step_bias, step_bias.detach()
                        )
                    step_logits = base_logits_4d[..., depth, :] + step_bias
                    step_fp32 = step_logits.float()
                    step_target = target_ids[..., depth]
                    gold = step_fp32.gather(
                        -1, step_target.unsqueeze(-1)
                    ).squeeze(-1)
                    top2_values, top2_ids = torch.topk(
                        step_fp32, k=2, dim=-1
                    )
                    best_other = torch.where(
                        top2_ids[..., 0].eq(step_target),
                        top2_values[..., 1],
                        top2_values[..., 0],
                    )
                    rollout_margins.append(gold - best_other)
                    predecessor = step_logits.detach().argmax(dim=-1)
                    sampled_steps.append(predecessor)
                rollout_target_margin = torch.stack(rollout_margins, dim=-1)

            if self.transition_credit_alpha > 0:
                transition_raw = base_logits.new_zeros((), dtype=torch.float32)
                transition_count = base_logits.new_zeros((), dtype=torch.float32)
                for depth in range(1, self.transition_credit_max_depth):
                    if 1 <= depth < self.transition_credit_max_depth:
                        margin = rollout_target_margin[..., depth]
                        weight = alive_before[..., depth].detach()
                        transition_raw = transition_raw + (
                            F.softplus(
                                (self.transition_credit_margin - margin)
                                / self.transition_credit_temperature
                            )
                            * weight
                        ).sum()
                        transition_count = transition_count + weight.sum()

                weighted_transition = (
                    state_scale * self.transition_credit_alpha * transition_raw
                )
                if self.transition_credit_loss_budget > 0:
                    allowed = (
                        self.transition_credit_loss_budget
                        * self.ce_loss_alpha
                        * ce_num.detach()
                    )
                    budget_scale = torch.clamp(
                        allowed / weighted_transition.detach().clamp_min(1e-6),
                        max=1.0,
                    )
                    transition_aux_num = weighted_transition * budget_scale
                else:
                    budget_scale = weighted_transition.new_zeros(())
                prefix_diagnostics.update(
                    {
                        "transition_credit_count": transition_count.detach(),
                        "transition_credit_budget_scale": budget_scale.detach(),
                        "transition_credit_effective_alpha": transition_raw.new_tensor(
                            state_scale * self.transition_credit_alpha
                        ),
                    }
                )

            # Transition-2 Conditional Margin (T2CM): repair only the second
            # draft decision, and only for blocks whose first draft token was
            # accepted.  The low-margin selector is detached so the model
            # cannot enter/leave the repair set by manipulating the mask.
            if self.transition2_margin_alpha > 0:
                depth = 1
                margin = rollout_target_margin[..., depth]
                fragile = margin.detach().lt(self.transition2_margin_floor)
                weight = (
                    alive_before[..., depth]
                    * ce_weight_mask[..., depth].detach()
                    * fragile.float()
                ).detach()
                if self.dialogue_t2cm_boundary_weighting:
                    transition_features = {
                        **dialogue_credit_features,
                        "boundary": dialogue_credit_features[
                            "depth_boundary"
                        ][..., depth],
                        "script_transition": dialogue_credit_features[
                            "depth_script_transition"
                        ][..., depth],
                    }
                    transition_multiplier = self._dialogue_credit_multiplier(
                        weight.gt(0),
                        transition_features,
                        use_phase=False,
                        use_boundary=True,
                        reference_weight=weight,
                    )
                    weight = weight * transition_multiplier
                transition2_raw = (
                    F.softplus(
                        (self.transition2_margin_floor - margin)
                        / self.transition2_margin_temperature
                    )
                    * weight
                ).sum()
                weighted_transition2 = (
                    state_scale
                    * self.transition2_margin_alpha
                    * transition2_raw
                )
                transition2_aux_num, budget_scale = self._budget_auxiliary_loss(
                    weighted_transition2,
                    self.transition2_margin_loss_budget,
                    ce_num,
                )
                prefix_diagnostics.update(
                    {
                        "transition2_margin_count": weight.gt(0).sum().detach(),
                        "transition2_margin_mean": (
                            (margin.detach() * weight).sum()
                            / weight.sum().clamp_min(1e-6)
                        ),
                        "transition2_margin_budget_scale": budget_scale.detach(),
                        "dialogue_t2cm_boundary_credit_mass": (
                            weight
                            * transition_features["boundary"].float()
                        ).sum().detach()
                        if self.dialogue_t2cm_boundary_weighting
                        else weight.new_zeros(()),
                        "dialogue_t2cm_script_transition_credit_mass": (
                            weight
                            * transition_features["script_transition"].float()
                        ).sum().detach()
                        if self.dialogue_t2cm_boundary_weighting
                        else weight.new_zeros(()),
                    }
                )

            # Prefix Bottleneck Margin (PBM): optimize a normalized soft-min of
            # the rollout margins over the still-reachable shallow prefix.  A
            # normalized soft-min preserves the value when all margins match,
            # avoiding a depth-count-dependent shift in the margin floor.
            if self.prefix_bottleneck_alpha > 0:
                depth_mask = torch.arange(
                    self.prefix_bottleneck_max_depth, device=device
                ).lt(self.prefix_bottleneck_max_depth).view(1, 1, -1)
                eligible = (
                    alive_before[..., : self.prefix_bottleneck_max_depth].bool()
                    & eval_mask[..., : self.prefix_bottleneck_max_depth].bool()
                    & depth_mask
                )
                softmin_tau = self.prefix_bottleneck_softmin_temperature
                neg_scaled = torch.where(
                    eligible,
                    -rollout_target_margin / softmin_tau,
                    torch.full_like(rollout_target_margin, -torch.inf),
                )
                eligible_count = eligible.sum(dim=-1)
                active = eligible_count.gt(0)
                prefix_margin = -softmin_tau * (
                    torch.logsumexp(neg_scaled, dim=-1)
                    - eligible_count.clamp_min(1).float().log()
                )
                prefix_margin = torch.where(
                    active, prefix_margin, torch.zeros_like(prefix_margin)
                )
                fragile = prefix_margin.detach().lt(
                    self.prefix_bottleneck_margin_floor
                )
                block_weight = (active & fragile).float().detach()
                bottleneck_raw = (
                    F.softplus(
                        (
                            self.prefix_bottleneck_margin_floor
                            - prefix_margin
                        )
                        / self.prefix_bottleneck_loss_temperature
                    )
                    * block_weight
                ).sum()
                weighted_bottleneck = (
                    state_scale
                    * self.prefix_bottleneck_alpha
                    * bottleneck_raw
                )
                bottleneck_aux_num, budget_scale = self._budget_auxiliary_loss(
                    weighted_bottleneck,
                    self.prefix_bottleneck_loss_budget,
                    ce_num,
                )
                prefix_diagnostics.update(
                    {
                        "prefix_bottleneck_block_count": block_weight.sum().detach(),
                        "prefix_bottleneck_margin_mean": (
                            (prefix_margin.detach() * block_weight).sum()
                            / block_weight.sum().clamp_min(1e-6)
                        ),
                        "prefix_bottleneck_budget_scale": budget_scale.detach(),
                    }
                )

            # Deep Survival Guard (DSG): unlike DFAP, a block need not be fully
            # accepted.  Once it has survived min_prefix decisions, protect the
            # already accepted deep prefix tokens whose rollout margin remains
            # below a small safety floor.
            if self.deep_survival_guard_alpha > 0:
                rejected_through = first_rejection_mask.cumsum(dim=-1)
                accepted_prefix = rejected_through.eq(0) & eval_mask.bool()
                survived = accepted_prefix.sum(dim=-1).ge(
                    self.deep_survival_guard_min_prefix
                )
                deep = torch.arange(self.block_size, device=device).ge(
                    self.deep_survival_guard_start_depth - 1
                ).view(1, 1, -1)
                guard_margin_floor = torch.full_like(
                    rollout_target_margin,
                    self.deep_survival_guard_margin_floor,
                )
                if self.dialogue_dsg_adaptive_margin:
                    late_or_closing = dialogue_credit_features["phase"].ge(
                        3
                    ).unsqueeze(-1)
                    linguistic_boundary = (
                        dialogue_credit_features["depth_boundary"]
                        | dialogue_credit_features[
                            "depth_script_transition"
                        ]
                    )
                    guard_margin_floor = (
                        guard_margin_floor
                        + late_or_closing.float()
                        * self.dialogue_dsg_late_margin_bonus
                        + linguistic_boundary.float()
                        * self.dialogue_dsg_boundary_margin_bonus
                    )
                fragile = rollout_target_margin.detach().lt(
                    guard_margin_floor
                )
                protected = (
                    accepted_prefix
                    & survived.unsqueeze(-1)
                    & deep
                    & fragile
                )
                protected_weight = (
                    ce_weight_mask.detach() * protected.float()
                )
                deep_guard_raw = (
                    F.softplus(
                        (
                            guard_margin_floor - rollout_target_margin
                        )
                        / self.deep_survival_guard_temperature
                    )
                    * protected_weight
                ).sum()
                weighted_deep_guard = (
                    state_scale
                    * self.deep_survival_guard_alpha
                    * deep_guard_raw
                )
                deep_guard_aux_num, budget_scale = self._budget_auxiliary_loss(
                    weighted_deep_guard,
                    self.deep_survival_guard_loss_budget,
                    ce_num,
                )
                prefix_diagnostics.update(
                    {
                        "deep_survival_guard_blocks": survived.sum().detach(),
                        "deep_survival_guard_tokens": protected.sum().detach(),
                        "deep_survival_guard_budget_scale": budget_scale.detach(),
                        "dialogue_dsg_adaptive_floor_mean": (
                            (guard_margin_floor.detach() * protected.float()).sum()
                            / protected.sum().clamp_min(1)
                        ),
                        "dialogue_dsg_late_protected_tokens": (
                            protected & late_or_closing
                        ).sum().detach()
                        if self.dialogue_dsg_adaptive_margin
                        else protected.new_zeros(()).sum(),
                        "dialogue_dsg_boundary_protected_tokens": (
                            protected & linguistic_boundary
                        ).sum().detach()
                        if self.dialogue_dsg_adaptive_margin
                        else protected.new_zeros(()).sum(),
                    }
                )

            if self.conv_gate_loss_alpha > 0:
                gate_logits = self.draft_model.survival_gate_logits()
                if not gate_logits:
                    raise ValueError(
                        "conv_gate_loss_alpha > 0 requires survival-gated convolution"
                    )
                gate_mask = alive_before.new_zeros(alive_before.shape)
                gate_mask[..., 1:] = eval_mask[..., 1:]
                gate_target = alive_before.detach()
                gate_raw = base_logits.new_zeros((), dtype=torch.float32)
                for logits in gate_logits:
                    per_group = F.binary_cross_entropy_with_logits(
                        logits.float(),
                        gate_target.unsqueeze(-1).expand_as(logits),
                        reduction="none",
                    ).mean(dim=-1)
                    gate_raw = gate_raw + (per_group * gate_mask).sum()
                gate_raw = gate_raw / len(gate_logits)
                gate_aux_num = (
                    state_scale * self.conv_gate_loss_alpha * gate_raw
                )
                prefix_diagnostics["conv_gate_supervised_tokens"] = (
                    gate_mask.sum().detach()
                )
                prefix_diagnostics["conv_gate_effective_alpha"] = (
                    gate_raw.new_tensor(state_scale * self.conv_gate_loss_alpha)
                )

        # ---- CARH-conditioned DFlash2 candidate-path selector ----
        # Candidates come from the CARH-corrected distribution.  The teacher
        # branch uses the gold predecessor, while the rollout branch walks its
        # own detached greedy selections, matching serving without attempting
        # to differentiate through discrete path decisions.
        selector_aux_num = base_logits.new_zeros((), dtype=torch.float32)
        selector_diagnostics = {}
        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        selector_teacher_phase, selector_distill_phase = (
            self._selector_phase_scales()
        )
        selector_weight_enabled = max(
            self.selector_loss_alpha,
            self.selector_distill_alpha,
            self.selector_preservation_alpha,
            self.selector_tree_loss_alpha,
        ) > 0 and (
            self.selector_training_mode != "survival-distill"
            or selector_teacher_phase > 0
            or selector_distill_phase > 0
        )
        if (
            candidate_selector is not None
            and self.selector_training_mode == "survival-distill"
            and not selector_weight_enabled
        ):
            selector_diagnostics = {
                "selector_teacher_phase": logits_4d.new_tensor(
                    selector_teacher_phase, dtype=torch.float32
                ),
                "selector_distill_phase": logits_4d.new_tensor(
                    selector_distill_phase, dtype=torch.float32
                ),
            }
        if candidate_selector is not None and selector_weight_enabled:
            strict_distill = self.selector_training_mode == "survival-distill"
            recoverability_aware = (
                self.selector_training_mode == "recoverability-aware"
            )
            strict_support = strict_distill or recoverability_aware
            top_k = int(candidate_selector.top_k)
            if strict_support:
                # Avoid a redundant teacher-forced full-vocabulary Top-k.  The
                # strict objectives below build candidates only from their
                # self-conditioned rollout states.
                teacher_ce = base_logits.new_zeros(target_ids.shape)
            else:
                unary_logits, candidate_ids = logits_4d.float().topk(
                    top_k, dim=-1
                )
                target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
                target_is_candidate = target_matches.any(dim=-1)
                target_candidate_index = torch.where(
                    target_is_candidate,
                    target_matches.long().argmax(dim=-1),
                    torch.full_like(target_ids, top_k - 1),
                )
                # Legacy DFlash2 training injects a missing target into the
                # weakest slot.  The survival-distill mode intentionally does
                # not: its teacher sees exactly the serving candidate set.
                missing_target = ~target_is_candidate
                target_candidate_index = torch.where(
                    target_is_candidate,
                    target_candidate_index,
                    torch.full_like(target_ids, top_k - 1),
                )
                candidate_ids = candidate_ids.clone()
                unary_logits = unary_logits.clone()
                candidate_ids[..., -1] = torch.where(
                    missing_target, target_ids, candidate_ids[..., -1]
                )
                gold_unary = logits_4d.float().gather(
                    -1, target_ids.unsqueeze(-1)
                ).squeeze(-1)
                unary_logits[..., -1] = torch.where(
                    missing_target, gold_unary, unary_logits[..., -1]
                )
                teacher_logits = candidate_selector.score_candidates(
                    candidate_ids=candidate_ids,
                    unary_logits=unary_logits,
                    hidden_states=hidden_4d,
                    predecessor_ids=prev_token_ids,
                )
                teacher_ce = F.cross_entropy(
                    teacher_logits.reshape(-1, top_k),
                    target_candidate_index.reshape(-1),
                    reduction="none",
                ).view_as(target_ids)

            predecessor = anchor_token_ids
            rollout_logits_by_depth = []
            rollout_unary_by_depth = []
            rollout_ids_by_depth = []
            rollout_independent_ids_by_depth = []
            rollout_targets_by_depth = []
            rollout_coverage_by_depth = []
            rollout_active_by_depth = []
            selector_temperatures = None
            if strict_distill:
                selector_temperatures, _ = self._sample_on_policy_temperatures(
                    bsz * n_blocks, device
                )
                selector_temperatures = selector_temperatures.view(bsz, n_blocks)
            for depth in range(self.block_size):
                # Recompute CARH with the actually selected predecessor.  This
                # makes the rollout branch self-conditioned in both CARH and
                # selector state, instead of changing only the selector edge.
                step_hidden = hidden_4d[..., depth, :]
                step_unary_full = (
                    base_logits_4d[..., depth, :].detach()
                    if strict_distill
                    else base_logits_4d[..., depth, :]
                ) + (
                    self.draft_model.markov_head.compute_step_bias(
                        predecessor,
                        hidden_states=(
                            step_hidden.detach() if strict_distill else step_hidden
                        ),
                        depth_idx=depth,
                        **_carh_predecessor_context_kwargs(
                            self.draft_model.markov_head,
                            hidden_4d.detach() if strict_distill else hidden_4d,
                            depth,
                        ),
                    )
                )
                step_unary, step_candidates = step_unary_full.float().topk(
                    top_k, dim=-1
                )
                rollout_independent_ids_by_depth.append(step_candidates[..., 0])
                step_target = target_ids[..., depth]
                step_matches = step_candidates.eq(step_target.unsqueeze(-1))
                step_covered = step_matches.any(dim=-1)
                step_target_index = torch.where(
                    step_covered,
                    step_matches.long().argmax(dim=-1),
                    torch.zeros_like(step_target),
                )
                if not strict_support:
                    step_target_index = torch.where(
                        step_covered,
                        step_target_index,
                        torch.full_like(step_target, top_k - 1),
                    )
                    step_candidates = step_candidates.clone()
                    step_unary = step_unary.clone()
                    step_candidates[..., -1] = torch.where(
                        step_covered, step_candidates[..., -1], step_target
                    )
                    step_gold_unary = step_unary_full.float().gather(
                        -1, step_target.unsqueeze(-1)
                    ).squeeze(-1)
                    step_unary[..., -1] = torch.where(
                        step_covered, step_unary[..., -1], step_gold_unary
                    )
                depth_logits = candidate_selector.score_candidates(
                    candidate_ids=step_candidates,
                    unary_logits=(step_unary.detach() if strict_distill else step_unary),
                    hidden_states=(
                        step_hidden.detach() if strict_distill else step_hidden
                    ),
                    predecessor_ids=predecessor,
                )
                rollout_logits_by_depth.append(depth_logits)
                rollout_unary_by_depth.append(step_unary)
                rollout_targets_by_depth.append(step_target_index)
                rollout_coverage_by_depth.append(step_covered)
                if self.selector_margin_threshold > 0:
                    step_active = (
                        step_unary[..., 0] - step_unary[..., 1]
                    ).detach().lt(self.selector_margin_threshold)
                else:
                    step_active = torch.ones_like(step_covered)
                rollout_active_by_depth.append(step_active)
                if strict_distill:
                    selected_index = self._sample_on_policy_tokens(
                        depth_logits.reshape(-1, top_k),
                        selector_temperatures.reshape(-1),
                    ).view(bsz, n_blocks)
                else:
                    selected_index = depth_logits.detach().argmax(dim=-1)
                selected = step_candidates.gather(
                    -1, selected_index.unsqueeze(-1)
                ).squeeze(-1)
                if recoverability_aware:
                    # Match linear serving: high-margin states retain unary
                    # Top-1 and only low-margin states follow the selector.
                    selected = torch.where(
                        step_active, selected, step_candidates[..., 0]
                    )
                rollout_ids_by_depth.append(selected)
                predecessor = selected
            rollout_logits = torch.stack(rollout_logits_by_depth, dim=-2)
            rollout_unary = torch.stack(rollout_unary_by_depth, dim=-2)
            rollout_ids = torch.stack(rollout_ids_by_depth, dim=-1)
            rollout_independent_ids = torch.stack(
                rollout_independent_ids_by_depth, dim=-1
            )
            rollout_target_index = torch.stack(rollout_targets_by_depth, dim=-1)
            rollout_coverage = torch.stack(rollout_coverage_by_depth, dim=-1)
            rollout_active = torch.stack(rollout_active_by_depth, dim=-1)
            rollout_ce = F.cross_entropy(
                rollout_logits.reshape(-1, top_k),
                rollout_target_index.reshape(-1),
                reduction="none",
            ).view_as(target_ids)

            # MACT tree-coverage surrogate.  At a prefix-reachable, low-margin
            # state, place the target token above the serving branch-width
            # frontier.  Unlike ordinary selector CE, this objective only asks
            # for inclusion in the conditional sibling set; it does not force
            # every useful alternative to become Top-1.
            tree_raw = rollout_logits.new_zeros(())
            tree_weight_mass = rollout_logits.new_zeros(())
            if self.selector_tree_loss_alpha > 0:
                branch_width = min(self.selector_tree_branch_width, top_k)
                rollout_correct_for_tree = (
                    rollout_ids.eq(target_ids) & eval_mask.bool()
                )
                rejected_before_tree = torch.cat(
                    [
                        torch.zeros_like(
                            rollout_correct_for_tree[..., :1], dtype=torch.int64
                        ),
                        (
                            (~rollout_correct_for_tree) & eval_mask.bool()
                        )[..., :-1].cumsum(dim=-1),
                    ],
                    dim=-1,
                )
                alive_before_tree = rejected_before_tree.eq(0) & eval_mask.bool()
                if self.selector_margin_threshold > 0:
                    tree_active = (
                        rollout_unary[..., 0] - rollout_unary[..., 1]
                    ).detach().lt(self.selector_margin_threshold)
                else:
                    tree_active = torch.ones_like(alive_before_tree)
                tree_weight = (
                    ce_weight_mask.detach()
                    * alive_before_tree.float()
                    * tree_active.float()
                    * rollout_coverage.float()
                )
                target_index = rollout_target_index.unsqueeze(-1)
                gold_score = rollout_logits.gather(-1, target_index).squeeze(-1)
                # Compare against the B-th strongest *non-target* candidate.
                # If the target itself sits exactly at rank B, using the raw
                # B-th order statistic makes frontier_score == gold_score and
                # cancels the gradient.  Masking the target gives a genuine
                # coverage-margin objective on the serving sibling frontier.
                competing_logits = rollout_logits.masked_fill(
                    F.one_hot(
                        rollout_target_index, num_classes=top_k
                    ).bool(),
                    torch.finfo(rollout_logits.dtype).min,
                )
                frontier_score = competing_logits.topk(
                    branch_width, dim=-1
                ).values[..., -1]
                tree_per_token = F.softplus(
                    (
                        frontier_score
                        - gold_score
                        + self.selector_tree_margin
                    )
                    / self.selector_tree_temperature
                )
                tree_raw = (tree_per_token * tree_weight).sum()
                tree_weight_mass = tree_weight.sum()

            if recoverability_aware:
                # RA-MACR uses the exact serving candidate support.  It spends
                # ranking credit only where unary Top-1 is wrong but the target
                # remains recoverable inside Top-K, and spends preservation
                # credit where unary Top-1 is already correct.  Target-missing
                # states receive no impossible ranking target.
                valid = eval_mask.bool()
                rollout_correct = rollout_ids.eq(target_ids) & valid
                rejected_before = torch.cat(
                    [
                        torch.zeros_like(
                            rollout_correct[..., :1], dtype=torch.int64
                        ),
                        ((~rollout_correct) & valid)[..., :-1].cumsum(dim=-1),
                    ],
                    dim=-1,
                )
                alive_before = rejected_before.eq(0) & valid
                unary_correct = (
                    rollout_independent_ids.eq(target_ids) & alive_before
                )
                recoverable = (
                    rollout_coverage
                    & ~rollout_independent_ids.eq(target_ids)
                    & alive_before
                )
                recall_miss = ~rollout_coverage & alive_before
                active_recoverable = recoverable & rollout_active
                active_preservation = unary_correct & rollout_active
                recovery_weight = (
                    ce_weight_mask.detach() * active_recoverable.float()
                )
                preservation_weight = (
                    ce_weight_mask.detach() * active_preservation.float()
                )
                recovery_raw = (rollout_ce * recovery_weight).sum()

                target_index = rollout_target_index.unsqueeze(-1)
                target_score = rollout_logits.gather(
                    -1, target_index
                ).squeeze(-1)
                top2_values, top2_indices = rollout_logits.topk(2, dim=-1)
                best_other = torch.where(
                    top2_indices[..., 0].eq(rollout_target_index),
                    top2_values[..., 1],
                    top2_values[..., 0],
                )
                preservation_per_token = F.softplus(
                    (best_other - target_score)
                    / self.selector_distill_temperature
                )
                preservation_raw = (
                    preservation_per_token * preservation_weight
                ).sum()
                weighted_selector = (
                    self.selector_loss_alpha * recovery_raw
                    + self.selector_preservation_alpha * preservation_raw
                    + self.selector_tree_loss_alpha * tree_raw
                )
                selector_aux_num, selector_budget_scale = (
                    self._budget_auxiliary_loss(
                        weighted_selector, self.selector_loss_budget, ce_num
                    )
                )

                with torch.no_grad():
                    reachable_count = alive_before.sum().clamp_min(1)
                    recoverable_count = active_recoverable.sum().clamp_min(1)
                    preservation_count = active_preservation.sum().clamp_min(1)
                    recovered = active_recoverable & rollout_ids.eq(target_ids)
                    overcorrected = active_preservation & rollout_ids.ne(target_ids)
                    selector_diagnostics = {
                        "selector_loss": (
                            recovery_raw / recovery_weight.sum().clamp_min(1e-6)
                        ).detach(),
                        "selector_preservation_loss": (
                            preservation_raw
                            / preservation_weight.sum().clamp_min(1e-6)
                        ).detach(),
                        "selector_active_rate": (
                            (rollout_active & alive_before).sum().float()
                            / reachable_count
                        ),
                        "selector_strict_topk_coverage": (
                            (rollout_coverage & alive_before).sum().float()
                            / reachable_count
                        ),
                        "selector_recoverable_rate": (
                            recoverable.sum().float() / reachable_count
                        ),
                        "selector_active_recoverable_rate": (
                            active_recoverable.sum().float() / reachable_count
                        ),
                        "selector_active_preservation_rate": (
                            active_preservation.sum().float() / reachable_count
                        ),
                        "selector_recall_miss_rate": (
                            recall_miss.sum().float() / reachable_count
                        ),
                        "selector_recovery_accuracy": (
                            recovered.sum().float() / recoverable_count
                        ),
                        "selector_overcorrection_rate": (
                            overcorrected.sum().float() / preservation_count
                        ),
                        "selector_path_accuracy": (
                            (rollout_ids.eq(target_ids) & alive_before).sum().float()
                            / reachable_count
                        ),
                        "selector_loss_alpha": logits_4d.new_tensor(
                            self.selector_loss_alpha, dtype=torch.float32
                        ),
                        "selector_loss_budget_scale": (
                            selector_budget_scale.detach()
                        ),
                    }
            elif not strict_distill:
                rho = self.selector_rollout_ratio
                if self.selector_margin_threshold > 0:
                    # Train the selector on the same uncertain unary states on
                    # which serving will invoke it.  The gate is detached: it
                    # routes supervision but does not encourage the backbone to
                    # manufacture artificially small margins.
                    teacher_top2 = logits_4d.float().topk(2, dim=-1).values
                    teacher_selector_active = (
                        teacher_top2[..., 0] - teacher_top2[..., 1]
                    ).detach().lt(self.selector_margin_threshold)
                    rollout_selector_active = (
                        rollout_unary[..., 0] - rollout_unary[..., 1]
                    ).detach().lt(self.selector_margin_threshold)
                    teacher_weight_mask = ce_weight_mask * (
                        teacher_selector_active.to(ce_weight_mask.dtype)
                    )
                    rollout_weight_mask = ce_weight_mask * (
                        rollout_selector_active.to(ce_weight_mask.dtype)
                    )
                else:
                    teacher_selector_active = torch.ones_like(
                        target_ids, dtype=torch.bool
                    )
                    rollout_selector_active = torch.ones_like(
                        target_ids, dtype=torch.bool
                    )
                    teacher_weight_mask = ce_weight_mask
                    rollout_weight_mask = ce_weight_mask
                selector_weighted_num = (
                    (1.0 - rho) * teacher_ce * teacher_weight_mask
                    + rho * rollout_ce * rollout_weight_mask
                )
                selector_weight_mass = (
                    (1.0 - rho) * teacher_weight_mask
                    + rho * rollout_weight_mask
                ).sum()
                selector_aux_num = self.selector_loss_alpha * (
                    selector_weighted_num.sum()
                ) + self.selector_tree_loss_alpha * tree_raw
                selector_budget_scale = selector_aux_num.new_ones(())
                if self.selector_tree_loss_alpha > 0:
                    selector_aux_num, selector_budget_scale = (
                        self._budget_auxiliary_loss(
                            selector_aux_num, self.selector_loss_budget, ce_num
                        )
                    )

                with torch.no_grad():
                    valid_count = eval_mask.sum().clamp_min(1.0)
                    independent_ids = logits_4d.argmax(dim=-1)
                    changed = rollout_ids.ne(independent_ids) & eval_mask.bool()
                    changed_success = changed & rollout_ids.eq(target_ids)
                    selector_diagnostics = {
                        "selector_loss": (
                            selector_weighted_num.sum()
                            / selector_weight_mass.clamp_min(1e-6)
                        ).detach(),
                        "selector_active_rate": (
                            rollout_selector_active.float() * eval_mask
                        ).sum()
                        / valid_count,
                        "selector_teacher_active_rate": (
                            teacher_selector_active.float() * eval_mask
                        ).sum()
                        / valid_count,
                        "selector_margin_threshold": logits_4d.new_tensor(
                            self.selector_margin_threshold, dtype=torch.float32
                        ),
                        "selector_oracle_hit_rate": (
                            rollout_coverage.float() * eval_mask
                        ).sum()
                        / valid_count,
                        "selector_path_accuracy": (
                            rollout_ids.eq(target_ids).float() * eval_mask
                        ).sum()
                        / valid_count,
                        "selector_changed_top1_rate": changed.float().sum()
                        / valid_count,
                        "selector_changed_top1_success_rate": (
                            changed_success.float().sum()
                            / changed.float().sum().clamp_min(1.0)
                        ),
                        "selector_loss_alpha": logits_4d.new_tensor(
                            self.selector_loss_alpha, dtype=torch.float32
                        ),
                        "selector_rollout_ratio": logits_4d.new_tensor(
                            rho, dtype=torch.float32
                        ),
                        "selector_tree_coverage_loss": (
                            tree_raw / tree_weight_mass.clamp_min(1e-6)
                        ).detach(),
                        "selector_tree_active_mass": tree_weight_mass.detach(),
                        "selector_tree_branch_width": logits_4d.new_tensor(
                            self.selector_tree_branch_width, dtype=torch.float32
                        ),
                        "selector_loss_budget_scale": (
                            selector_budget_scale.detach()
                        ),
                    }
            else:
                # Prefix reachability is defined by the selector's actual
                # detached rollout, not by teacher-forced predecessor states.
                valid = eval_mask.bool()
                rollout_correct = rollout_ids.eq(target_ids) & valid
                rejected_before = torch.cat(
                    [
                        torch.zeros_like(rollout_correct[..., :1], dtype=torch.int64),
                        ((~rollout_correct) & valid)[..., :-1].cumsum(dim=-1),
                    ],
                    dim=-1,
                )
                alive_before = rejected_before.eq(0) & valid
                unary_correct = rollout_independent_ids.eq(target_ids) & alive_before
                recoverable = (
                    rollout_coverage
                    & ~rollout_independent_ids.eq(target_ids)
                    & alive_before
                )
                recall_miss = ~rollout_coverage & alive_before
                survival_weight = ce_weight_mask.detach() * alive_before.float()
                recovery_weight = survival_weight * recoverable.float()
                preservation_weight = survival_weight * unary_correct.float()

                teacher_scale = selector_teacher_phase
                distill_scale = selector_distill_phase
                teacher_raw = (rollout_ce * recovery_weight).sum()

                temperature = self.selector_distill_temperature
                teacher_log_probs = F.log_softmax(
                    rollout_logits.detach() / temperature, dim=-1
                )
                teacher_probs = teacher_log_probs.exp()
                student_log_probs = F.log_softmax(
                    rollout_unary / temperature, dim=-1
                )
                student_probs = student_log_probs.exp()
                target_index = rollout_target_index.unsqueeze(-1)
                teacher_gold = teacher_probs.gather(-1, target_index).squeeze(-1)
                student_gold = student_probs.gather(-1, target_index).squeeze(-1)
                advantage = (
                    teacher_gold - student_gold
                ).detach() > self.selector_advantage_margin
                distill_weight = recovery_weight * advantage.float()
                distill_kl = (
                    teacher_probs
                    * (teacher_log_probs - student_log_probs)
                ).sum(dim=-1)
                distill_raw = (distill_kl * distill_weight).sum()

                teacher_gold_logit = rollout_logits.gather(
                    -1, target_index
                ).squeeze(-1)
                top2_values, top2_indices = rollout_logits.topk(2, dim=-1)
                best_other = torch.where(
                    top2_indices[..., 0].eq(rollout_target_index),
                    top2_values[..., 1],
                    top2_values[..., 0],
                )
                preservation_raw = (
                    F.softplus(-(teacher_gold_logit - best_other))
                    * preservation_weight
                ).sum()

                weighted_selector = (
                    teacher_scale * self.selector_loss_alpha * teacher_raw
                    + distill_scale * self.selector_distill_alpha * distill_raw
                    + teacher_scale
                    * self.selector_preservation_alpha
                    * preservation_raw
                    + teacher_scale * self.selector_tree_loss_alpha * tree_raw
                )
                selector_aux_num, selector_budget_scale = (
                    self._budget_auxiliary_loss(
                        weighted_selector, self.selector_loss_budget, ce_num
                    )
                )

                with torch.no_grad():
                    reachable_count = alive_before.sum().clamp_min(1)
                    recoverable_count = recoverable.sum().clamp_min(1)
                    correct_count = unary_correct.sum().clamp_min(1)
                    teacher_selected = rollout_ids
                    recovered = recoverable & teacher_selected.eq(target_ids)
                    overcorrected = unary_correct & teacher_selected.ne(target_ids)
                    selector_diagnostics = {
                        "selector_loss": (
                            teacher_raw / recovery_weight.sum().clamp_min(1e-6)
                        ).detach(),
                        "selector_distill_loss": (
                            distill_raw / distill_weight.sum().clamp_min(1e-6)
                        ).detach(),
                        "selector_preservation_loss": (
                            preservation_raw
                            / preservation_weight.sum().clamp_min(1e-6)
                        ).detach(),
                        "selector_strict_topk_coverage": (
                            (rollout_coverage & alive_before).sum().float()
                            / reachable_count
                        ),
                        "selector_recoverable_rate": (
                            recoverable.sum().float() / reachable_count
                        ),
                        "selector_recall_miss_rate": (
                            recall_miss.sum().float() / reachable_count
                        ),
                        "selector_teacher_recovery_accuracy": (
                            recovered.sum().float() / recoverable_count
                        ),
                        "selector_teacher_overcorrection_rate": (
                            overcorrected.sum().float() / correct_count
                        ),
                        "selector_advantage_rate": (
                            (advantage & recoverable).sum().float()
                            / recoverable_count
                        ),
                        "selector_teacher_target_probability": (
                            (teacher_gold * recoverable.float()).sum()
                            / recoverable_count
                        ),
                        "selector_student_target_probability": (
                            (student_gold.detach() * recoverable.float()).sum()
                            / recoverable_count
                        ),
                        "selector_probability_gain": (
                            (
                                (teacher_gold - student_gold.detach())
                                * recoverable.float()
                            ).sum()
                            / recoverable_count
                        ),
                        "selector_teacher_phase": logits_4d.new_tensor(
                            teacher_scale, dtype=torch.float32
                        ),
                        "selector_distill_phase": logits_4d.new_tensor(
                            distill_scale, dtype=torch.float32
                        ),
                        "selector_loss_budget_scale": selector_budget_scale.detach(),
                        "selector_loss_alpha": logits_4d.new_tensor(
                            self.selector_loss_alpha, dtype=torch.float32
                        ),
                        "selector_tree_coverage_loss": (
                            tree_raw / tree_weight_mass.clamp_min(1e-6)
                        ).detach(),
                        "selector_tree_active_mass": tree_weight_mass.detach(),
                        "selector_tree_branch_width": logits_4d.new_tensor(
                            self.selector_tree_branch_width, dtype=torch.float32
                        ),
                    }
                    for depth in range(self.block_size):
                        depth_reachable = alive_before[..., depth]
                        depth_recoverable = recoverable[..., depth]
                        depth_reachable_count = depth_reachable.sum().clamp_min(1)
                        depth_recoverable_count = (
                            depth_recoverable.sum().clamp_min(1)
                        )
                        selector_diagnostics.update(
                            {
                                f"selector_strict_topk_coverage_depth_{depth + 1}": (
                                    (
                                        rollout_coverage[..., depth]
                                        & depth_reachable
                                    ).sum().float()
                                    / depth_reachable_count
                                ),
                                f"selector_recoverable_rate_depth_{depth + 1}": (
                                    depth_recoverable.sum().float()
                                    / depth_reachable_count
                                ),
                                f"selector_teacher_recovery_depth_{depth + 1}": (
                                    (
                                        depth_recoverable
                                        & teacher_selected[..., depth].eq(
                                            target_ids[..., depth]
                                        )
                                    ).sum().float()
                                    / depth_recoverable_count
                                ),
                            }
                        )

        # ---- Confidence head BCE ----
        conf_num = base_logits.new_zeros((), dtype=torch.float32)
        if (
            self.draft_model.confidence_head is not None
            and self.confidence_head_alpha > 0
        ):
            if self.draft_model.confidence_head_with_markov:
                prev_emb = self.draft_model.markov_head.get_prev_embeddings(
                    prev_token_ids
                ).to(hidden_4d.dtype)
                if recoverability_enabled:
                    route = carh_recoverable_mask.unsqueeze(-1)
                    prev_emb = torch.where(route, prev_emb, prev_emb.detach())
                conf_features = torch.cat([hidden_4d, prev_emb], dim=-1)
            else:
                conf_features = hidden_4d
            if self.confidence_detach_backbone:
                conf_features = conf_features.detach()
            confidence_pred = self.draft_model.confidence_head(conf_features).float()
            confidence_target = accept_rate.detach()
            confidence_weight = decay_weight_mask
            if self.vat_enabled:
                confidence_target = vat_confidence_target.detach()
                confidence_weight = vat_confidence_weight
            elif self.confidence_target_mode == "on-policy-survival":
                on_policy_confidence = getattr(
                    self, "_last_on_policy_confidence", None
                )
                if on_policy_confidence is None:
                    confidence_weight = torch.zeros_like(decay_weight_mask)
                else:
                    confidence_target, confidence_weight = on_policy_confidence
                    confidence_target = confidence_target.detach()
                    confidence_weight = confidence_weight * decay_weight_mask
            conf_bce = F.binary_cross_entropy_with_logits(
                confidence_pred, confidence_target, reduction="none"
            ) * confidence_weight
            conf_num = conf_bce.sum()

        # ---- Pooled global loss (DeepSpec _build_loss) ----
        # Local numerators over a cross-rank-summed denominator, x dp_size to
        # cancel FSDP's mean gradient reduction -> a true token-pooled global mean
        # rather than a mean-of-per-rank-means.
        from specforge.distributed import get_dp_group

        dp_group = get_dp_group() if dist.is_initialized() else None
        dp_size = dist.get_world_size(dp_group) if dp_group is not None else 1
        global_den = local_den.detach().clone()
        if dp_size > 1:
            dist.all_reduce(global_den, op=dist.ReduceOp.SUM, group=dp_group)
        global_den = global_den + 1e-6
        bv_loss = base_logits.new_zeros((), dtype=torch.float32)
        if bv_terms is not None:
            # BV is a mean over valid BLOCKS, separate from CE/aux token mass.
            global_bv_den = bv_terms.denominator.detach().clone()
            if dp_size > 1:
                dist.all_reduce(global_bv_den, op=dist.ReduceOp.SUM, group=dp_group)
            bv_loss = bv_terms.numerator / global_bv_den.clamp_min(1.0)
        global_predictive_den = predictive_den.detach().clone()
        if (self.conv_source_semantic_alpha > 0 or self.carh_reference_calibration_alpha > 0) and dp_size > 1:
            dist.all_reduce(global_predictive_den, op=dist.ReduceOp.SUM, group=dp_group)
        global_predictive_den = global_predictive_den.clamp_min(1e-6)
        global_offline_lk_den = offline_lk_den.detach().clone()
        global_offline_e2e_den = offline_e2e_den.detach().clone()
        if self.offline_acceptance_objective != "none" and dp_size > 1:
            dist.all_reduce(
                global_offline_lk_den, op=dist.ReduceOp.SUM, group=dp_group
            )
            dist.all_reduce(
                global_offline_e2e_den, op=dist.ReduceOp.SUM, group=dp_group
            )
        global_offline_lk_den = global_offline_lk_den.clamp_min(1e-6)
        global_offline_e2e_den = global_offline_e2e_den.clamp_min(1e-6)
        hard_loss_alpha = (
            self.vat_hard_loss_alpha
            if self.vat_enabled
            else self.ce_loss_alpha
        )
        verification_head_alpha = (
            self.vat_verification_head_alpha
            if self.vat_enabled
            else self.confidence_head_alpha
        )
        auxiliary_loss = (
            predictive_scale * (
                self.conv_source_semantic_alpha * semantic_num
                + self.carh_reference_calibration_alpha * reference_num
            ) / global_predictive_den
            + self.vat_soft_loss_alpha * vat_soft_num / global_den
            + self.ce_loss_alpha * rollout_aux_num / global_den
            + prefix_aux_num / global_den
            + frc_aux_num / global_den
            + dfap_aux_num / global_den
            + transition_aux_num / global_den
            + transition2_aux_num / global_den
            + bottleneck_aux_num / global_den
            + deep_guard_aux_num / global_den
            + gate_aux_num / global_den
            + carh_recoverability_aux_num / global_den
            + recall_aux_num / global_den
            + selector_aux_num / global_den
            + on_policy_aux_num / global_den
            + elastic_aux_num / global_den
            + verification_head_alpha * conf_num / global_den
        )
        offline_e2e_blend = self._offline_acceptance_e2e_blend()
        if self.offline_acceptance_objective == "angel-lk-e2e":
            offline_lk_loss = offline_lk_num / global_offline_lk_den
            offline_e2e_loss = offline_e2e_num / global_offline_e2e_den
            offline_stochastic_loss = (
                (1.0 - offline_e2e_blend) * offline_lk_loss
                + offline_e2e_blend * offline_e2e_loss
            )
            loss = (
                self.offline_acceptance_greedy_weight * ce_num / global_den
                + self.offline_acceptance_stochastic_weight
                * offline_stochastic_loss
                + auxiliary_loss
            ) * dp_size
        else:
            offline_lk_loss = offline_lk_num.new_zeros(())
            offline_e2e_loss = offline_e2e_num.new_zeros(())
            loss = (
                hard_loss_alpha * ce_num / global_den
                + auxiliary_loss
                + self.l1_loss_alpha * l1_num / global_den
                + self.bv_loss_alpha * bv_loss
            ) * dp_size

        counterfactual_summary_loss = loss.new_zeros(())
        if self.block_summary_counterfactual_alpha > 0:
            # Compare the same teacher-forced block with/without its summary.
            # This is a cheap greedy proxy, not an on-policy verification event.
            max_blocks = min(8, bsz * n_blocks)
            sampled_blocks = torch.linspace(
                0, bsz * n_blocks - 1, steps=max_blocks, device=device
            ).long()
            raw = hidden_before_summary.reshape(-1, self.block_size, hidden_4d.size(-1))[sampled_blocks]
            gold = target_ids.reshape(-1, self.block_size)[sampled_blocks]
            valid = eval_bool.reshape(-1, self.block_size)[sampled_blocks]
            with torch.no_grad():
                off_logits = self.lm_head(raw.reshape(-1, raw.size(-1))).float().view(max_blocks, self.block_size, -1)
                off_logp = off_logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1) - off_logits.logsumexp(dim=-1)
                off_correct = off_logits.argmax(dim=-1).eq(gold) & valid
                survived_before = torch.cat((
                    torch.ones_like(off_correct[:, :1]),
                    off_correct[:, :-1].to(torch.int32).cumprod(dim=-1).bool(),
                ), dim=-1)
                repair = valid & survived_before & ~off_correct
                protect = valid & survived_before & off_correct
                repair[:, 0] = False  # The anchor slot is intentionally unchanged.
                protect[:, :2] = False
            on_logits = base_logits_4d.reshape(-1, self.block_size, vocab_size)[sampled_blocks].float()
            on_logp = on_logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1) - on_logits.logsumexp(dim=-1)
            delta = on_logp - off_logp
            repair_num = (torch.relu(0.05 - delta) * repair).sum()
            protect_num = (torch.relu(-delta - 0.01) * protect).sum()
            counterfactual_summary_loss = (
                repair_num + protect_num
            ) / (repair.sum() + protect.sum()).clamp_min(1)
            loss = loss + self.block_summary_counterfactual_alpha * counterfactual_summary_loss

        # Per-component loss values (per-rank local means) for logging only — lets
        # you watch L1 fall while the greedy-CE proxy plateaus.
        local_den_eps = local_den + 1e-6
        loss_components = {
            "source_semantic_loss": (semantic_num / predictive_den.clamp_min(1e-6)).detach(),
            "reference_calibration_loss": (reference_num / predictive_den.clamp_min(1e-6)).detach(),
            "reference_cosine_error": (reference_cosine_num / predictive_den.clamp_min(1e-6)).detach(),
            "reference_relative_mse": (reference_mse_num / predictive_den.clamp_min(1e-6)).detach(),
            "reference_norm_ratio": (reference_ratio_num / predictive_den.clamp_min(1e-6)).detach(),
            "predictive_aux_weighted_count": predictive_den.detach(),
            "source_semantic_effective_alpha": local_den.new_tensor(predictive_scale * self.conv_source_semantic_alpha),
            "reference_calibration_effective_alpha": local_den.new_tensor(predictive_scale * self.carh_reference_calibration_alpha),
            "block_summary_counterfactual_loss": counterfactual_summary_loss.detach(),
            "ce_loss": (ce_num / local_den_eps).detach(),
            "carh_sampled_prefix_memory_scale": base_logits.new_tensor(
                getattr(memory_head, "sampled_prefix_memory_scale", 0.0)
                if getattr(memory_head, "sampled_prefix_memory", None) is not None
                else 0.0,
                dtype=torch.float32,
            ),
            "l1_loss": (l1_num / local_den_eps).detach(),
            "confidence_loss": (conf_num / local_den_eps).detach(),
            "vat_soft_loss": (vat_soft_num / local_den_eps).detach(),
            "pace_aux_loss": (rollout_aux_num / local_den_eps).detach(),
            "prefix_aux_loss": (prefix_aux_num / local_den_eps).detach(),
            "shallow_frc_loss": (frc_aux_num / local_den_eps).detach(),
            "dfap_loss": (dfap_aux_num / local_den_eps).detach(),
            "transition_credit_loss": (
                transition_aux_num / local_den_eps
            ).detach(),
            "transition2_margin_loss": (
                transition2_aux_num / local_den_eps
            ).detach(),
            "prefix_bottleneck_loss": (
                bottleneck_aux_num / local_den_eps
            ).detach(),
            "deep_survival_guard_loss": (
                deep_guard_aux_num / local_den_eps
            ).detach(),
            "conv_gate_loss": (gate_aux_num / local_den_eps).detach(),
            "carh_recoverability_loss": (
                carh_recoverability_aux_num / local_den_eps
            ).detach(),
            "recall_correction_loss": (
                recall_aux_num / local_den_eps
            ).detach(),
            "selector_aux_loss": (selector_aux_num / local_den_eps).detach(),
            "on_policy_survival_loss": (
                on_policy_aux_num / local_den_eps
            ).detach(),
            "elastic_projective_loss": (
                elastic_aux_num / local_den_eps
            ).detach(),
            "offline_acceptance_lk_loss": (
                offline_lk_num / offline_lk_den.clamp_min(1e-6)
            ).detach(),
            "offline_acceptance_e2e_loss": (
                offline_e2e_num / offline_e2e_den.clamp_min(1e-6)
            ).detach(),
            "offline_acceptance_e2e_blend": base_logits.new_tensor(
                offline_e2e_blend, dtype=torch.float32
            ),
            "offline_acceptance_mean": offline_acceptance_mean.detach(),
            "offline_expected_draft_length": offline_expected_length.detach(),
            "offline_acceptance_lk_lambda_mean": (
                offline_lk_lambda_mean.detach()
            ),
        }
        if bv_terms is not None:
            bv_local_den = bv_terms.denominator.clamp_min(1.0)
            loss_components.update({
                "bv_loss": (bv_terms.numerator / bv_local_den).detach(),
                "bv_weighted_loss": (self.bv_loss_alpha * bv_terms.numerator / bv_local_den).detach(),
                "bv_beta": base_logits.new_tensor(current_bv_beta, dtype=torch.float32),
                "bv_block_log_loss": (bv_terms.block_log_sum / bv_local_den).detach(),
                "bv_estimated_draft_length": (bv_terms.length_sum / bv_local_den).detach(),
                "bv_valid_blocks": bv_terms.denominator.detach(),
                "bv_floor_fraction": (bv_terms.floored_positions / bv_terms.valid_positions.clamp_min(1)).detach(),
            })
        loss_components.update(pace_diagnostics)
        loss_components.update(carh_pair_diagnostics)
        loss_components.update(prefix_diagnostics)
        loss_components.update(carh_recoverability_diagnostics)
        loss_components.update(recall_diagnostics)
        loss_components.update(selector_diagnostics)
        loss_components.update(on_policy_diagnostics)
        loss_components.update(vat_diagnostics)
        loss_components.update(elastic_diagnostics)
        fusion_weights = getattr(
            self.draft_model, "target_layer_fusion_weights", None
        )
        if fusion_weights is not None:
            with torch.no_grad():
                normalized_fusion = torch.softmax(
                    fusion_weights.float(), dim=-1
                )
                fusion_entropy = -(
                    normalized_fusion
                    * normalized_fusion.clamp_min(1e-12).log()
                ).sum(dim=-1)
                loss_components.update(
                    {
                        "target_layer_fusion_entropy": fusion_entropy.mean(),
                        "target_layer_fusion_max_weight": (
                            normalized_fusion.max(dim=-1).values.mean()
                        ),
                    }
                )
                loss_components.update(_target_fusion_source_metrics(normalized_fusion))

        # ---- Metrics (cross-entropy based; all block_size slots are productive) ----
        with torch.no_grad():
            flat_binary = eval_mask.reshape(-1)
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (flat_binary > 0.5)
            accuracy = correct.sum().float() / flat_binary.sum().clamp(min=1e-6)

            base_pred_ids = base_logits_4d.argmax(dim=-1)
            corrected_pred_ids = logits_4d.argmax(dim=-1)
            valid = eval_mask.bool()
            beneficial = (
                base_pred_ids.ne(target_ids)
                & corrected_pred_ids.eq(target_ids)
                & valid
            )
            harmful = (
                base_pred_ids.eq(target_ids)
                & corrected_pred_ids.ne(target_ids)
                & valid
            )
            valid_count = eval_mask.sum().clamp_min(1.0)
            loss_components["causal_beneficial_flip_rate"] = (
                beneficial.sum().float() / valid_count
            )
            loss_components["causal_harmful_flip_rate"] = (
                harmful.sum().float() / valid_count
            )
            gate_mean = getattr(
                self.draft_model.markov_head, "_last_gate_mean", None
            )
            if gate_mean is not None:
                loss_components["carh_gate_mean"] = gate_mean.detach()
            conv_gate_logits = self.draft_model.survival_gate_logits()
            if conv_gate_logits:
                conv_gates = torch.stack(
                    [torch.sigmoid(value.float()) for value in conv_gate_logits],
                    dim=0,
                )
                loss_components["conv_survival_gate_mean"] = (
                    conv_gates.mean().detach()
                )
                for depth in range(self.block_size):
                    loss_components[f"conv_survival_gate_depth_{depth + 1}"] = (
                        conv_gates[:, :, :, depth, :].mean().detach()
                    )
            prefix_survival_logits = (
                self.draft_model.prefix_state_survival_logits()
            )
            if prefix_survival_logits is not None:
                prefix_survival = torch.sigmoid(prefix_survival_logits.float())
                loss_components["spsm_survival_gate_mean"] = (
                    prefix_survival.mean().detach()
                )
                for depth in range(self.block_size):
                    loss_components[f"spsm_survival_gate_depth_{depth + 1}"] = (
                        prefix_survival[:, depth].mean().detach()
                    )
            for name, value in self.draft_model.prefix_state_diagnostics().items():
                loss_components[f"spsm_{name}"] = value.detach()

            count_per_position = eval_mask.sum(dim=(0, 1))
            count_pp = count_per_position.clamp(min=1.0)
            loss_per_position = (ce_per_token * eval_mask).sum(dim=(0, 1)) / count_pp
            acc_per_position = (
                correct.view(bsz, n_blocks, self.block_size).float().sum(dim=(0, 1))
                / count_pp
            )

        netprefix_hook = getattr(self, "netprefix_hook", None)
        if netprefix_hook is not None and netprefix_hook.mode != "off":
            netprefix_loss = netprefix_hook(
                base_logits_4d, hidden_4d, anchor_positions, eval_mask,
                input_ids, attention_mask,
            )
            # Reference replay trains only the one-sided prefix constraint;
            # it must not silently add another ordinary training batch.
            loss = netprefix_loss if netprefix_hook.mode == "protect" else loss + netprefix_loss
            loss_components["netprefix_aux_loss"] = netprefix_loss.detach()

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )
