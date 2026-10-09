# coding=utf-8
"""DFlash Training Wrapper."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.legacy.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None

# NPU workaround: flex_attention is not available on Ascend NPU.
if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False


_VALID_LOSS_TYPES = {
    "dflash",
    "vp_drafter",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
    "davca-scalar",
    "davca",
    "cva-direct",
    "cva-residual",
    "cfu-frontier",
    "cfu-hard",
    "cfu-soft",
    "cfu-expected",
    "cfu-advantage",
    "vcrd",
}
_DPACE_LOSS_TYPES = {
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_DAVCA_LOSS_TYPES = {"davca-scalar", "davca"}
_CVA_LOSS_TYPES = {"cva-direct", "cva-residual"}
_CFU_LOSS_TYPES = {
    "cfu-frontier",
    "cfu-hard",
    "cfu-soft",
    "cfu-expected",
    "cfu-advantage",
}
_VCRD_LOSS_TYPES = {"vcrd"}
_VALID_LK_LOSS_TYPES = {None, "alpha", "lambda", "tv"}


def _selector_training_candidates(
    objective_logits: torch.Tensor,
    target_ids: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the serving-aligned DFlash2 K-way selector training set.

    Training and serving both use the strict unary top-k. Targets outside this
    candidate set are backbone-recall failures and are masked from selector CE
    instead of being injected into the candidate set.
    """

    _, strict_candidate_ids = objective_logits.topk(top_k, dim=-1)
    target_matches = strict_candidate_ids.eq(target_ids.unsqueeze(-1))
    contains_target = target_matches.any(dim=-1)
    target_positions = target_matches.long().argmax(dim=-1)
    strict_unary_logits = objective_logits.gather(-1, strict_candidate_ids)
    return (
        strict_unary_logits,
        strict_candidate_ids,
        target_positions,
        contains_target,
    )


class VerificationSurvivalCritic(nn.Module):
    """Training-only residual survival calibrator used by DAVCA.

    The critic predicts a residual correction to D-PACE's conditional survival
    proxy.  Its output layer is zero-initialized, so DAVCA starts exactly from
    the D-PACE survival estimate instead of injecting random position weights.
    ``scalar`` processes positions independently; ``dependency`` uses a small
    GRU to model block-internal error dependence.
    """

    def __init__(self, feature_size: int, hidden_size: int, mode: str) -> None:
        super().__init__()
        if mode not in {"scalar", "dependency"}:
            raise ValueError(f"unknown DAVCA critic mode {mode!r}")
        self.mode = mode
        if mode == "scalar":
            self.encoder = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1),
            )
            output = self.encoder[-1]
        else:
            self.encoder = nn.GRU(
                input_size=feature_size,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.output = nn.Linear(hidden_size, 1)
            output = self.output
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.mode == "scalar":
            return self.encoder(features).squeeze(-1)
        hidden, _ = self.encoder(features)
        return self.output(hidden).squeeze(-1)


def create_dflash_sdpa_mask(
    anchor_positions,
    block_keep_mask,
    S,
    block_size,
    device,
    active_block_size=None,
):
    active_block_size = (
        block_size if active_block_size is None else int(active_block_size)
    )
    if not 1 <= active_block_size <= block_size:
        raise ValueError("active_block_size must be in [1, block_size]")
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size
    q_depth = q_indices % block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    kv_depth = (kv_indices - S) % block_size
    active_query = q_depth < active_block_size
    active_draft = kv_depth < active_block_size
    mask_draft = (
        is_draft
        & (q_block_ids == kv_block_ids)
        & active_query
        & active_draft
    )

    # The fixed-shape K_max graph still contains inactive query rows. Give
    # each such row a self edge to avoid an all-masked softmax; their outputs
    # are excluded from every objective below.
    inactive_self = (
        is_draft
        & ~active_query
        & (kv_indices == S + q_indices)
    )

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (
        (mask_context & active_query) | mask_draft | inactive_self
    ) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    active_block_size: Optional[int] = None,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional within ``active_block_size``.
         The remaining fixed-shape K_max suffix is isolated.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    active_block_size = (
        block_size if active_block_size is None else int(active_block_size)
    )
    if not 1 <= active_block_size <= block_size:
        raise ValueError("active_block_size must be in [1, block_size]")

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        q_depth = q_idx % block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        kv_depth = (kv_idx - S) % block_size
        active_query = q_depth < active_block_size
        mask_draft = (
            is_draft
            & (q_block_id == kv_block_id)
            & active_query
            & (kv_depth < active_block_size)
        )
        inactive_self = (
            is_draft
            & ~active_query
            & (kv_idx == S + q_idx)
        )

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (
            (mask_context & active_query) | mask_draft | inactive_self
        ) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with acceptance-aware objectives."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 0,
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
        selector_loss_alpha: float = 1.0,
        selector_warmup_ratio: float = 0.0,
        selector_ramp_ratio: float = 0.0,
        lk_loss_type: Optional[str] = None,
        kl_scale: float = 1.0,
        kl_decay: float = 1.0,
        dpace_auf_mode: str = "none",
        dpace_auf_suffix_retention: float = 0.0,
        dpace_auf_preserve_weight_mass: bool = False,
        davca_hidden_size: int = 32,
        davca_critic_beta: float = 0.05,
        davca_warmup_ratio: float = 0.10,
        davca_ramp_ratio: float = 0.30,
        cva_continuation_power: float = 1.0,
        cva_residual_min: float = 1.0,
        cva_residual_max: float = 1.2,
        cva_residual_temperature: float = 1.0,
        cva_warmup_ratio: float = 0.10,
        cva_ramp_ratio: float = 0.20,
        cfu_loss_alpha: float = 0.10,
        cfu_frontier_temperature: float = 1.0,
        cfu_loss_temperature: float = 1.0,
        cfu_target_margin: float = 0.0,
        cfu_continuation_power: float = 1.0,
        cfu_utility_transform: str = "linear",
        cfu_warmup_ratio: float = 0.05,
        cfu_ramp_ratio: float = 0.10,
        cfu_advantage_beta: float = 0.25,
        cfu_advantage_temperature: float = 0.5,
        cfu_advantage_min_factor: float = 0.75,
        cfu_advantage_max_factor: float = 1.25,
        cfu_advantage_warmup_ratio: float = 0.30,
        cfu_advantage_ramp_ratio: float = 0.20,
        vcrd_loss_alpha: float = 0.05,
        vcrd_temperature: float = 1.0,
        vcrd_topk: int = 32,
        vcrd_interval: int = 4,
        vcrd_warmup_ratio: float = 0.20,
        vcrd_ramp_ratio: float = 0.20,
        vcrd_min_gain: float = 0.0,
        path_acceptance_alpha: float = 0.0,
        path_acceptance_num_samples: int = 2,
        path_acceptance_topk: int = 32,
        path_acceptance_sampling_temperature: float = 1.0,
        path_acceptance_reward_temperature: float = 1.0,
        path_acceptance_warmup_ratio: float = 0.20,
        path_acceptance_ramp_ratio: float = 0.15,
        acceptance_replay_signals: bool = False,
        prefix_weight_base: float = 0.9,
        survival_loss_alpha: float = 0.0,
        survival_temperature: float = 1.0,
        survival_warmup_ratio: float = 0.05,
        survival_ramp_ratio: float = 0.10,
        survival_hard_alive: bool = True,
        survival_leaky_eta: float = 0.0,
        first_rejection_loss_alpha: float = 0.0,
        first_rejection_margin: float = 0.0,
        first_rejection_temperature: float = 1.0,
        first_rejection_continuation_power: float = 1.0,
        first_rejection_warmup_ratio: float = 0.05,
        first_rejection_ramp_ratio: float = 0.10,
        first_rejection_reduction: str = "normalized",
        first_rejection_mass_match_min: float = 0.0,
        first_rejection_mass_match_max: float = 256.0,
        first_rejection_gradient_mode: str = "none",
        first_rejection_max_depth: int = 0,
        first_rejection_adaptive_survival: bool = False,
        first_rejection_survival_ema_decay: float = 0.99,
        first_rejection_survival_gamma: float = 1.0,
        first_rejection_adaptive_margin_temperature: float = 1.0,
        first_rejection_adaptive_scale_min: float = 0.25,
        first_rejection_adaptive_scale_max: float = 4.0,
        first_rejection_deep_anchor_alpha: float = 0.0,
        first_rejection_deep_anchor_min_depth: int = 4,
        first_rejection_deep_anchor_margin: float = 0.5,
        first_rejection_deep_anchor_temperature: float = 1.0,
        first_rejection_deep_anchor_quantile: float = 0.0,
        first_rejection_deep_anchor_ema_decay: float = 0.99,
        first_rejection_deep_anchor_margin_min: float = 0.0,
        first_rejection_deep_anchor_margin_max: float = 1.5,
        first_rejection_auxiliary_gradient_budget: float = 0.0,
        frbo_survival_alpha: float = 0.0,
        frbo_boundary_alpha: float = 0.0,
        frbo_temperature: float = 1.0,
        frbo_boundary_gamma: float = 2.0,
        frbo_warmup_ratio: float = 0.05,
        frbo_ramp_ratio: float = 0.10,
        frbo_ema_decay: float = 0.99,
        frbo_scale_min: float = 0.1,
        frbo_scale_max: float = 10.0,
        frbo_gradient_mode: str = "none",
        # Markov Scaffold (train-time corrective bias, removed at serving)
        markov_scaffold_rank: int = 0,
        markov_scaffold_ce_alpha: float = 0.0,
        scaffold_distill_alpha: float = 0.0,
        scaffold_temperature: float = 1.0,
        scaffold_on_policy: bool = False,
        scaffold_vp_base: float = 0.7,
        scaffold_vp_min_prefix: int = 1,
        scaffold_vp_warmup_ratio: float = 0.05,
        scaffold_vp_ramp_ratio: float = 0.10,
        scaffold_rescue_only_kd: bool = False,
        scaffold_kd_mode: str = "full",
        scaffold_advantage_global_floor: float = 0.1,
        scaffold_advantage_continuation_power: float = 1.0,
        auxiliary_anneal_start_ratio: float = 1.0,
        auxiliary_anneal_end_ratio: float = 1.0,
        auxiliary_anneal_floor: float = 0.0,
    ):
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")
        if selector_loss_alpha < 0:
            raise ValueError("selector_loss_alpha must be non-negative")
        if not 0.0 <= selector_warmup_ratio <= 1.0:
            raise ValueError("selector_warmup_ratio must be in [0, 1]")
        if not 0.0 <= selector_ramp_ratio <= 1.0:
            raise ValueError("selector_ramp_ratio must be in [0, 1]")
        if (
            getattr(draft_model, "candidate_selector", None) is not None
            and loss_type not in ({"dflash"} | _DPACE_LOSS_TYPES)
        ):
            raise ValueError(
                "DFlash2 candidate selection requires dflash or D-PACE"
            )
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks must be >= 0")
        if lk_loss_type not in _VALID_LK_LOSS_TYPES:
            raise ValueError(
                "lk_loss_type must be one of None, 'alpha', 'lambda', or 'tv'"
            )
        if lk_loss_type is not None and loss_type not in (
            {"dflash"} | _DPACE_LOSS_TYPES
        ):
            raise ValueError(
                "LK objectives currently require dflash or a D-PACE loss type"
            )
        if kl_scale < 0:
            raise ValueError("kl_scale must be non-negative")
        if kl_decay < 0:
            raise ValueError("kl_decay must be non-negative")
        if dpace_auf_mode not in {"none", "hard", "soft"}:
            raise ValueError("dpace_auf_mode must be 'none', 'hard', or 'soft'")
        if not 0.0 <= dpace_auf_suffix_retention <= 1.0:
            raise ValueError("dpace_auf_suffix_retention must be in [0, 1]")
        if dpace_auf_mode == "hard" and dpace_auf_suffix_retention != 0.0:
            raise ValueError("hard D-PACE AUF requires suffix_retention=0")
        if dpace_auf_mode == "soft" and not (
            0.0 < dpace_auf_suffix_retention < 1.0
        ):
            raise ValueError("soft D-PACE AUF requires suffix_retention in (0, 1)")
        if dpace_auf_mode != "none" and loss_type not in (
            _DPACE_LOSS_TYPES | _VCRD_LOSS_TYPES
        ):
            raise ValueError("D-PACE AUF gating requires a D-PACE-based loss type")
        if davca_hidden_size <= 0:
            raise ValueError("davca_hidden_size must be positive")
        if davca_critic_beta < 0:
            raise ValueError("davca_critic_beta must be non-negative")
        if not 0.0 <= davca_warmup_ratio <= 1.0:
            raise ValueError("davca_warmup_ratio must be in [0, 1]")
        if not 0.0 <= davca_ramp_ratio <= 1.0:
            raise ValueError("davca_ramp_ratio must be in [0, 1]")
        if davca_warmup_ratio + davca_ramp_ratio > 1.0:
            raise ValueError("DAVCA warmup + ramp ratios must be <= 1")
        if cva_continuation_power < 0:
            raise ValueError("cva_continuation_power must be non-negative")
        if cva_residual_min <= 0:
            raise ValueError("cva_residual_min must be positive")
        if cva_residual_max < cva_residual_min:
            raise ValueError("cva_residual_max must be >= cva_residual_min")
        if cva_residual_temperature <= 0:
            raise ValueError("cva_residual_temperature must be positive")
        if not 0.0 <= cva_warmup_ratio <= 1.0:
            raise ValueError("cva_warmup_ratio must be in [0, 1]")
        if not 0.0 <= cva_ramp_ratio <= 1.0:
            raise ValueError("cva_ramp_ratio must be in [0, 1]")
        if cva_warmup_ratio + cva_ramp_ratio > 1.0:
            raise ValueError("CVA warmup + ramp ratios must be <= 1")
        if cfu_loss_alpha < 0:
            raise ValueError("cfu_loss_alpha must be non-negative")
        if cfu_frontier_temperature <= 0:
            raise ValueError("cfu_frontier_temperature must be positive")
        if cfu_loss_temperature <= 0:
            raise ValueError("cfu_loss_temperature must be positive")
        if cfu_continuation_power < 0:
            raise ValueError("cfu_continuation_power must be non-negative")
        if cfu_utility_transform not in {"linear", "log1p"}:
            raise ValueError(
                "cfu_utility_transform must be 'linear' or 'log1p'"
            )
        if not 0.0 <= cfu_warmup_ratio <= 1.0:
            raise ValueError("cfu_warmup_ratio must be in [0, 1]")
        if not 0.0 <= cfu_ramp_ratio <= 1.0:
            raise ValueError("cfu_ramp_ratio must be in [0, 1]")
        if cfu_warmup_ratio + cfu_ramp_ratio > 1.0:
            raise ValueError("CFU warmup + ramp ratios must be <= 1")
        if cfu_advantage_beta < 0:
            raise ValueError("cfu_advantage_beta must be non-negative")
        if cfu_advantage_temperature <= 0:
            raise ValueError("cfu_advantage_temperature must be positive")
        if cfu_advantage_min_factor <= 0:
            raise ValueError("cfu_advantage_min_factor must be positive")
        if cfu_advantage_max_factor < cfu_advantage_min_factor:
            raise ValueError(
                "cfu_advantage_max_factor must be >= "
                "cfu_advantage_min_factor"
            )
        if not 0.0 <= cfu_advantage_warmup_ratio <= 1.0:
            raise ValueError(
                "cfu_advantage_warmup_ratio must be in [0, 1]"
            )
        if not 0.0 <= cfu_advantage_ramp_ratio <= 1.0:
            raise ValueError(
                "cfu_advantage_ramp_ratio must be in [0, 1]"
            )
        if (
            cfu_advantage_warmup_ratio + cfu_advantage_ramp_ratio > 1.0
        ):
            raise ValueError(
                "CFU advantage warmup + ramp ratios must be <= 1"
            )
        if vcrd_loss_alpha < 0:
            raise ValueError("vcrd_loss_alpha must be non-negative")
        if vcrd_temperature <= 0:
            raise ValueError("vcrd_temperature must be positive")
        if vcrd_topk <= 0:
            raise ValueError("vcrd_topk must be positive")
        if vcrd_interval <= 0:
            raise ValueError("vcrd_interval must be positive")
        if not 0.0 <= vcrd_warmup_ratio <= 1.0:
            raise ValueError("vcrd_warmup_ratio must be in [0, 1]")
        if not 0.0 <= vcrd_ramp_ratio <= 1.0:
            raise ValueError("vcrd_ramp_ratio must be in [0, 1]")
        if vcrd_warmup_ratio + vcrd_ramp_ratio > 1.0:
            raise ValueError("VCRD warmup + ramp ratios must be <= 1")
        if vcrd_min_gain < 0:
            raise ValueError("vcrd_min_gain must be non-negative")
        if path_acceptance_alpha < 0:
            raise ValueError("path_acceptance_alpha must be non-negative")
        if path_acceptance_num_samples < 2:
            raise ValueError("path_acceptance_num_samples must be at least 2")
        if path_acceptance_topk <= 0:
            raise ValueError("path_acceptance_topk must be positive")
        if path_acceptance_sampling_temperature <= 0:
            raise ValueError(
                "path_acceptance_sampling_temperature must be positive"
            )
        if path_acceptance_reward_temperature <= 0:
            raise ValueError(
                "path_acceptance_reward_temperature must be positive"
            )
        if not 0.0 <= path_acceptance_warmup_ratio <= 1.0:
            raise ValueError("path_acceptance_warmup_ratio must be in [0, 1]")
        if not 0.0 <= path_acceptance_ramp_ratio <= 1.0:
            raise ValueError("path_acceptance_ramp_ratio must be in [0, 1]")
        if path_acceptance_warmup_ratio + path_acceptance_ramp_ratio > 1.0:
            raise ValueError("path-acceptance warmup + ramp ratios must be <= 1")
        if path_acceptance_alpha > 0 and loss_type not in (
            _DPACE_LOSS_TYPES | _VCRD_LOSS_TYPES
        ):
            raise ValueError(
                "path acceptance requires a D-PACE-based primary loss"
            )
        if prefix_weight_base is None:
            prefix_weight_base = 0.9
        if prefix_weight_base <= 0.0:
            raise ValueError(
                f"prefix_weight_base must be positive, got {prefix_weight_base}"
            )
        if survival_loss_alpha < 0:
            raise ValueError(
                f"survival_loss_alpha must be non-negative, got {survival_loss_alpha}"
            )
        if survival_temperature <= 0:
            raise ValueError(
                f"survival_temperature must be positive, got {survival_temperature}"
            )
        if not 0.0 <= survival_leaky_eta <= 1.0:
            raise ValueError(
                f"survival_leaky_eta must be in [0, 1], got {survival_leaky_eta}"
            )
        if survival_leaky_eta > 0 and not survival_hard_alive:
            raise ValueError(
                "survival_leaky_eta > 0 requires survival_hard_alive=True"
            )
        if first_rejection_loss_alpha < 0:
            raise ValueError(
                "first_rejection_loss_alpha must be non-negative, "
                f"got {first_rejection_loss_alpha}"
            )
        if first_rejection_temperature <= 0:
            raise ValueError(
                "first_rejection_temperature must be positive, "
                f"got {first_rejection_temperature}"
            )
        if first_rejection_continuation_power < 0:
            raise ValueError(
                "first_rejection_continuation_power must be non-negative, "
                f"got {first_rejection_continuation_power}"
            )
        if not 0.0 <= first_rejection_warmup_ratio <= 1.0:
            raise ValueError(
                "first_rejection_warmup_ratio must be in [0, 1], "
                f"got {first_rejection_warmup_ratio}"
            )
        if not 0.0 <= first_rejection_ramp_ratio <= 1.0:
            raise ValueError(
                "first_rejection_ramp_ratio must be in [0, 1], "
                f"got {first_rejection_ramp_ratio}"
            )
        if first_rejection_warmup_ratio + first_rejection_ramp_ratio > 1.0:
            raise ValueError(
                "first_rejection warmup + ramp ratios must be <= 1"
            )
        if first_rejection_reduction not in {
            "normalized",
            "batch",
            "dpace-matched",
        }:
            raise ValueError(
                "first_rejection_reduction must be normalized, batch, or "
                f"dpace-matched; got {first_rejection_reduction!r}"
            )
        if (
            first_rejection_reduction == "dpace-matched"
            and loss_type not in _DPACE_LOSS_TYPES | _VCRD_LOSS_TYPES
        ):
            raise ValueError(
                "first_rejection_reduction='dpace-matched' requires a "
                "D-PACE-based loss type"
            )
        if (
            first_rejection_mass_match_min < 0
            or first_rejection_mass_match_max < first_rejection_mass_match_min
        ):
            raise ValueError(
                "first-rejection mass-match bounds must satisfy 0 <= min <= max"
            )
        if first_rejection_gradient_mode not in {"none", "margin-pcgrad"}:
            raise ValueError(
                "first_rejection_gradient_mode must be 'none' or "
                "'margin-pcgrad'"
            )
        if first_rejection_max_depth < 0:
            raise ValueError("first_rejection_max_depth must be non-negative")
        if not 0.0 <= first_rejection_survival_ema_decay < 1.0:
            raise ValueError(
                "first_rejection_survival_ema_decay must be in [0, 1)"
            )
        if first_rejection_survival_gamma < 0:
            raise ValueError("first_rejection_survival_gamma must be non-negative")
        if first_rejection_adaptive_margin_temperature <= 0:
            raise ValueError(
                "first_rejection_adaptive_margin_temperature must be positive"
            )
        if (
            first_rejection_adaptive_scale_min <= 0
            or first_rejection_adaptive_scale_max
            < first_rejection_adaptive_scale_min
        ):
            raise ValueError(
                "first-rejection adaptive scales must satisfy 0 < min <= max"
            )
        if first_rejection_deep_anchor_alpha < 0:
            raise ValueError(
                "first_rejection_deep_anchor_alpha must be non-negative"
            )
        if first_rejection_deep_anchor_min_depth <= 0:
            raise ValueError(
                "first_rejection_deep_anchor_min_depth must be positive"
            )
        if first_rejection_deep_anchor_temperature <= 0:
            raise ValueError(
                "first_rejection_deep_anchor_temperature must be positive"
            )
        if not 0.0 <= first_rejection_deep_anchor_quantile < 1.0:
            raise ValueError(
                "first_rejection_deep_anchor_quantile must be in [0, 1)"
            )
        if not 0.0 <= first_rejection_deep_anchor_ema_decay < 1.0:
            raise ValueError(
                "first_rejection_deep_anchor_ema_decay must be in [0, 1)"
            )
        if (
            first_rejection_deep_anchor_margin_max
            < first_rejection_deep_anchor_margin_min
        ):
            raise ValueError(
                "deep-anchor margin bounds must satisfy min <= max"
            )
        if first_rejection_auxiliary_gradient_budget < 0:
            raise ValueError(
                "first_rejection_auxiliary_gradient_budget must be non-negative"
            )
        if frbo_survival_alpha < 0 or frbo_boundary_alpha < 0:
            raise ValueError("FRBO loss weights must be non-negative")
        if frbo_temperature <= 0:
            raise ValueError("frbo_temperature must be positive")
        if frbo_boundary_gamma < 0:
            raise ValueError("frbo_boundary_gamma must be non-negative")
        if not 0.0 <= frbo_warmup_ratio <= 1.0:
            raise ValueError("frbo_warmup_ratio must be in [0, 1]")
        if not 0.0 <= frbo_ramp_ratio <= 1.0:
            raise ValueError("frbo_ramp_ratio must be in [0, 1]")
        if frbo_warmup_ratio + frbo_ramp_ratio > 1.0:
            raise ValueError("FRBO warmup + ramp ratios must be <= 1")
        if not 0.0 <= frbo_ema_decay < 1.0:
            raise ValueError("frbo_ema_decay must be in [0, 1)")
        if frbo_scale_min <= 0 or frbo_scale_max < frbo_scale_min:
            raise ValueError("FRBO scales must satisfy 0 < min <= max")
        if frbo_gradient_mode not in {"none", "margin-pcgrad"}:
            raise ValueError(
                "frbo_gradient_mode must be 'none' or 'margin-pcgrad'"
            )

        # --- Markov Scaffold validation ---
        if markov_scaffold_ce_alpha < 0:
            raise ValueError(
                f"markov_scaffold_ce_alpha must be non-negative, "
                f"got {markov_scaffold_ce_alpha}"
            )
        if scaffold_distill_alpha < 0:
            raise ValueError(
                f"scaffold_distill_alpha must be non-negative, "
                f"got {scaffold_distill_alpha}"
            )
        if scaffold_temperature <= 0:
            raise ValueError(
                f"scaffold_temperature must be positive, "
                f"got {scaffold_temperature}"
            )
        if not 0.0 < scaffold_vp_base < 1.0:
            raise ValueError(
                f"scaffold_vp_base must be in (0, 1), got {scaffold_vp_base}"
            )
        if scaffold_vp_min_prefix < 0:
            raise ValueError(
                f"scaffold_vp_min_prefix must be non-negative, "
                f"got {scaffold_vp_min_prefix}"
            )
        _scaffold_wants_params = (
            markov_scaffold_ce_alpha > 0 or scaffold_distill_alpha > 0
        )
        if _scaffold_wants_params and markov_scaffold_rank <= 0:
            raise ValueError(
                "Markov Scaffold losses require markov_scaffold_rank > 0"
            )
        if scaffold_on_policy and markov_scaffold_rank <= 0:
            raise ValueError(
                "scaffold_on_policy requires markov_scaffold_rank > 0"
            )
        if scaffold_rescue_only_kd and scaffold_distill_alpha <= 0:
            raise ValueError(
                "scaffold_rescue_only_kd requires scaffold_distill_alpha > 0"
            )
        valid_scaffold_kd_modes = {
            "full",
            "rescue",
            "advantage",
            "first-rejection-advantage",
        }
        if scaffold_kd_mode not in valid_scaffold_kd_modes:
            raise ValueError(
                f"scaffold_kd_mode={scaffold_kd_mode!r}; must be one of "
                f"{sorted(valid_scaffold_kd_modes)}"
            )
        if scaffold_rescue_only_kd and scaffold_kd_mode != "full":
            raise ValueError(
                "Use either scaffold_rescue_only_kd (legacy) or an explicit "
                "scaffold_kd_mode, not both"
            )
        if scaffold_advantage_global_floor < 0:
            raise ValueError("scaffold_advantage_global_floor must be >= 0")
        if scaffold_advantage_continuation_power < 0:
            raise ValueError(
                "scaffold_advantage_continuation_power must be >= 0"
            )
        if not 0.0 <= auxiliary_anneal_start_ratio <= 1.0:
            raise ValueError("auxiliary_anneal_start_ratio must be in [0, 1]")
        if not 0.0 <= auxiliary_anneal_end_ratio <= 1.0:
            raise ValueError("auxiliary_anneal_end_ratio must be in [0, 1]")
        if auxiliary_anneal_end_ratio < auxiliary_anneal_start_ratio:
            raise ValueError(
                "auxiliary_anneal_end_ratio must be >= "
                "auxiliary_anneal_start_ratio"
            )
        if not 0.0 <= auxiliary_anneal_floor <= 1.0:
            raise ValueError("auxiliary_anneal_floor must be in [0, 1]")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.loss_type = loss_type
        self.dpace_alpha = dpace_alpha
        self.selector_loss_alpha = float(selector_loss_alpha)
        self.selector_warmup_ratio = float(selector_warmup_ratio)
        self.selector_ramp_ratio = float(selector_ramp_ratio)
        self.lk_loss_type = lk_loss_type
        self.kl_scale = float(kl_scale)
        self.kl_decay = float(kl_decay)
        self.dpace_auf_mode = dpace_auf_mode
        self.dpace_auf_suffix_retention = float(dpace_auf_suffix_retention)
        self.dpace_auf_preserve_weight_mass = bool(
            dpace_auf_preserve_weight_mass
        )
        self.davca_hidden_size = int(davca_hidden_size)
        self.davca_critic_beta = float(davca_critic_beta)
        self.davca_warmup_ratio = float(davca_warmup_ratio)
        self.davca_ramp_ratio = float(davca_ramp_ratio)
        self.cva_continuation_power = float(cva_continuation_power)
        self.cva_residual_min = float(cva_residual_min)
        self.cva_residual_max = float(cva_residual_max)
        self.cva_residual_temperature = float(cva_residual_temperature)
        self.cva_warmup_ratio = float(cva_warmup_ratio)
        self.cva_ramp_ratio = float(cva_ramp_ratio)
        self.cfu_loss_alpha = float(cfu_loss_alpha)
        self.cfu_frontier_temperature = float(cfu_frontier_temperature)
        self.cfu_loss_temperature = float(cfu_loss_temperature)
        self.cfu_target_margin = float(cfu_target_margin)
        self.cfu_continuation_power = float(cfu_continuation_power)
        self.cfu_utility_transform = cfu_utility_transform
        self.cfu_warmup_ratio = float(cfu_warmup_ratio)
        self.cfu_ramp_ratio = float(cfu_ramp_ratio)
        self.cfu_advantage_beta = float(cfu_advantage_beta)
        self.cfu_advantage_temperature = float(cfu_advantage_temperature)
        self.cfu_advantage_min_factor = float(cfu_advantage_min_factor)
        self.cfu_advantage_max_factor = float(cfu_advantage_max_factor)
        self.cfu_advantage_warmup_ratio = float(
            cfu_advantage_warmup_ratio
        )
        self.cfu_advantage_ramp_ratio = float(cfu_advantage_ramp_ratio)
        self.vcrd_loss_alpha = float(vcrd_loss_alpha)
        self.vcrd_temperature = float(vcrd_temperature)
        self.vcrd_topk = int(vcrd_topk)
        self.vcrd_interval = int(vcrd_interval)
        self.vcrd_warmup_ratio = float(vcrd_warmup_ratio)
        self.vcrd_ramp_ratio = float(vcrd_ramp_ratio)
        self.vcrd_min_gain = float(vcrd_min_gain)
        self.path_acceptance_alpha = float(path_acceptance_alpha)
        self.path_acceptance_num_samples = int(path_acceptance_num_samples)
        self.path_acceptance_topk = int(path_acceptance_topk)
        self.path_acceptance_sampling_temperature = float(
            path_acceptance_sampling_temperature
        )
        self.path_acceptance_reward_temperature = float(
            path_acceptance_reward_temperature
        )
        self.path_acceptance_warmup_ratio = float(
            path_acceptance_warmup_ratio
        )
        self.path_acceptance_ramp_ratio = float(path_acceptance_ramp_ratio)
        self.acceptance_replay_signals = bool(acceptance_replay_signals)
        self._last_acceptance_replay_signals: Optional[torch.Tensor] = None
        self.prefix_weight_base = prefix_weight_base

        self.verification_critic: Optional[VerificationSurvivalCritic] = None
        if loss_type in _DAVCA_LOSS_TYPES:
            critic_mode = "scalar" if loss_type == "davca-scalar" else "dependency"
            self.verification_critic = VerificationSurvivalCritic(
                feature_size=6,
                hidden_size=self.davca_hidden_size,
                mode=critic_mode,
            )
            # OnlineDFlashModel is constructed after the draft has already been
            # placed on its training device/dtype.
            reference = next(target_embed_tokens.parameters())
            self.verification_critic.to(
                device=reference.device,
                dtype=reference.dtype,
            )

        self.survival_loss_alpha = float(survival_loss_alpha)
        self.survival_temperature = float(survival_temperature)
        self.survival_warmup_ratio = float(survival_warmup_ratio)
        self.survival_ramp_ratio = float(survival_ramp_ratio)
        self.survival_hard_alive = survival_hard_alive
        self.survival_leaky_eta = float(survival_leaky_eta)
        self.first_rejection_loss_alpha = float(first_rejection_loss_alpha)
        self.first_rejection_margin = float(first_rejection_margin)
        self.first_rejection_temperature = float(first_rejection_temperature)
        self.first_rejection_continuation_power = float(
            first_rejection_continuation_power
        )
        self.first_rejection_warmup_ratio = float(first_rejection_warmup_ratio)
        self.first_rejection_ramp_ratio = float(first_rejection_ramp_ratio)
        self.first_rejection_reduction = first_rejection_reduction
        self.first_rejection_mass_match_min = float(
            first_rejection_mass_match_min
        )
        self.first_rejection_mass_match_max = float(
            first_rejection_mass_match_max
        )
        self.first_rejection_gradient_mode = first_rejection_gradient_mode
        self.first_rejection_max_depth = int(first_rejection_max_depth)
        self.first_rejection_adaptive_survival = bool(
            first_rejection_adaptive_survival
        )
        self.first_rejection_survival_ema_decay = float(
            first_rejection_survival_ema_decay
        )
        self.first_rejection_survival_gamma = float(
            first_rejection_survival_gamma
        )
        self.first_rejection_adaptive_margin_temperature = float(
            first_rejection_adaptive_margin_temperature
        )
        self.first_rejection_adaptive_scale_min = float(
            first_rejection_adaptive_scale_min
        )
        self.first_rejection_adaptive_scale_max = float(
            first_rejection_adaptive_scale_max
        )
        self.first_rejection_deep_anchor_alpha = float(
            first_rejection_deep_anchor_alpha
        )
        self.first_rejection_deep_anchor_min_depth = int(
            first_rejection_deep_anchor_min_depth
        )
        self.first_rejection_deep_anchor_margin = float(
            first_rejection_deep_anchor_margin
        )
        self.first_rejection_deep_anchor_temperature = float(
            first_rejection_deep_anchor_temperature
        )
        self.first_rejection_deep_anchor_quantile = float(
            first_rejection_deep_anchor_quantile
        )
        self.first_rejection_deep_anchor_ema_decay = float(
            first_rejection_deep_anchor_ema_decay
        )
        self.first_rejection_deep_anchor_margin_min = float(
            first_rejection_deep_anchor_margin_min
        )
        self.first_rejection_deep_anchor_margin_max = float(
            first_rejection_deep_anchor_margin_max
        )
        self.first_rejection_auxiliary_gradient_budget = float(
            first_rejection_auxiliary_gradient_budget
        )
        self.register_buffer(
            "first_rejection_survival_ema", torch.ones(self.block_size)
        )
        self.register_buffer(
            "first_rejection_survival_ema_updates",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "deep_anchor_margin_ema",
            torch.full(
                (self.block_size,), self.first_rejection_deep_anchor_margin
            ),
        )
        self.register_buffer(
            "deep_anchor_margin_ema_updates",
            torch.tensor(0, dtype=torch.long),
        )
        self.frbo_survival_alpha = float(frbo_survival_alpha)
        self.frbo_boundary_alpha = float(frbo_boundary_alpha)
        self.frbo_temperature = float(frbo_temperature)
        self.frbo_boundary_gamma = float(frbo_boundary_gamma)
        self.frbo_warmup_ratio = float(frbo_warmup_ratio)
        self.frbo_ramp_ratio = float(frbo_ramp_ratio)
        self.frbo_ema_decay = float(frbo_ema_decay)
        self.frbo_scale_min = float(frbo_scale_min)
        self.frbo_scale_max = float(frbo_scale_max)
        self.frbo_gradient_mode = frbo_gradient_mode
        self.register_buffer("frbo_primary_ema", torch.tensor(0.0))
        self.register_buffer("frbo_survival_ema", torch.tensor(0.0))
        self.register_buffer("frbo_boundary_ema", torch.tensor(0.0))
        self.register_buffer(
            "frbo_ema_updates", torch.tensor(0, dtype=torch.long)
        )

        # Markov Scaffold attributes
        self.markov_scaffold_rank = markov_scaffold_rank
        self.markov_scaffold_ce_alpha = float(markov_scaffold_ce_alpha)
        self.scaffold_distill_alpha = float(scaffold_distill_alpha)
        self.scaffold_temperature = float(scaffold_temperature)
        self.scaffold_on_policy = scaffold_on_policy
        self.scaffold_vp_base = float(scaffold_vp_base)
        self.scaffold_vp_min_prefix = int(scaffold_vp_min_prefix)
        self.scaffold_vp_warmup_ratio = float(scaffold_vp_warmup_ratio)
        self.scaffold_vp_ramp_ratio = float(scaffold_vp_ramp_ratio)
        self.scaffold_rescue_only_kd = scaffold_rescue_only_kd
        self.scaffold_kd_mode = (
            "rescue" if scaffold_rescue_only_kd else scaffold_kd_mode
        )
        self.scaffold_advantage_global_floor = float(
            scaffold_advantage_global_floor
        )
        self.scaffold_advantage_continuation_power = float(
            scaffold_advantage_continuation_power
        )
        self.auxiliary_anneal_start_ratio = float(auxiliary_anneal_start_ratio)
        self.auxiliary_anneal_end_ratio = float(auxiliary_anneal_end_ratio)
        self.auxiliary_anneal_floor = float(auxiliary_anneal_floor)

        # Training progress tracking (set by training script via
        # set_training_progress).
        self._global_step: int = 0
        self._total_steps: int = 0

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def set_training_progress(self, global_step: int, total_steps: int) -> None:
        """Update training progress for survival loss warmup/ramp scheduling.

        Both ``global_step`` and ``total_steps`` should use the same unit —
        optimizer steps (i.e. micro-batch count divided by
        ``accumulation_steps``).  The caller is responsible for converting
        if their internal counter tracks micro-batches.
        """
        self._global_step = global_step
        self._total_steps = total_steps

    def _effective_selector_loss_alpha(self) -> float:
        """Return the scheduled DFlash2 selector weight.

        The base unary model can optionally train alone before the selector is
        enabled, after which the selector weight is linearly ramped.  This
        mirrors the latest upstream DFlash2 training strategy.
        """
        target = self.selector_loss_alpha
        if target <= 0 or self._total_steps <= 0:
            return target
        warmup_steps = int(self._total_steps * self.selector_warmup_ratio)
        if self._global_step < warmup_steps:
            return 0.0
        ramp_steps = int(self._total_steps * self.selector_ramp_ratio)
        if ramp_steps <= 0:
            return target
        ramp_progress = min(
            max((self._global_step - warmup_steps + 1) / ramp_steps, 0.0),
            1.0,
        )
        return target * ramp_progress

    def _effective_davca_blend(self) -> float:
        """Blend from the D-PACE baseline to learned DAVCA weights."""
        if self.loss_type not in _DAVCA_LOSS_TYPES or self._total_steps <= 0:
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.davca_warmup_ratio:
            return 0.0
        if self.davca_ramp_ratio > 0:
            ramp_end = self.davca_warmup_ratio + self.davca_ramp_ratio
            if progress < ramp_end:
                return (progress - self.davca_warmup_ratio) / self.davca_ramp_ratio
        return 1.0

    def _effective_cva_blend(self) -> float:
        """Blend from D-PACE to counterfactual verification credit."""
        if self.loss_type not in _CVA_LOSS_TYPES or self._total_steps <= 0:
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.cva_warmup_ratio:
            return 0.0
        if self.cva_ramp_ratio > 0:
            ramp_end = self.cva_warmup_ratio + self.cva_ramp_ratio
            if progress < ramp_end:
                return (progress - self.cva_warmup_ratio) / self.cva_ramp_ratio
        return 1.0

    def _effective_cfu_alpha(self) -> float:
        """Warm up the independent counterfactual-frontier auxiliary loss."""
        if (
            self.loss_type not in _CFU_LOSS_TYPES
            or self.cfu_loss_alpha <= 0
            or self._total_steps <= 0
        ):
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.cfu_warmup_ratio:
            return 0.0
        if self.cfu_ramp_ratio > 0:
            ramp_end = self.cfu_warmup_ratio + self.cfu_ramp_ratio
            if progress < ramp_end:
                ramp = (progress - self.cfu_warmup_ratio) / self.cfu_ramp_ratio
                return self.cfu_loss_alpha * ramp
        return self.cfu_loss_alpha

    def _effective_cfu_advantage_beta(self) -> float:
        """Delay observed CFU correction until the smooth baseline is stable."""
        if (
            self.loss_type != "cfu-advantage"
            or self.cfu_advantage_beta <= 0
            or self._total_steps <= 0
        ):
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.cfu_advantage_warmup_ratio:
            return 0.0
        if self.cfu_advantage_ramp_ratio > 0:
            ramp_end = (
                self.cfu_advantage_warmup_ratio
                + self.cfu_advantage_ramp_ratio
            )
            if progress < ramp_end:
                ramp = (
                    progress - self.cfu_advantage_warmup_ratio
                ) / self.cfu_advantage_ramp_ratio
                return self.cfu_advantage_beta * ramp
        return self.cfu_advantage_beta

    def _effective_vcrd_alpha(self) -> float:
        """Return the scheduled VCRD coefficient; zero means exact D-PACE."""
        if (
            self.loss_type not in _VCRD_LOSS_TYPES
            or self.vcrd_loss_alpha <= 0
            or self._total_steps <= 0
        ):
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.vcrd_warmup_ratio:
            return 0.0
        if self.vcrd_ramp_ratio > 0:
            ramp_end = self.vcrd_warmup_ratio + self.vcrd_ramp_ratio
            if progress < ramp_end:
                ramp = (
                    progress - self.vcrd_warmup_ratio
                ) / self.vcrd_ramp_ratio
                return self.vcrd_loss_alpha * ramp
        return self.vcrd_loss_alpha

    def _should_run_vcrd(self, effective_alpha: float) -> bool:
        """Use optimizer-step sparsity so rollout overhead is predictable."""
        return (
            effective_alpha > 0
            and self._global_step > 0
            and self._global_step % self.vcrd_interval == 0
        )

    def _effective_path_acceptance_alpha(self) -> float:
        """Warm up sampled path ranking after token calibration stabilizes."""
        if self.path_acceptance_alpha <= 0 or self._total_steps <= 0:
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.path_acceptance_warmup_ratio:
            return 0.0
        ramp_end = (
            self.path_acceptance_warmup_ratio
            + self.path_acceptance_ramp_ratio
        )
        if self.path_acceptance_ramp_ratio > 0 and progress < ramp_end:
            ramp = (
                progress - self.path_acceptance_warmup_ratio
            ) / self.path_acceptance_ramp_ratio
            return self.path_acceptance_alpha * ramp
        return self.path_acceptance_alpha

    @staticmethod
    def _empty_vcrd_diagnostics(
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Return a stable logging schema on steps without a repair rollout."""
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {
            "vcrd_loss": zero,
            "vcrd_rollout_active": zero,
            "vcrd_candidate_fraction": zero,
            "vcrd_selected_depth": zero,
            "vcrd_frontier_score": zero,
            "vcrd_base_continuation": zero,
            "vcrd_teacher_continuation": zero,
            "vcrd_raw_gain": zero,
            "vcrd_positive_gain_fraction": zero,
            "vcrd_distill_token_fraction": zero,
            "vcrd_credit_mass": zero,
            "vcrd_teacher_topk_mass": zero,
        }

    def _effective_survival_alpha(self) -> float:
        """Compute effective survival loss alpha with ratio-based warmup and ramp.

        Schedule (ratio-based, portable across dataset/batch sizes):

        - First ``survival_warmup_ratio`` of total_steps: alpha = 0
        - Next ``survival_ramp_ratio`` of total_steps: linear ramp to target
        - After: target ``survival_loss_alpha``
        """
        if self.survival_loss_alpha <= 0:
            return 0.0
        if self._total_steps <= 0:
            return 0.0

        progress = self._global_step / self._total_steps
        warmup_end = self.survival_warmup_ratio
        ramp_end = warmup_end + self.survival_ramp_ratio

        if progress < warmup_end:
            return 0.0
        elif progress < ramp_end:
            ramp_progress = (progress - warmup_end) / self.survival_ramp_ratio
            alpha = self.survival_loss_alpha * ramp_progress
        else:
            alpha = self.survival_loss_alpha
        return alpha * self._auxiliary_anneal_factor()

    def _effective_first_rejection_alpha(self) -> float:
        """Warm up and ramp First-Rejection Continuation Loss independently."""
        if self.first_rejection_loss_alpha <= 0 or self._total_steps <= 0:
            return 0.0

        progress = self._global_step / self._total_steps
        warmup_end = self.first_rejection_warmup_ratio
        ramp_end = warmup_end + self.first_rejection_ramp_ratio
        if progress < warmup_end:
            return 0.0
        if self.first_rejection_ramp_ratio > 0 and progress < ramp_end:
            ramp_progress = (
                progress - warmup_end
            ) / self.first_rejection_ramp_ratio
            return self.first_rejection_loss_alpha * ramp_progress
        return self.first_rejection_loss_alpha

    def _effective_deep_anchor_alpha(self) -> float:
        """Warm up DFAP independently, using the shared acceptance schedule."""
        if self.first_rejection_deep_anchor_alpha <= 0 or self._total_steps <= 0:
            return 0.0

        progress = self._global_step / self._total_steps
        warmup_end = self.first_rejection_warmup_ratio
        ramp_end = warmup_end + self.first_rejection_ramp_ratio
        if progress < warmup_end:
            return 0.0
        if self.first_rejection_ramp_ratio > 0 and progress < ramp_end:
            ramp_progress = (
                progress - warmup_end
            ) / self.first_rejection_ramp_ratio
            return self.first_rejection_deep_anchor_alpha * ramp_progress
        return self.first_rejection_deep_anchor_alpha

    @torch.no_grad()
    def _update_first_rejection_survival_ema(
        self,
        hard_accept: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Update per-depth hard chain-survival rates for adaptive FRC."""
        valid = valid_mask.bool()
        accepted_or_invalid = hard_accept.bool() | (~valid)
        survived = torch.cumprod(
            accepted_or_invalid.to(torch.int32), dim=-1
        ).bool()
        counts = valid.float().sum(dim=(0, 1))
        batch_rate = (
            (survived & valid).float().sum(dim=(0, 1))
            / counts.clamp_min(1.0)
        )
        observed = counts.gt(0)
        if self.first_rejection_survival_ema_updates.item() == 0:
            updated = batch_rate
        else:
            decay = self.first_rejection_survival_ema_decay
            updated = (
                decay * self.first_rejection_survival_ema
                + (1.0 - decay) * batch_rate
            )
        self.first_rejection_survival_ema.copy_(
            torch.where(observed, updated, self.first_rejection_survival_ema)
        )
        self.first_rejection_survival_ema_updates.add_(1)
        return batch_rate

    @torch.no_grad()
    def _update_deep_anchor_margin_ema(
        self,
        *,
        target_margin: torch.Tensor,
        valid_mask: torch.Tensor,
        hard_accept: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update vulnerable full-chain margin floors from a batch quantile."""
        from specforge.core.acceptance_losses import (
            deep_full_accept_margin_quantiles,
        )

        batch_floor, counts = deep_full_accept_margin_quantiles(
            target_margin=target_margin,
            valid_mask=valid_mask,
            hard_accept=hard_accept,
            min_depth=self.first_rejection_deep_anchor_min_depth,
            quantile=self.first_rejection_deep_anchor_quantile,
            fallback=self.deep_anchor_margin_ema,
        )
        batch_floor = batch_floor.clamp(
            min=self.first_rejection_deep_anchor_margin_min,
            max=self.first_rejection_deep_anchor_margin_max,
        )
        observed = counts.gt(0)
        if self.deep_anchor_margin_ema_updates.item() == 0:
            updated = batch_floor
        else:
            decay = self.first_rejection_deep_anchor_ema_decay
            updated = (
                decay * self.deep_anchor_margin_ema
                + (1.0 - decay) * batch_floor
            )
        self.deep_anchor_margin_ema.copy_(
            torch.where(observed, updated, self.deep_anchor_margin_ema)
        )
        self.deep_anchor_margin_ema_updates.add_(1)
        return batch_floor, counts

    def _effective_frbo_ramp(self) -> float:
        """Return the shared FRBO warmup/ramp multiplier."""
        if self._total_steps <= 0:
            return 0.0
        progress = self._global_step / self._total_steps
        if progress < self.frbo_warmup_ratio:
            return 0.0
        ramp_end = self.frbo_warmup_ratio + self.frbo_ramp_ratio
        if self.frbo_ramp_ratio > 0 and progress < ramp_end:
            return (progress - self.frbo_warmup_ratio) / self.frbo_ramp_ratio
        return 1.0

    # ------------------------------------------------------------------
    # Markov Scaffold helpers
    # ------------------------------------------------------------------

    def _auxiliary_anneal_factor(self) -> float:
        """Return the shared late-training multiplier for auxiliary losses.

        ``start=end=1`` preserves the historical fixed-weight behavior.  With
        ``start < end``, the factor is one through ``start``, decays linearly
        to ``auxiliary_anneal_floor``, and stays at that floor from ``end``
        onward.  The default floor of zero preserves the original schedule.
        """
        if self._total_steps <= 0:
            return 1.0
        start = self.auxiliary_anneal_start_ratio
        end = self.auxiliary_anneal_end_ratio
        floor = self.auxiliary_anneal_floor
        if start >= 1.0:
            return 1.0
        progress = self._global_step / self._total_steps
        if end == start:
            return 1.0 if progress < start else floor
        if progress <= start:
            return 1.0
        if progress >= end:
            return floor
        decay_progress = (progress - start) / (end - start)
        return 1.0 - decay_progress * (1.0 - floor)

    def _effective_scaffold_ce_alpha(self) -> float:
        return self.markov_scaffold_ce_alpha * self._auxiliary_anneal_factor()

    def _effective_scaffold_kd_alpha(self) -> float:
        """Compute effective scaffold KD alpha with warmup and ramp.

        Schedule (ratio-based, same unit as survival):
        - First ``scaffold_vp_warmup_ratio`` of total_steps: alpha = 0
        - Next ``scaffold_vp_ramp_ratio``: linear ramp to target
        - After: target ``scaffold_distill_alpha``
        """
        if self.scaffold_distill_alpha <= 0:
            return 0.0
        if self._total_steps <= 0:
            return 0.0

        progress = self._global_step / self._total_steps
        warmup_end = self.scaffold_vp_warmup_ratio
        ramp_end = warmup_end + self.scaffold_vp_ramp_ratio

        if progress < warmup_end:
            return 0.0
        elif progress < ramp_end:
            ramp_progress = (progress - warmup_end) / self.scaffold_vp_ramp_ratio
            alpha = self.scaffold_distill_alpha * ramp_progress
        else:
            alpha = self.scaffold_distill_alpha
        return alpha * self._auxiliary_anneal_factor()

    def _effective_scaffold_vp_exposure(self) -> float:
        """Return the fraction of blocks exposed to on-policy predecessors.

        The schedule deliberately mirrors scaffold KD:

        - warmup: every block is fully teacher-forced;
        - ramp: an increasing fraction of blocks uses geometric VP;
        - full phase: every block uses geometric VP.

        Returning zero when training progress has not been initialized is the
        safe default: callers cannot accidentally enable aggressive AR
        exposure before ``set_training_progress`` has been called.
        """
        if not self.scaffold_on_policy or self._total_steps <= 0:
            return 0.0

        progress = self._global_step / self._total_steps
        warmup_end = self.scaffold_vp_warmup_ratio
        ramp_end = warmup_end + self.scaffold_vp_ramp_ratio

        if progress < warmup_end:
            return 0.0
        if self.scaffold_vp_ramp_ratio > 0 and progress < ramp_end:
            return (progress - warmup_end) / self.scaffold_vp_ramp_ratio
        return 1.0

    def _scaffold_vp_prefix_lengths(
        self, bsz: int, n_blocks: int, device: torch.device
    ) -> torch.Tensor:
        """Sample geometric prefix lengths for scaffold on-policy (Machine B).

        Uses an independent RNG stream seeded from ``_global_step`` so that
        prefix-length sampling does not perturb anchor-position sampling
        between Machine A and Machine B runs.

        During warmup, returns ``block_size`` for every block, which makes
        every position teacher-forced under the ``d < prefix_len`` boundary.
        During the ramp, only a scheduled fraction of blocks uses a sampled
        prefix; the rest remain fully teacher-forced.  After the ramp, every
        block uses a prefix in ``[min_prefix, block_size - 1]``.
        """
        exposure = self._effective_scaffold_vp_exposure()
        full_tf = torch.full(
            (bsz, n_blocks),
            self.block_size,
            dtype=torch.long,
            device=device,
        )
        if exposure <= 0.0:
            return full_tf

        # Independent generator — deterministic given global_step, yet
        # independent of the default RNG that drives anchor sampling.
        gen = torch.Generator(device=device)
        gen.manual_seed(self._global_step * 1_000_003 + 42)

        min_prefix = self.scaffold_vp_min_prefix
        max_prefix = self.block_size - 1
        if max_prefix <= min_prefix:
            sampled_prefixes = torch.full(
                (bsz, n_blocks), min_prefix, dtype=torch.long, device=device
            )
        else:
            prefix_ids = torch.arange(min_prefix, max_prefix + 1, device=device)
            weights = torch.pow(
                torch.full_like(
                    prefix_ids, self.scaffold_vp_base, dtype=torch.float32
                ),
                prefix_ids.float(),
            )
            samples = torch.multinomial(
                weights,
                num_samples=bsz * n_blocks,
                replacement=True,
                generator=gen,
            ).reshape(bsz, n_blocks)
            sampled_prefixes = samples + min_prefix

        if exposure >= 1.0:
            return sampled_prefixes

        # Block-level curriculum: exposed blocks use geometric VP, while the
        # remaining blocks stay fully TF.
        exposed = torch.rand(
            (bsz, n_blocks), device=device, generator=gen
        ) < exposure
        return torch.where(exposed, sampled_prefixes, full_tf)

    def _scaffold_teacher_forced_bias(
        self,
        target_ids: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Markov scaffold bias with teacher-forced predecessors.

        Position 0's predecessor is the anchor (ground truth).
        Position d>0's predecessor is ``target_ids[d-1]``.

        Args:
            target_ids: ``[B, N, K]`` — ground-truth token ids per block.
            anchor_token_ids: ``[B, N]`` — anchor (ground-truth) tokens.

        Returns:
            markov_bias: ``[B, N, K, V]`` — additive bias per position.
        """
        # Shift right: position d gets predecessor at d-1.
        prev_ids = target_ids.roll(shifts=1, dims=-1)  # [B, N, K]
        prev_ids[:, :, 0] = anchor_token_ids  # anchor = ground-truth predecessor
        return self.draft_model.markov_scaffold.compute_step_bias(
            prev_ids
        )  # [B, N, K, V]

    def _scaffold_on_policy_depth_scan(
        self,
        base_logits_4d: torch.Tensor,
        target_ids: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        prefix_lengths: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """On-policy depth scan for scaffold bias (Machine B / VP).

        For each block depth *d*:

        - ``d < prefix_len``: predecessor = ground truth (teacher-forced).
        - ``d >= prefix_len``: predecessor = argmax(corrected_{d-1}) (AR).

        The AR token ids are **detached** — no REINFORCE gradient.

        The loop is serial over block positions (at most K-1 steps)
        but fully parallel across batch and anchor dimensions.

        Args:
            base_logits_4d: ``[B, N, K, V]`` — base DFlash logits (will be
                detached internally so corrected CE trains scaffold only).
            target_ids: ``[B, N, K]``.
            anchor_token_ids: ``[B, N]``.
            prefix_lengths: ``[B, N]`` — per-block TF prefix boundary.
            weight_mask: ``[B, N, K]`` — valid position mask.

        Returns:
            corrected_logits_4d: ``[B, N, K, V]`` (base_det + bias).
            markov_bias_4d: ``[B, N, K, V]``.
            on_policy_ids: ``[B, N, K]`` — predecessor IDs used (detached).
        """
        B, N, K, V = base_logits_4d.shape
        device = base_logits_4d.device

        # Stop-gradient: corrected CE trains scaffold only, not backbone.
        base_det = base_logits_4d.detach()

        on_policy_ids = target_ids.new_zeros(B, N, K)
        markov_bias_4d = base_logits_4d.new_zeros(B, N, K, V)

        for d in range(K):
            if d == 0:
                # Position 0 always uses anchor (ground truth).
                prev_ids = anchor_token_ids  # [B, N]
            else:
                gt_prev = target_ids[:, :, d - 1]  # [B, N]

                # Corrected logits at d-1 = base_det[d-1] + bias[d-1].
                corrected_d_minus_1 = (
                    base_det[:, :, d - 1, :] + markov_bias_4d[:, :, d - 1, :]
                )
                ar_pred = corrected_d_minus_1.argmax(dim=-1).detach()  # [B, N]

                # d < prefix_len → TF; else → AR.
                is_tf = d < prefix_lengths  # [B, N]
                prev_ids = torch.where(is_tf, gt_prev, ar_pred)  # [B, N]

            on_policy_ids[:, :, d] = prev_ids

            bias_d = self.draft_model.markov_scaffold.compute_step_bias(
                prev_ids
            )  # [B, N, V]
            markov_bias_4d[:, :, d, :] = bias_d

        corrected_logits_4d = base_det + markov_bias_4d
        return corrected_logits_4d, markov_bias_4d, on_policy_ids

    def _scaffold_kd_loss(
        self,
        base_logits_4d: torch.Tensor,
        corrected_logits_4d: torch.Tensor,
        weight_mask: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaffold-to-Base Distillation: D_KL(sg[p^c] || p^b).

        Uses ``F.kl_div`` with ``log_target=True`` for numerically stable
        KL with temperature scaling.  The corrected side is detached so
        gradient flows **only** to the backbone + lm_head.

        Returns:
            kd_loss: scalar (weighted mean), scaled by T².
            kl_per_token: ``[B, N, K]`` (detached, for diagnostics).
        """
        T = temperature
        base_scaled = base_logits_4d.float() / T  # [B, N, K, V]
        corrected_scaled = corrected_logits_4d.float() / T

        log_p_base = F.log_softmax(base_scaled, dim=-1)
        log_p_corrected = F.log_softmax(corrected_scaled, dim=-1).detach()

        # F.kl_div(input, target, log_target=True) computes
        #   exp(target) * (target - input)  element-wise.
        # Summed over V: sum_v p_c(v) * [log p_c(v) - log p_b(v)]
        #             = D_KL(p_c || p_b)  (positive).
        kl_per_token = F.kl_div(
            log_p_base,
            log_p_corrected,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)  # [B, N, K]

        valid_count = weight_mask.sum() + 1e-6
        kd_loss = (kl_per_token * weight_mask).sum() / valid_count * (T ** 2)
        return kd_loss, kl_per_token.detach()

    @staticmethod
    def _scaffold_rescue_kd_mask(
        base_logits_4d: torch.Tensor,
        corrected_logits_4d: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a detached mask where the scaffold rescues base top-1."""
        with torch.no_grad():
            base_pred = base_logits_4d.argmax(dim=-1)
            corrected_pred = corrected_logits_4d.argmax(dim=-1)
            rescue = (
                base_pred.ne(target_ids)
                & corrected_pred.eq(target_ids)
                & weight_mask.gt(0)
            )
            return weight_mask * rescue.float()

    @staticmethod
    def _target_top2_margin(
        logits_4d: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return target logit minus the strongest non-target logit."""
        # Keep the vocabulary-sized tensor in its native dtype.  Only the
        # gathered target/top-2 values are promoted to fp32, avoiding another
        # full-vocabulary fp32 allocation during online training.
        target_logits = logits_4d.gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1).float()
        top2_values, top2_ids = logits_4d.topk(k=2, dim=-1)
        top2_values = top2_values.float()
        max_other = torch.where(
            top2_ids[..., 0].eq(target_ids),
            top2_values[..., 1],
            top2_values[..., 0],
        )
        return target_logits - max_other

    @classmethod
    def _scaffold_advantage_kd_mask(
        cls,
        base_logits_4d: torch.Tensor,
        corrected_logits_4d: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        first_rejection: bool = False,
        global_floor: float = 0.1,
        continuation_power: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build detached continuous verification-advantage KD weights.

        Positive target-margin improvement is compressed with ``log1p``.  In
        first-rejection mode it is further weighted by remaining valid suffix
        capacity at the base model's causal truncation point, while retaining
        ``global_floor`` times the global advantage signal.
        """
        with torch.no_grad():
            valid = weight_mask.gt(0)
            base_margin = cls._target_top2_margin(base_logits_4d, target_ids)
            corrected_margin = cls._target_top2_margin(
                corrected_logits_4d, target_ids
            )
            raw_advantage = (corrected_margin - base_margin).clamp_min(0.0)
            advantage = torch.log1p(raw_advantage) * weight_mask

            first_rejection_mask = torch.zeros_like(valid)
            if first_rejection:
                base_correct = base_logits_4d.argmax(dim=-1).eq(target_ids)
                accepted_or_invalid = base_correct | (~valid)
                alive_before = torch.ones_like(valid)
                alive_before[..., 1:] = (
                    accepted_or_invalid[..., :-1].int().cumprod(dim=-1).bool()
                )
                first_rejection_mask = valid & alive_before & (~base_correct)
                continuation = valid.float().flip(-1).cumsum(-1).flip(-1)
                continuation = continuation.pow(continuation_power)
                multiplier = global_floor + (
                    first_rejection_mask.float() * continuation
                )
                advantage = advantage * multiplier

            valid_count = valid.float().sum() + 1e-6
            positive = (raw_advantage > 0) & valid
            positive_count = positive.float().sum()
            valid_by_depth = valid.float().sum(dim=(0, 1)) + 1e-6
            positive_by_depth = positive.float().sum(dim=(0, 1))
            diag = {
                "positive_fraction": positive_count / valid_count,
                "mean_positive_advantage": (
                    (raw_advantage * positive.float()).sum()
                    / (positive_count + 1e-6)
                ),
                "first_rejection_fraction": (
                    first_rejection_mask.float().sum() / valid_count
                ),
                "positive_fraction_by_depth": (
                    positive_by_depth / valid_by_depth
                ),
                "mean_positive_advantage_by_depth": (
                    (raw_advantage * positive.float()).sum(dim=(0, 1))
                    / (positive_by_depth + 1e-6)
                ),
                "first_rejection_fraction_by_depth": (
                    first_rejection_mask.float().sum(dim=(0, 1))
                    / valid_by_depth
                ),
            }
            return advantage, diag

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _sample_prefix_lengths(
        self, bsz: int, n_blocks: int, device: torch.device
    ) -> torch.Tensor:
        """Sample visible prefix lengths for VP-Drafter training.

        A prefix length i means block positions [0, i) are visible real tokens and
        positions [i, block_size) are masked prediction targets. The sampled
        range follows D2SD's variable-prefix recipe while avoiding the degenerate
        fixed-anchor DFlash case.
        """
        min_prefix = min(2, self.block_size - 1)
        max_prefix = self.block_size - 1
        if max_prefix <= min_prefix:
            return torch.full(
                (bsz, n_blocks), min_prefix, dtype=torch.long, device=device
            )

        prefix_ids = torch.arange(min_prefix, max_prefix + 1, device=device)
        weights = torch.pow(
            torch.full_like(prefix_ids, self.prefix_weight_base, dtype=torch.float32),
            prefix_ids.float(),
        )
        samples = torch.multinomial(
            weights, num_samples=bsz * n_blocks, replacement=True
        ).reshape(bsz, n_blocks)
        return samples + min_prefix

    def prepare_noise_input(
        self, input_ids: torch.Tensor, block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if block_ids is not None:
            is_block_start = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
            is_block_start[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
        else:
            positions = torch.arange(seq_len, device=device)
            is_block_start = (positions % self.block_size) == 0
            is_block_start = is_block_start.unsqueeze(0).expand(bsz, -1)

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[is_block_start] = input_ids[is_block_start]
        return noise_input_ids

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _create_vp_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare VP-Drafter inputs with variable visible prefixes."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        offsets = torch.arange(bs, device=device).view(1, 1, -1)
        token_positions = anchor_positions.unsqueeze(-1) + offsets
        safe_positions = token_positions.clamp(0, seq_len - 1)

        real_tokens = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n, -1),
            2,
            safe_positions,
        )
        visible_prefix = offsets < prefix_lengths.unsqueeze(-1)
        valid_positions = token_positions < seq_len
        fill_mask = visible_prefix & block_keep_mask.unsqueeze(-1) & valid_positions

        mask_tokens = torch.full_like(real_tokens, self.mask_token_id)
        noise_ids = torch.where(fill_mask, real_tokens, mask_tokens)
        return self.embed_tokens(noise_ids.reshape(bsz, n * bs))

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Compute detached D-PACE position weights.

        ``prob`` is the draft probability on the target token at each draft
        position. Invalid positions are treated as multiplicative no-ops inside
        prefix products and excluded from suffix sums; the caller still
        multiplies the returned weights by ``binary_mask`` before reduction.
        """
        smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
        smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)

        if loss_type == "dpace-cumulative-confidence-only":
            return prefix

        suffix = torch.flip(
            torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
            dims=[-1],
        )

        if loss_type == "dpace":
            return suffix
        if loss_type == "dpace-continuation-value-only":
            return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")

    @staticmethod
    def _davca_features(
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        neg_log_q: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build detached, label-nontrivial critic features without softmax."""
        with torch.no_grad():
            target_logit = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            # CE = logsumexp(logits) - target_logit, avoiding another vocab-wide
            # reduction solely for the critic.
            log_normalizer = target_logit + neg_log_q
            top2 = logits.topk(k=2, dim=-1).values
            q = torch.exp(-neg_log_q).clamp(1e-6, 1.0 - 1e-6)
            top1_prob = torch.exp(top2[..., 0] - log_normalizer).clamp(0.0, 1.0)
            top_gap = torch.tanh((top2[..., 0] - top2[..., 1]) / 5.0)
            log_q = torch.log(q)
            valid = valid_mask.to(log_q.dtype)
            valid_count = torch.cumsum(valid, dim=-1).clamp_min(1.0)
            cumulative_mean_log_q = torch.cumsum(log_q * valid, dim=-1) / valid_count
            depth = torch.arange(
                logits.shape[-2], device=logits.device, dtype=log_q.dtype
            )
            depth = depth / max(logits.shape[-2] - 1, 1)
            depth = depth.view(1, 1, -1).expand_as(log_q)
            features = torch.stack(
                [
                    log_q / 10.0,
                    q,
                    top1_prob,
                    top_gap,
                    cumulative_mean_log_q / 10.0,
                    depth,
                ],
                dim=-1,
            )
            return features

    def _davca_objective(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        neg_log_q: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute dependency-aware verification credit assignment.

        The critic learns actual prefix-survival events.  Its residual hazard is
        anchored to D-PACE, and all credit weights are detached so gradients from
        draft CE cannot manipulate the critic or flow through argmax labels.
        """
        if self.verification_critic is None:
            raise RuntimeError("DAVCA objective requires verification_critic")

        valid_b = valid_mask > 0
        q = torch.exp(-neg_log_q.detach())
        smooth_q = (1.0 - self.dpace_alpha) * q + self.dpace_alpha
        smooth_q = torch.where(valid_b, smooth_q, torch.ones_like(smooth_q))

        features = self._davca_features(
            logits.detach(), target_ids, neg_log_q.detach(), valid_mask
        )
        critic_dtype = next(self.verification_critic.parameters()).dtype
        flat_features = features.reshape(-1, features.shape[-2], features.shape[-1])
        residual = self.verification_critic(flat_features.to(critic_dtype))
        residual = residual.reshape_as(neg_log_q).float()

        eps = 1e-4
        base_logit = torch.logit(smooth_q.float().clamp(eps, 1.0 - eps))
        conditional_survival = torch.sigmoid(base_logit + residual)
        conditional_survival = torch.where(
            valid_b, conditional_survival, torch.ones_like(conditional_survival)
        )
        predicted_survival = torch.cumprod(conditional_survival, dim=-1)

        with torch.no_grad():
            correct = logits.argmax(dim=-1).eq(target_ids)
            survival_event = torch.where(valid_b, correct, torch.ones_like(correct))
            survival_target = torch.cumprod(survival_event.int(), dim=-1).float()

        critic_per_position = F.binary_cross_entropy(
            predicted_survival.clamp(eps, 1.0 - eps),
            survival_target,
            reduction="none",
        )
        critic_loss = (critic_per_position * valid_mask).sum() / (
            valid_mask.sum() + 1e-6
        )

        davca_weights = torch.flip(
            torch.cumsum(
                torch.flip(predicted_survival * valid_mask, dims=[-1]), dim=-1
            ),
            dims=[-1],
        ).detach()
        dpace_weights = self._dpace_weight(q, valid_mask, valid_b, "dpace")
        blend = self._effective_davca_blend()
        credit_weights = (1.0 - blend) * dpace_weights + blend * davca_weights
        draft_loss = (neg_log_q * valid_mask * credit_weights).sum() / float(
            logits.shape[0]
        )
        total = draft_loss + self.davca_critic_beta * critic_loss

        valid_count = valid_mask.sum() + 1e-6
        brier = (
            (predicted_survival.detach() - survival_target).square() * valid_mask
        ).sum() / valid_count
        diagnostics = {
            "davca_draft_loss": draft_loss.detach(),
            "davca_critic_loss": critic_loss.detach(),
            "davca_blend": torch.tensor(blend, device=logits.device),
            "davca_brier": brier.detach(),
            "davca_predicted_survival": (
                predicted_survival.detach() * valid_mask
            ).sum()
            / valid_count,
            "davca_actual_survival": (survival_target * valid_mask).sum()
            / valid_count,
            "davca_weight_ratio_to_dpace": (
                (davca_weights * valid_mask).sum()
                / ((dpace_weights * valid_mask).sum() + 1e-6)
            ).detach(),
        }
        for depth in range(1, self.block_size):
            valid_d = valid_mask[..., depth]
            count_d = valid_d.sum() + 1e-6
            diagnostics[f"davca_predicted_survival_depth_{depth}"] = (
                predicted_survival[..., depth].detach() * valid_d
            ).sum() / count_d
            diagnostics[f"davca_actual_survival_depth_{depth}"] = (
                survival_target[..., depth] * valid_d
            ).sum() / count_d
        return total, diagnostics

    @staticmethod
    def _counterfactual_continuation(
        correct: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Tokens recoverable if the current position were made correct.

        For position ``d`` this is one (the hypothetically repaired token) plus
        the already-correct consecutive run beginning at ``d + 1``.  The
        current prediction is deliberately excluded, making this a
        counterfactual repair value rather than another correctness weight.
        """
        valid_b = valid_mask > 0
        potential = torch.zeros_like(valid_mask, dtype=torch.float32)
        running = torch.zeros_like(valid_mask[..., 0], dtype=torch.float32)
        for depth in range(valid_mask.shape[-1] - 1, -1, -1):
            if depth == valid_mask.shape[-1] - 1:
                future = torch.zeros_like(running)
            else:
                next_good = valid_b[..., depth + 1] & correct[..., depth + 1]
                future = torch.where(next_good, running, torch.zeros_like(running))
            running = torch.where(
                valid_b[..., depth], 1.0 + future, torch.zeros_like(future)
            )
            potential[..., depth] = running
        return potential

    @staticmethod
    def _expected_counterfactual_continuation(
        conditional_prob: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """D-PACE expectation corresponding to counterfactual continuation."""
        valid_b = valid_mask > 0
        expected = torch.zeros_like(valid_mask, dtype=torch.float32)
        running = torch.zeros_like(valid_mask[..., 0], dtype=torch.float32)
        prob = conditional_prob.float()
        for depth in range(valid_mask.shape[-1] - 1, -1, -1):
            if depth == valid_mask.shape[-1] - 1:
                future = torch.zeros_like(running)
            else:
                next_valid = valid_b[..., depth + 1]
                future = torch.where(
                    next_valid,
                    prob[..., depth + 1] * running,
                    torch.zeros_like(running),
                )
            running = torch.where(
                valid_b[..., depth], 1.0 + future, torch.zeros_like(future)
            )
            expected[..., depth] = running
        return expected

    @staticmethod
    def _match_credit_mass(
        candidate: torch.Tensor,
        reference: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Match candidate/reference weight mass independently for each block."""
        candidate_mass = (candidate * valid_mask).sum(dim=-1, keepdim=True)
        reference_mass = (reference * valid_mask).sum(dim=-1, keepdim=True)
        scale = reference_mass / candidate_mass.clamp_min(1e-6)
        scale = torch.where(candidate_mass > 0, scale, torch.ones_like(scale))
        return candidate * scale

    def _cva_objective(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        neg_log_q: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Counterfactual Verification Advantage credit assignment.

        ``cva-direct`` uses the recoverable suffix length directly.  The safer
        ``cva-residual`` retains D-PACE and applies only a bounded positive
        correction when observed recoverable continuation exceeds its D-PACE
        expectation.  Both candidates are mass-normalized per block before the
        scheduled blend, isolating credit allocation from learning-rate scale.
        """
        valid_b = valid_mask > 0
        with torch.no_grad():
            q = torch.exp(-neg_log_q)
            smooth_q = (1.0 - self.dpace_alpha) * q + self.dpace_alpha
            smooth_q = torch.where(valid_b, smooth_q, torch.ones_like(smooth_q))
            dpace_weights = self._dpace_weight(
                q, valid_mask, valid_b, "dpace"
            )
            correct = logits.argmax(dim=-1).eq(target_ids)
            potential = self._counterfactual_continuation(correct, valid_mask)
            expected = self._expected_counterfactual_continuation(
                smooth_q, valid_mask
            )

            if self.loss_type == "cva-direct":
                candidate = potential.pow(self.cva_continuation_power)
                modifier = candidate / expected.pow(
                    self.cva_continuation_power
                ).clamp_min(1e-6)
            elif self.loss_type == "cva-residual":
                log_ratio = torch.log(
                    potential.clamp_min(1e-6) / expected.clamp_min(1e-6)
                ) / self.cva_residual_temperature
                modifier = torch.exp(log_ratio).clamp(
                    min=self.cva_residual_min,
                    max=self.cva_residual_max,
                )
                modifier = torch.where(valid_b, modifier, torch.ones_like(modifier))
                candidate = dpace_weights * modifier
            else:
                raise ValueError(f"unknown CVA loss_type {self.loss_type!r}")

            candidate = self._match_credit_mass(
                candidate, dpace_weights, valid_mask
            )
            blend = self._effective_cva_blend()
            credit_weights = (1.0 - blend) * dpace_weights + blend * candidate
            effective_multiplier = candidate / dpace_weights.clamp_min(1e-6)

        loss = (neg_log_q * valid_mask * credit_weights).sum() / float(
            logits.shape[0]
        )
        valid_count = valid_mask.sum() + 1e-6
        first_rejection = valid_b & (~correct)
        alive_before = torch.ones_like(valid_b)
        accepted_or_invalid = correct | (~valid_b)
        alive_before[..., 1:] = (
            accepted_or_invalid[..., :-1].int().cumprod(dim=-1).bool()
        )
        first_rejection &= alive_before
        frontier_count = first_rejection.float().sum() + 1e-6

        diagnostics = {
            "cva_loss": loss.detach(),
            "cva_blend": torch.tensor(blend, device=logits.device),
            "cva_potential": (potential * valid_mask).sum() / valid_count,
            "cva_expected_potential": (expected * valid_mask).sum() / valid_count,
            "cva_frontier_potential": (
                potential * first_rejection.float()
            ).sum()
            / frontier_count,
            "cva_modifier": (modifier * valid_mask).sum() / valid_count,
            "cva_upweight_fraction": (
                ((modifier > 1.0) & valid_b).float().sum() / valid_count
            ),
            "cva_candidate_weight_ratio_to_dpace": (
                (candidate * valid_mask).sum()
                / ((dpace_weights * valid_mask).sum() + 1e-6)
            ),
            "cva_effective_multiplier_min": torch.where(
                valid_b,
                effective_multiplier,
                torch.full_like(effective_multiplier, float("inf")),
            ).amin(),
            "cva_effective_multiplier_max": torch.where(
                valid_b,
                effective_multiplier,
                torch.zeros_like(effective_multiplier),
            ).amax(),
        }
        for depth in range(1, self.block_size):
            valid_d = valid_mask[..., depth]
            count_d = valid_d.sum() + 1e-6
            diagnostics[f"cva_potential_depth_{depth}"] = (
                potential[..., depth] * valid_d
            ).sum() / count_d
            diagnostics[f"cva_modifier_depth_{depth}"] = (
                modifier[..., depth] * valid_d
            ).sum() / count_d
        return loss, {key: value.detach() for key, value in diagnostics.items()}

    def _vcrd_objective(
        self,
        *,
        base_logits: torch.Tensor,
        target_ids: torch.Tensor,
        base_q: torch.Tensor,
        valid_mask: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        full_position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Distil a verifier-repaired second draft pass into the first pass.

        The base branch is the deployment-time all-mask DFlash path.  A
        training-only teacher pass reveals the target prefix through the
        highest-probability verification frontier, predicts the remaining
        suffix under ``no_grad``, and supervises the base distribution only
        when that intervention improves expected continuation.
        """
        device = base_logits.device
        valid = valid_mask.bool()

        # Internal holes terminate the strict verification chain.  Position 0
        # is the visible anchor and is a multiplicative identity, not a target.
        chain_valid = valid.int()
        chain_valid[..., 0] = 1
        chain_valid = chain_valid.cumprod(dim=-1).bool()
        chain_valid[..., 0] = False
        has_future = torch.zeros_like(chain_valid)
        has_future[..., :-1] = chain_valid[..., 1:]
        candidate = chain_valid & has_future

        with torch.no_grad():
            smooth_q = (1.0 - self.dpace_alpha) * base_q + self.dpace_alpha
            prefix_factor = torch.where(
                chain_valid,
                smooth_q,
                torch.ones_like(smooth_q),
            )
            reach_before = torch.ones_like(base_q)
            if base_q.size(-1) > 1:
                reach_before[..., 1:] = torch.cumprod(
                    prefix_factor[..., :-1], dim=-1
                )
            frontier_score = (
                reach_before * (1.0 - base_q) * candidate.float()
            )
            selected_depth = frontier_score.argmax(dim=-1)
            has_candidate = candidate.any(dim=-1)
            selected_depth = torch.where(
                has_candidate,
                selected_depth,
                torch.zeros_like(selected_depth),
            )
            prefix_lengths = (selected_depth + 1).clamp(
                min=1, max=self.block_size
            )
            rollout_keep = block_keep_mask & has_candidate

            teacher_noise = self._create_vp_noise_embed(
                input_ids,
                anchor_positions,
                rollout_keep,
                prefix_lengths,
            )
            was_training = self.draft_model.training
            try:
                self.draft_model.eval()
                teacher_hidden = self.draft_model(
                    position_ids=full_position_ids,
                    noise_embedding=teacher_noise,
                    target_hidden=hidden_states,
                    attention_mask=attention_mask,
                )
                teacher_logits = self.lm_head(teacher_hidden).view_as(
                    base_logits
                )
            finally:
                self.draft_model.train(was_training)

            teacher_nll = F.cross_entropy(
                teacher_logits.reshape(-1, teacher_logits.size(-1)),
                target_ids.reshape(-1),
                reduction="none",
            ).view_as(target_ids)
            teacher_q = torch.exp(-teacher_nll)
            topk = min(self.vcrd_topk, teacher_logits.size(-1))
            teacher_top_values, teacher_top_ids = torch.topk(
                teacher_logits, k=topk, dim=-1
            )
            teacher_topk_mass = torch.exp(
                torch.logsumexp(teacher_top_values.float(), dim=-1)
                - torch.logsumexp(teacher_logits, dim=-1).float()
            )

            base_expected = self._expected_counterfactual_continuation(
                base_q, chain_valid.float()
            )
            teacher_expected = self._expected_counterfactual_continuation(
                teacher_q, chain_valid.float()
            )
            selected_index = selected_depth.unsqueeze(-1)
            base_selected_value = base_expected.gather(
                -1, selected_index
            ).squeeze(-1)
            teacher_selected_value = teacher_expected.gather(
                -1, selected_index
            ).squeeze(-1)
            raw_gain = teacher_selected_value - base_selected_value
            gain = (raw_gain - self.vcrd_min_gain).clamp_min(0.0)
            gain = gain * has_candidate.float()

            smooth_teacher_q = (
                (1.0 - self.dpace_alpha) * teacher_q + self.dpace_alpha
            )
            survival_before = torch.zeros_like(teacher_q)
            alive = torch.ones_like(teacher_q[..., 0])
            positions = torch.arange(
                self.block_size, device=device
            ).view(1, 1, -1)
            suffix_mask = (
                positions > selected_depth.unsqueeze(-1)
            ) & chain_valid
            for depth in range(self.block_size):
                after_frontier = depth > selected_depth
                survival_before[..., depth] = torch.where(
                    after_frontier,
                    alive,
                    torch.zeros_like(alive),
                )
                alive = torch.where(
                    after_frontier,
                    alive * smooth_teacher_q[..., depth],
                    alive,
                )
            distill_credit = (
                suffix_mask.float()
                * survival_before
                * gain.unsqueeze(-1)
            ).detach()

            del teacher_logits, teacher_hidden, teacher_noise

        active = distill_credit > 0
        credit_mass = distill_credit.sum()
        if active.any():
            temperature = self.vcrd_temperature
            student_active = base_logits[active].float() / temperature
            active_top_ids = teacher_top_ids[active]
            student_top_logits = student_active.gather(-1, active_top_ids)
            student_log_z = torch.logsumexp(
                student_active, dim=-1, keepdim=True
            )
            student_top_log_probs = student_top_logits - student_log_z

            teacher_top_log_probs = F.log_softmax(
                teacher_top_values[active].float() / temperature,
                dim=-1,
            )
            teacher_top_probs = teacher_top_log_probs.exp()
            kl_per_position = (
                teacher_top_probs
                * (teacher_top_log_probs - student_top_log_probs)
            ).sum(dim=-1) * (temperature ** 2)
            active_credit = distill_credit[active]
            vcrd_loss = (
                kl_per_position * active_credit
            ).sum() / credit_mass.clamp_min(1e-6)
        else:
            # Preserve a valid graph when no intervention improves the suffix.
            vcrd_loss = base_logits.sum() * 0.0

        candidate_count = has_candidate.float().sum().clamp_min(1.0)
        valid_count = chain_valid.float().sum().clamp_min(1.0)
        diagnostics = {
            "vcrd_loss": vcrd_loss.detach(),
            "vcrd_rollout_active": torch.ones(
                (), device=device, dtype=torch.float32
            ),
            "vcrd_candidate_fraction": has_candidate.float().mean(),
            "vcrd_selected_depth": (
                selected_depth.float() * has_candidate.float()
            ).sum()
            / candidate_count,
            "vcrd_frontier_score": (
                frontier_score.gather(-1, selected_depth.unsqueeze(-1))
                .squeeze(-1)
                .sum()
                / candidate_count
            ),
            "vcrd_base_continuation": (
                base_selected_value * has_candidate.float()
            ).sum()
            / candidate_count,
            "vcrd_teacher_continuation": (
                teacher_selected_value * has_candidate.float()
            ).sum()
            / candidate_count,
            "vcrd_raw_gain": (
                raw_gain * has_candidate.float()
            ).sum()
            / candidate_count,
            "vcrd_positive_gain_fraction": (
                gain.gt(0).float() * has_candidate.float()
            ).sum()
            / candidate_count,
            "vcrd_distill_token_fraction": active.float().sum()
            / valid_count,
            "vcrd_credit_mass": credit_mass.detach(),
            "vcrd_teacher_topk_mass": (
                teacher_topk_mass * active.float()
            ).sum()
            / active.float().sum().clamp_min(1.0),
        }
        return vcrd_loss, {
            key: value.detach() for key, value in diagnostics.items()
        }

    def _dflash_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Return additive DFlash/D-PACE, LK, selector, and metric terms."""

        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        objective_logits = (
            self.draft_model.transform_unary_logits(logits)
            if candidate_selector is not None
            else logits
        )
        neg_log_q = F.cross_entropy(
            objective_logits.reshape(-1, objective_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        target_probability = torch.exp(-neg_log_q)

        loss_weights = weight_mask
        if self.loss_type == "dflash":
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                positions = torch.arange(
                    self.block_size, device=hidden.device
                ).view(1, 1, -1)
                decay_weights = torch.exp(
                    -(positions - 1).clamp(min=0).float()
                    / self.loss_decay_gamma
                )
                loss_weights = loss_weights * decay_weights
            loss_den = loss_weights.sum()
        elif self.loss_type in _DPACE_LOSS_TYPES:
            with torch.no_grad():
                dpace_weights = self._dpace_weight(
                    target_probability.detach(),
                    weight_mask,
                    weight_mask > 0,
                    self.loss_type,
                )
            loss_weights = weight_mask * dpace_weights
            loss_den = loss_weights.sum()
        else:
            raise ValueError(f"unknown loss_type {self.loss_type!r}")

        ce_loss_num = (neg_log_q * loss_weights).sum()
        if self.lk_loss_type in {"lambda", "tv"}:
            tv_loss_num = ((1.0 - target_probability) * loss_weights).sum()
        else:
            tv_loss_num = ce_loss_num.new_zeros(())
        target_probability_num = (
            target_probability.detach() * weight_mask
        ).sum()

        selector_ce_num = ce_loss_num.new_zeros(())
        selector_tv_num = ce_loss_num.new_zeros(())
        selector_probability_num = ce_loss_num.new_zeros(())
        selector_correct_num = ce_loss_num.new_zeros(())
        selector_weight_den = ce_loss_num.new_zeros(())
        selector_covered_num = ce_loss_num.new_zeros(())
        if candidate_selector is not None and self.selector_loss_alpha > 0:
            (
                unary_logits,
                candidate_ids,
                target_candidate_index,
                target_is_candidate,
            ) = _selector_training_candidates(
                objective_logits,
                target_ids,
                candidate_selector.top_k,
            )
            selector_logits = candidate_selector.score_candidates(
                candidate_ids=candidate_ids,
                unary_logits=unary_logits,
                hidden_states=hidden,
                predecessor_ids=predecessor_ids,
            )
            selector_ce = F.cross_entropy(
                selector_logits.float().reshape(-1, selector_logits.shape[-1]),
                target_candidate_index.reshape(-1),
                reduction="none",
            ).reshape_as(target_ids)
            selector_probability = torch.exp(-selector_ce)
            selector_loss_weights = (
                loss_weights * target_is_candidate.float()
            )
            selector_metric_mask = (
                weight_mask * target_is_candidate.float()
            )
            selector_ce_num = (selector_ce * selector_loss_weights).sum()
            selector_probability_num = (
                selector_probability.detach() * selector_metric_mask
            ).sum()
            selector_weight_den = selector_loss_weights.sum()
            selector_covered_num = (
                weight_mask * target_is_candidate.float()
            ).sum()
            with torch.no_grad():
                selected_ids = candidate_ids.gather(
                    -1, selector_logits.argmax(dim=-1, keepdim=True)
                ).squeeze(-1)
                selector_correct_num = (
                    selected_ids.eq(target_ids).float()
                    * selector_loss_weights
                ).sum()

        with torch.no_grad():
            predicted_ids = objective_logits.argmax(dim=-1)
            correct_num = (
                (predicted_ids.eq(target_ids) & (weight_mask > 0.5))
                .sum()
                .float()
            )
            accuracy_den = weight_mask.sum()
        return (
            ce_loss_num,
            tv_loss_num,
            loss_den,
            target_probability_num,
            correct_num,
            accuracy_den,
            selector_ce_num,
            selector_tv_num,
            selector_probability_num,
            selector_correct_num,
            selector_weight_den,
            selector_covered_num,
        )

    def _compose_token_objective(
        self,
        ce_num: torch.Tensor,
        tv_num: torch.Tensor,
        probability_num: torch.Tensor,
        probability_den: torch.Tensor,
    ) -> torch.Tensor:
        """Compose a hard-target CE/TV/LK numerator."""

        if self.lk_loss_type is None or self.lk_loss_type == "alpha":
            # With a one-hot target distribution, LK-alpha equals NLL/CE.
            return ce_num
        if self.lk_loss_type == "tv":
            return tv_num
        if self.lk_loss_type == "lambda":
            acceptance = probability_num / probability_den.clamp_min(1.0)
            kl_weight = self.kl_scale * torch.exp(
                -self.kl_decay * acceptance.detach()
            )
            return kl_weight * ce_num + (1.0 - kl_weight) * tv_num
        raise ValueError(f"unknown lk_loss_type {self.lk_loss_type!r}")

    def _can_use_chunked_objective(self) -> bool:
        """Whether the memory-bounded PR objective covers this configuration.

        The legacy branch contains many research auxiliaries that require the
        complete vocabulary tensor. They retain their existing path; the clean
        DFlash/D-PACE + DFlash2 objective uses chunked logits.
        """

        if self.objective_chunk_blocks <= 0:
            return False
        if self.loss_type not in ({"dflash"} | _DPACE_LOSS_TYPES):
            return False
        return not any(
            (
                self.dpace_auf_mode != "none",
                self.path_acceptance_alpha > 0,
                self.survival_loss_alpha > 0,
                self.first_rejection_loss_alpha > 0,
                self.first_rejection_deep_anchor_alpha > 0,
                self.frbo_survival_alpha > 0,
                self.frbo_boundary_alpha > 0,
                self.markov_scaffold_ce_alpha > 0,
                self.scaffold_distill_alpha > 0,
                self.acceptance_replay_signals,
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Parallel block-wise training forward pass.

        Returns:
            loss: Combined CE (and optionally D-PACE) + survival loss.
            accuracy: Per-token greedy accuracy over valid positions.
            loss_components: Dict of detached loss component scalars
                (empty when ``survival_loss_alpha == 0``).
        """
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        prefix_lengths = None
        if self.loss_type == "vp_drafter":
            prefix_lengths = self._sample_prefix_lengths(
                bsz, anchor_positions.shape[1], device
            )
            noise_embedding = self._create_vp_noise_embed(
                input_ids, anchor_positions, block_keep_mask, prefix_lengths
            )
        else:
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
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        if self.loss_type == "vp_drafter":
            weight_mask = (
                weight_mask * (pos_in_block >= prefix_lengths.unsqueeze(-1)).float()
            )
        else:
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        if self._can_use_chunked_objective():
            predecessor_ids = torch.cat(
                [target_ids[:, :, :1], target_ids[:, :, :-1]], dim=-1
            )
            hidden_4d = output_hidden.reshape(
                bsz,
                anchor_positions.shape[1],
                self.block_size,
                -1,
            )
            (
                ce_loss_num,
                tv_loss_num,
                loss_den,
                target_probability_num,
                correct_num,
                accuracy_den,
                selector_ce_num,
                selector_tv_num,
                selector_probability_num,
                selector_correct_num,
                selector_weight_den,
                selector_covered_num,
            ) = checkpointed_chunk_reduce(
                self._dflash_objective_chunk_terms,
                hidden_4d,
                target_ids,
                weight_mask,
                predecessor_ids,
                chunk_size=self.objective_chunk_blocks,
                dim=1,
            )
            loss_num = self._compose_token_objective(
                ce_loss_num,
                tv_loss_num,
                target_probability_num,
                accuracy_den,
            )
            effective_selector_alpha = self._effective_selector_loss_alpha()
            selector_loss_num = loss_num.new_zeros(())
            has_selector_objective = (
                getattr(self.draft_model, "candidate_selector", None) is not None
                and self.selector_loss_alpha > 0
            )
            if has_selector_objective:
                # Selector is a categorical distribution over serving top-k;
                # keep its calibrated CE independent of optional LK/TV.
                selector_loss_num = selector_ce_num
                loss_num = (
                    loss_num
                    + effective_selector_alpha * selector_loss_num
                )

            loss_denominator = loss_den.clamp_min(1e-6)
            accuracy_den = accuracy_den.clamp_min(1.0)
            loss = loss_num / loss_denominator
            accuracy = correct_num / accuracy_den
            loss_components: Dict[str, torch.Tensor] = {
                "primary_loss": self._compose_token_objective(
                    ce_loss_num,
                    tv_loss_num,
                    target_probability_num,
                    accuracy_den,
                ).detach()
                / loss_denominator.detach(),
                "target_probability": (
                    target_probability_num / accuracy_den
                ).detach(),
                "objective_chunk_blocks": loss.new_tensor(
                    float(self.objective_chunk_blocks)
                ),
            }
            if self.lk_loss_type == "lambda":
                loss_components["lk_kl_weight"] = (
                    self.kl_scale
                    * torch.exp(
                        -self.kl_decay
                        * (target_probability_num / accuracy_den).detach()
                    )
                ).detach()
            if has_selector_objective:
                selector_den = selector_weight_den.clamp_min(1e-6)
                loss_components.update(
                    {
                        "selector_loss": (
                            selector_loss_num / loss_denominator
                        ).detach(),
                        "selector_accuracy": (
                            selector_correct_num / selector_den
                        ).detach(),
                        "selector_coverage": (
                            selector_covered_num / accuracy_den
                        ).detach(),
                        "selector_target_probability": (
                            selector_probability_num
                            / selector_covered_num.clamp_min(1.0)
                        ).detach(),
                        "selector_loss_alpha": loss.new_tensor(
                            effective_selector_alpha
                        ),
                    }
                )
            return loss, accuracy, loss_components

        logits = self.lm_head(output_hidden)
        if getattr(self.draft_model, "candidate_selector", None) is not None:
            # DFlash2 applies its unary-logit transform to both the primary
            # objective and the selector, matching serving-time candidate
            # construction and the latest upstream training contract.
            logits = self.draft_model.transform_unary_logits(logits)

        binary_eval_mask = weight_mask.view(-1)

        # --- Markov Scaffold: compute corrected logits ---
        n_blocks = anchor_positions.shape[1]
        effective_scaffold_ce_alpha = self._effective_scaffold_ce_alpha()
        effective_scaffold_kd_alpha = self._effective_scaffold_kd_alpha()
        scaffold_active = (
            hasattr(self.draft_model, "markov_scaffold")
            and (
                effective_scaffold_ce_alpha > 0
                or effective_scaffold_kd_alpha > 0
            )
        )

        markov_bias_4d: Optional[torch.Tensor] = None
        corrected_logits_4d: Optional[torch.Tensor] = None
        scaffold_prefix_lengths: Optional[torch.Tensor] = None

        if scaffold_active:
            logits_4d = logits.view(bsz, n_blocks, self.block_size, -1)

            # Anchor token IDs (ground truth at each anchor position).
            valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
            anchor_token_ids = torch.gather(
                input_ids, 1, valid_anchor_positions
            )  # [B, N]

            if not self.scaffold_on_policy:
                # Machine A (Teacher-Forced): all predecessors are GT.
                markov_bias_4d = self._scaffold_teacher_forced_bias(
                    target_ids, anchor_token_ids
                )  # [B, N, K, V]
                # Stop-gradient on base: corrected CE trains scaffold ONLY.
                corrected_logits_4d = logits_4d.detach() + markov_bias_4d
            else:
                # Machine B (Variable-Prefix / On-Policy).
                scaffold_prefix_lengths = self._scaffold_vp_prefix_lengths(
                    bsz, n_blocks, device
                )  # [B, N]
                corrected_logits_4d, markov_bias_4d, _on_policy_ids = (
                    self._scaffold_on_policy_depth_scan(
                        logits_4d, target_ids, anchor_token_ids,
                        scaffold_prefix_lengths, weight_mask,
                    )
                )

        # --- Cross entropy ---
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        dpace_reference_weight_mass = None
        dpace_auf_diagnostics: Optional[Dict[str, torch.Tensor]] = None
        if self.loss_type == "dflash" or self.loss_type in _CFU_LOSS_TYPES:
            # Preserve the existing DFlash weighted-mean behavior.
            loss_weights = weight_mask
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                k = torch.arange(self.block_size, device=device).view(1, 1, -1)
                decay_weights = torch.exp(
                    -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
                )
                loss_weights = loss_weights * decay_weights

            flat_weights = loss_weights.view(-1)
            valid_token_count = flat_weights.sum() + 1e-6
            primary_loss_denominator = valid_token_count
            ce_loss_num = (loss_per_token * flat_weights).sum()
            target_probability = torch.exp(
                -loss_per_token.view_as(target_ids)
            )
            tv_loss_num = (
                (1.0 - target_probability) * loss_weights
            ).sum()
            target_probability_num = (
                target_probability.detach() * weight_mask
            ).sum()
            loss = self._compose_token_objective(
                ce_loss_num,
                tv_loss_num,
                target_probability_num,
                weight_mask.sum(),
            ) / primary_loss_denominator

            if self.loss_type in _CFU_LOSS_TYPES:
                # Make supervision contiguous after the anchor.  An internal
                # invalid position ends the verification chain rather than
                # allowing a later label to become a false frontier.
                cfu_valid_mask = (weight_mask > 0).int()
                cfu_valid_mask[:, :, 0] = 1
                cfu_valid_mask = cfu_valid_mask.cumprod(dim=-1).float()
                cfu_valid_mask[:, :, 0] = 0.0

                from specforge.core.acceptance_losses import (
                    counterfactual_frontier_utility_loss,
                )

                cfu_mode = self.loss_type.removeprefix("cfu-")
                effective_cfu_advantage_beta = (
                    self._effective_cfu_advantage_beta()
                )
                cfu_loss, cfu_diagnostics = (
                    counterfactual_frontier_utility_loss(
                        logits=logits.view(
                            bsz, n_blocks, self.block_size, -1
                        ),
                        target_ids=target_ids,
                        valid_mask=cfu_valid_mask,
                        mode=cfu_mode,
                        frontier_temperature=self.cfu_frontier_temperature,
                        loss_temperature=self.cfu_loss_temperature,
                        target_margin=self.cfu_target_margin,
                        continuation_power=self.cfu_continuation_power,
                        utility_transform=self.cfu_utility_transform,
                        advantage_beta=effective_cfu_advantage_beta,
                        advantage_temperature=(
                            self.cfu_advantage_temperature
                        ),
                        advantage_min_factor=(
                            self.cfu_advantage_min_factor
                        ),
                        advantage_max_factor=(
                            self.cfu_advantage_max_factor
                        ),
                    )
                )
                effective_cfu_alpha = self._effective_cfu_alpha()
                loss = loss + effective_cfu_alpha * cfu_loss
        elif self.loss_type == "vp_drafter":
            loss_weights = weight_mask
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                k = torch.arange(self.block_size, device=device).view(1, 1, -1)
                effective_pos = (
                    k.float() - prefix_lengths.unsqueeze(-1).float()
                ).clamp(min=0)
                decay_weights = torch.exp(-effective_pos / self.loss_decay_gamma)
                loss_weights = loss_weights * decay_weights

            flat_weights = loss_weights.view(-1)
            valid_token_count = flat_weights.sum() + 1e-6
            primary_loss_denominator = valid_token_count
            loss = (loss_per_token * flat_weights).sum() / valid_token_count
        elif (
            self.loss_type in _DPACE_LOSS_TYPES
            or self.loss_type in _VCRD_LOSS_TYPES
        ):
            neg_log_q = loss_per_token.view_as(target_ids)
            q = torch.exp(-neg_log_q)
            with torch.no_grad():
                dpace_weights = self._dpace_weight(
                    q.detach(),
                    weight_mask,
                    weight_mask > 0,
                    (
                        "dpace"
                        if self.loss_type in _VCRD_LOSS_TYPES
                        else self.loss_type
                    ),
                )
            loss_weights = weight_mask * dpace_weights
            if self.dpace_auf_mode != "none":
                from specforge.core.acceptance_losses import (
                    reachability_gated_dpace_weights,
                )

                with torch.no_grad():
                    hard_accept = logits.view(
                        bsz, n_blocks, self.block_size, -1
                    ).argmax(dim=-1).eq(target_ids)
                loss_weights, dpace_auf_diagnostics = (
                    reachability_gated_dpace_weights(
                        dpace_weights=loss_weights,
                        hard_accept=hard_accept,
                        valid_mask=weight_mask > 0,
                        suffix_retention=(
                            0.0
                            if self.dpace_auf_mode == "hard"
                            else self.dpace_auf_suffix_retention
                        ),
                        preserve_weight_mass=(
                            self.dpace_auf_preserve_weight_mass
                        ),
                    )
                )
            dpace_reference_weight_mass = loss_weights.sum(dim=(1, 2)).detach()
            # Normalize by the same effective continuation-aware token mass
            # used in the numerator.  This keeps base and selector terms on a
            # stable scale as anchor count and D-PACE weights vary.
            primary_loss_denominator = loss_weights.sum().clamp_min(1e-6)
            ce_loss_num = (neg_log_q * loss_weights).sum()
            tv_loss_num = ((1.0 - q) * loss_weights).sum()
            target_probability_num = (q.detach() * weight_mask).sum()
            loss = self._compose_token_objective(
                ce_loss_num,
                tv_loss_num,
                target_probability_num,
                weight_mask.sum(),
            ) / primary_loss_denominator
            if self.loss_type in _VCRD_LOSS_TYPES:
                effective_vcrd_alpha = self._effective_vcrd_alpha()
                vcrd_diagnostics = self._empty_vcrd_diagnostics(device)
                if self._should_run_vcrd(effective_vcrd_alpha):
                    vcrd_loss, vcrd_diagnostics = self._vcrd_objective(
                        base_logits=logits.view(
                            bsz, n_blocks, self.block_size, -1
                        ),
                        target_ids=target_ids,
                        base_q=q,
                        valid_mask=weight_mask,
                        input_ids=input_ids,
                        anchor_positions=anchor_positions,
                        block_keep_mask=block_keep_mask,
                        full_position_ids=full_position_ids,
                        hidden_states=hidden_states,
                        attention_mask=dflash_attn_mask,
                    )
                    loss = loss + effective_vcrd_alpha * vcrd_loss
        elif self.loss_type in _DAVCA_LOSS_TYPES:
            neg_log_q = loss_per_token.view_as(target_ids)
            loss, davca_diagnostics = self._davca_objective(
                logits.view(bsz, n_blocks, self.block_size, -1),
                target_ids,
                neg_log_q,
                weight_mask,
            )
        elif self.loss_type in _CVA_LOSS_TYPES:
            neg_log_q = loss_per_token.view_as(target_ids)
            loss, cva_diagnostics = self._cva_objective(
                logits.view(bsz, n_blocks, self.block_size, -1),
                target_ids,
                neg_log_q,
                weight_mask,
            )
        else:
            raise ValueError(f"unknown loss_type {self.loss_type!r}")

        # --- Sampled Path-Level Acceptance ---
        # Unlike another position-wise reweighting rule, this term compares
        # complete sampled verification paths.  It is intentionally kept as
        # a small auxiliary on top of D-PACE so token calibration remains the
        # stable primary objective.
        effective_path_acceptance_alpha = (
            self._effective_path_acceptance_alpha()
        )
        path_acceptance_diagnostics: Dict[str, torch.Tensor] = {}
        if effective_path_acceptance_alpha > 0:
            from specforge.core.acceptance_losses import (
                sampled_path_acceptance_objective,
            )

            path_valid_mask = (weight_mask > 0).int()
            path_valid_mask[:, :, 0] = 1
            path_valid_mask = path_valid_mask.cumprod(dim=-1).float()
            path_valid_mask[:, :, 0] = 0.0
            path_acceptance_loss, path_acceptance_diagnostics = (
                sampled_path_acceptance_objective(
                    logits=logits.view(
                        bsz, n_blocks, self.block_size, -1
                    ),
                    target_ids=target_ids,
                    valid_mask=path_valid_mask,
                    num_samples=self.path_acceptance_num_samples,
                    topk=self.path_acceptance_topk,
                    sampling_temperature=(
                        self.path_acceptance_sampling_temperature
                    ),
                    reward_temperature=(
                        self.path_acceptance_reward_temperature
                    ),
                )
            )
            loss = (
                loss
                + effective_path_acceptance_alpha * path_acceptance_loss
            )

        # --- Corrected CE loss (Markov Scaffold only, base detached) ---
        corrected_ce_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if scaffold_active and effective_scaffold_ce_alpha > 0:
            corrected_flat_logits = corrected_logits_4d.reshape(
                -1, corrected_logits_4d.size(-1)
            )
            corrected_loss_per_token = F.cross_entropy(
                corrected_flat_logits, flat_targets, reduction="none"
            )
            flat_weights = weight_mask.view(-1)
            valid_token_count = flat_weights.sum() + 1e-6
            corrected_ce_loss = (
                (corrected_loss_per_token * flat_weights).sum() / valid_token_count
            )
            loss = loss + effective_scaffold_ce_alpha * corrected_ce_loss

        # --- Scaffold-to-Base KD loss (backbone only, corrected detached) ---
        scaffold_kd_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        scaffold_kd_fraction = torch.tensor(
            0.0 if self.scaffold_kd_mode != "full" else 1.0,
            device=device,
            dtype=torch.float32,
        )
        scaffold_positive_advantage_fraction = torch.tensor(
            0.0, device=device, dtype=torch.float32
        )
        scaffold_mean_positive_advantage = torch.tensor(
            0.0, device=device, dtype=torch.float32
        )
        scaffold_first_rejection_fraction = torch.tensor(
            0.0, device=device, dtype=torch.float32
        )
        scaffold_positive_fraction_by_depth = torch.zeros(
            self.block_size, device=device, dtype=torch.float32
        )
        scaffold_mean_advantage_by_depth = torch.zeros(
            self.block_size, device=device, dtype=torch.float32
        )
        scaffold_first_rejection_by_depth = torch.zeros(
            self.block_size, device=device, dtype=torch.float32
        )
        if scaffold_active and effective_scaffold_kd_alpha > 0:
            # base_logits WITH gradient (trains backbone).
            # corrected side detached (no gradient to scaffold).
            base_logits_for_kd = logits.view(
                bsz, n_blocks, self.block_size, -1
            )
            kd_weight_mask = weight_mask
            if self.scaffold_kd_mode == "rescue":
                # Distil only useful scaffold corrections: base is wrong
                # while corrected is right.  The discrete mask is detached;
                # gradients still flow only into base logits through KL.
                kd_weight_mask = self._scaffold_rescue_kd_mask(
                    base_logits_for_kd,
                    corrected_logits_4d,
                    target_ids,
                    weight_mask,
                )
            elif self.scaffold_kd_mode in {
                "advantage",
                "first-rejection-advantage",
            }:
                kd_weight_mask, advantage_diag = (
                    self._scaffold_advantage_kd_mask(
                        base_logits_for_kd,
                        corrected_logits_4d,
                        target_ids,
                        weight_mask,
                        first_rejection=(
                            self.scaffold_kd_mode
                            == "first-rejection-advantage"
                        ),
                        global_floor=self.scaffold_advantage_global_floor,
                        continuation_power=(
                            self.scaffold_advantage_continuation_power
                        ),
                    )
                )
                scaffold_positive_advantage_fraction = advantage_diag[
                    "positive_fraction"
                ]
                scaffold_mean_positive_advantage = advantage_diag[
                    "mean_positive_advantage"
                ]
                scaffold_first_rejection_fraction = advantage_diag[
                    "first_rejection_fraction"
                ]
                scaffold_positive_fraction_by_depth = advantage_diag[
                    "positive_fraction_by_depth"
                ]
                scaffold_mean_advantage_by_depth = advantage_diag[
                    "mean_positive_advantage_by_depth"
                ]
                scaffold_first_rejection_by_depth = advantage_diag[
                    "first_rejection_fraction_by_depth"
                ]
            if self.scaffold_kd_mode != "full":
                scaffold_kd_fraction = (
                    kd_weight_mask.gt(0).float().sum()
                    / (weight_mask.gt(0).float().sum() + 1e-6)
                )
            scaffold_kd_loss, _kl_per_token = self._scaffold_kd_loss(
                base_logits_for_kd,
                corrected_logits_4d.detach(),
                kd_weight_mask,
                self.scaffold_temperature,
            )
            loss = (
                loss
                + effective_scaffold_kd_alpha * scaffold_kd_loss
            )

        loss_components: Dict[str, torch.Tensor] = {}

        # --- DFlash2 candidate-path selector ---
        # Serving and training use the same strict unary top-k. Candidate misses
        # belong to backbone recall and therefore carry no selector CE.
        candidate_selector = getattr(self.draft_model, "candidate_selector", None)
        if candidate_selector is not None and self.selector_loss_alpha > 0:
            # The draft model outputs one logit per produced token, i.e. an
            # (S,) sequence dimension that is a flat view of N x K blocks.
            # Operate on the block lattice [B, N, K, V] so the candidate
            # tensor and target_ids share the same leading block axes.
            selector_source_logits = logits.view(
                bsz, n_blocks, self.block_size, -1
            )
            (
                unary_logits,
                candidate_ids,
                target_candidate_index,
                target_is_candidate,
            ) = _selector_training_candidates(
                selector_source_logits,
                target_ids,
                candidate_selector.top_k,
            )

            predecessor_ids = torch.cat(
                [target_ids[:, :, :1], target_ids[:, :, :-1]], dim=-1
            )
            selector_logits = candidate_selector.score_candidates(
                candidate_ids=candidate_ids,
                unary_logits=unary_logits,
                hidden_states=output_hidden.view(
                    bsz, n_blocks, self.block_size, -1
                ),
                predecessor_ids=predecessor_ids,
            )
            selector_ce = F.cross_entropy(
                selector_logits.float().reshape(-1, selector_logits.shape[-1]),
                target_candidate_index.reshape(-1),
                reduction="none",
            ).reshape_as(target_ids)
            selector_probability = torch.exp(-selector_ce)
            selector_loss_weights = (
                loss_weights * target_is_candidate.float()
            )
            selector_metric_mask = (
                weight_mask * target_is_candidate.float()
            )
            selector_den = selector_loss_weights.sum().clamp_min(1e-6)
            # Selector remains a calibrated categorical CE regardless of the
            # base model's optional LK/TV objective.
            selector_loss_num = (selector_ce * selector_loss_weights).sum()
            selector_loss = selector_loss_num / primary_loss_denominator
            effective_selector_alpha = self._effective_selector_loss_alpha()
            loss = loss + effective_selector_alpha * selector_loss

            with torch.no_grad():
                selected_ids = candidate_ids.gather(
                    -1, selector_logits.argmax(dim=-1, keepdim=True)
                ).squeeze(-1)
                selector_accuracy = (
                    selected_ids.eq(target_ids).float()
                    * selector_loss_weights
                ).sum() / selector_den
                selector_coverage = (
                    target_is_candidate.float() * weight_mask
                ).sum() / weight_mask.sum().clamp_min(1e-6)
            loss_components.update(
                {
                    "selector_loss": selector_loss.detach(),
                    "selector_accuracy": selector_accuracy.detach(),
                    "selector_coverage": selector_coverage.detach(),
                    "selector_target_probability": (
                        (
                            selector_probability.detach()
                            * selector_metric_mask
                        ).sum()
                        / selector_metric_mask.sum().clamp_min(1e-6)
                    ).detach(),
                    "selector_loss_alpha": logits.new_tensor(
                        effective_selector_alpha, dtype=torch.float32
                    ),
                }
            )

        # --- Block Survival Loss (verification-native acceptance objective) ---
        if self.path_acceptance_alpha > 0:
            loss_components.update(path_acceptance_diagnostics)
            loss_components["path_acceptance_alpha"] = torch.tensor(
                effective_path_acceptance_alpha,
                device=device,
                dtype=torch.float32,
            )
            loss_components["path_acceptance_num_samples"] = torch.tensor(
                self.path_acceptance_num_samples,
                device=device,
                dtype=torch.float32,
            )
        if dpace_auf_diagnostics is not None:
            loss_components.update(dpace_auf_diagnostics)
            loss_components["dpace_auf_suffix_retention"] = torch.tensor(
                self.dpace_auf_suffix_retention,
                device=device,
                dtype=torch.float32,
            )
            loss_components["dpace_auf_preserve_weight_mass"] = torch.tensor(
                float(self.dpace_auf_preserve_weight_mass),
                device=device,
                dtype=torch.float32,
            )
        if self.loss_type in _DAVCA_LOSS_TYPES:
            loss_components.update(davca_diagnostics)
        if self.loss_type in _CVA_LOSS_TYPES:
            loss_components.update(cva_diagnostics)
        if self.loss_type in _VCRD_LOSS_TYPES:
            loss_components.update(vcrd_diagnostics)
            loss_components["vcrd_alpha"] = torch.tensor(
                effective_vcrd_alpha,
                device=device,
                dtype=torch.float32,
            )
            loss_components["vcrd_interval"] = torch.tensor(
                self.vcrd_interval,
                device=device,
                dtype=torch.float32,
            )
        if self.loss_type in _CFU_LOSS_TYPES:
            loss_components["cfu_loss"] = cfu_loss.detach()
            loss_components["cfu_alpha"] = torch.tensor(
                effective_cfu_alpha, device=device, dtype=torch.float32
            )
            loss_components["cfu_credit_mass"] = cfu_diagnostics[
                "credit_mass"
            ]
            loss_components["cfu_hard_frontier_count"] = cfu_diagnostics[
                "hard_frontier_count"
            ]
            loss_components["cfu_soft_frontier_mass"] = cfu_diagnostics[
                "soft_frontier_mass"
            ]
            loss_components["cfu_active_block_fraction"] = cfu_diagnostics[
                "active_block_fraction"
            ]
            loss_components["cfu_mean_margin"] = cfu_diagnostics[
                "mean_margin"
            ]
            loss_components["cfu_mean_frontier_margin"] = cfu_diagnostics[
                "mean_frontier_margin"
            ]
            loss_components["cfu_mean_frontier_continuation"] = (
                cfu_diagnostics["mean_frontier_continuation"]
            )
            loss_components["cfu_advantage_beta"] = torch.tensor(
                effective_cfu_advantage_beta,
                device=device,
                dtype=torch.float32,
            )
            loss_components["cfu_mean_observed_utility"] = cfu_diagnostics[
                "mean_observed_utility"
            ]
            loss_components["cfu_mean_expected_utility"] = cfu_diagnostics[
                "mean_expected_utility"
            ]
            loss_components["cfu_continuation_correlation"] = (
                cfu_diagnostics["continuation_correlation"]
            )
            loss_components["cfu_advantage_mean"] = cfu_diagnostics[
                "advantage_mean"
            ]
            loss_components["cfu_advantage_std"] = cfu_diagnostics[
                "advantage_std"
            ]
            loss_components["cfu_advantage_clip_fraction"] = (
                cfu_diagnostics["advantage_clip_fraction"]
            )
            loss_components["cfu_credit_effective_sample_size"] = (
                cfu_diagnostics["credit_effective_sample_size"]
            )
        loss_components["auxiliary_anneal_factor"] = torch.tensor(
            self._auxiliary_anneal_factor(),
            device=device,
            dtype=torch.float32,
        )
        loss_components["scaffold_effective_ce_alpha"] = torch.tensor(
            effective_scaffold_ce_alpha,
            device=device,
            dtype=torch.float32,
        )
        loss_components["scaffold_effective_kd_alpha"] = torch.tensor(
            effective_scaffold_kd_alpha,
            device=device,
            dtype=torch.float32,
        )
        loss_components["scaffold_vp_exposure"] = torch.tensor(
            self._effective_scaffold_vp_exposure(),
            device=device,
            dtype=torch.float32,
        )

        effective_survival_alpha = self._effective_survival_alpha()
        survival_schedule_active = self._auxiliary_anneal_factor() > 0
        if self.survival_loss_alpha > 0 and survival_schedule_active:

            # Reshape logits to block-structured form [B, N_blocks, K, V].
            # IMPORTANT: uses base logits (not corrected), since serving
            # removes the Markov scaffold.
            draft_logits_4d = logits.view(bsz, n_blocks, self.block_size, -1)

            # Construct survival_valid_mask: contiguous supervised prefix
            # (DSpark eval_mask pattern).  Position 0 is the anchor
            # (weight_mask=0); set it to 1 before cumprod so the anchor
            # doesn't kill the entire chain, then zero it out afterwards.
            prod_input = (weight_mask > 0).int()
            prod_input[:, :, 0] = 1  # anchor = multiplicative identity
            survival_valid_mask = prod_input.cumprod(dim=-1).float()
            survival_valid_mask[:, :, 0] = 0.0  # anchor never contributes

            from specforge.core.acceptance_losses import block_survival_loss

            survival_loss, survival_diag = block_survival_loss(
                logits=draft_logits_4d,
                target_ids=target_ids,
                valid_mask=survival_valid_mask,
                temperature=self.survival_temperature,
                hard_alive=self.survival_hard_alive,
                leaky_eta=self.survival_leaky_eta,
            )

            loss = loss + effective_survival_alpha * survival_loss

            loss_components["survival_loss"] = survival_loss.detach()
            loss_components["survival_score"] = (1.0 - survival_loss).detach()
            loss_components["survival_mean_margin"] = survival_diag[
                "mean_margin"
            ]
            loss_components["survival_hard_accept_rate"] = survival_diag[
                "mean_hard_accept"
            ]
            loss_components["survival_alpha"] = torch.tensor(
                effective_survival_alpha, device=device, dtype=torch.float32,
            )
        elif self.survival_loss_alpha > 0:
            # After late annealing, skip the vocabulary-wide margin work and
            # leave a zero alpha in logs so the pure-DFlash phase is explicit.
            loss_components["survival_alpha"] = torch.tensor(
                0.0, device=device, dtype=torch.float32,
            )

        # --- First-Rejection Continuation Loss / Deep Full-Accept Preservation ---
        # DFAP can run as a standalone auxiliary for clean ablations.  The
        # shared objective call also provides its live verification margin and
        # detached hard-accept mask when FRC itself is disabled.
        if (
            self.first_rejection_loss_alpha > 0
            or self.first_rejection_deep_anchor_alpha > 0
        ):
            primary_loss = loss.detach()
            first_rejection_valid_mask = (weight_mask > 0).int()
            first_rejection_valid_mask[:, :, 0] = 1
            first_rejection_valid_mask = first_rejection_valid_mask.cumprod(
                dim=-1
            ).float()
            first_rejection_valid_mask[:, :, 0] = 0.0

            from specforge.core.acceptance_losses import (
                first_rejection_continuation_objective,
            )

            first_rejection_reference_mass = dpace_reference_weight_mass
            if (
                self.first_rejection_reduction == "dpace-matched"
                and self.first_rejection_max_depth > 0
            ):
                # Match only the D-PACE mass in the same shallow repair
                # window.  Matching the entire chain onto depths 1..M would
                # over-concentrate the residual gradient and undermine the
                # deep-chain protection below.
                repair_end = min(
                    self.first_rejection_max_depth + 1,
                    loss_weights.size(-1),
                )
                first_rejection_reference_mass = loss_weights[
                    :, :, 1:repair_end
                ].sum(dim=(1, 2)).detach()

            (
                first_rejection_loss,
                first_rejection_margin_live,
                first_rejection_diag,
            ) = (
                first_rejection_continuation_objective(
                    logits=logits.view(bsz, n_blocks, self.block_size, -1),
                    target_ids=target_ids,
                    valid_mask=first_rejection_valid_mask,
                    margin=self.first_rejection_margin,
                    temperature=self.first_rejection_temperature,
                    continuation_power=self.first_rejection_continuation_power,
                    reduction=self.first_rejection_reduction,
                    reference_weight_mass=first_rejection_reference_mass,
                    mass_match_min=self.first_rejection_mass_match_min,
                    mass_match_max=self.first_rejection_mass_match_max,
                    max_depth=self.first_rejection_max_depth,
                    adaptive_survival_rate=(
                        self.first_rejection_survival_ema.detach()
                        if self.first_rejection_adaptive_survival
                        else None
                    ),
                    adaptive_gamma=self.first_rejection_survival_gamma,
                    adaptive_margin_temperature=(
                        self.first_rejection_adaptive_margin_temperature
                    ),
                    adaptive_scale_min=self.first_rejection_adaptive_scale_min,
                    adaptive_scale_max=self.first_rejection_adaptive_scale_max,
                )
            )
            survival_batch_rate = logits.new_zeros((self.block_size,))
            if (
                self.first_rejection_loss_alpha > 0
                and self.first_rejection_adaptive_survival
                and self.training
            ):
                survival_batch_rate = self._update_first_rejection_survival_ema(
                    first_rejection_diag["hard_accept"],
                    first_rejection_valid_mask,
                )
            effective_first_rejection_alpha = (
                self._effective_first_rejection_alpha()
            )
            weighted_first_rejection_live = (
                effective_first_rejection_alpha * first_rejection_loss
            )
            first_rejection_grad_cosine = logits.new_zeros(())
            first_rejection_gradient_conflict = logits.new_zeros(())

            # Deep Full-Accept Preservation (DFAP): FRC repairs only rejected
            # frontiers, so its shared-parameter update can accidentally erode
            # already-good long chains.  Reuse the live margin and detached
            # D-PACE weights to keep those full-accept blocks above a safety
            # margin.  It shares the FRC warmup/ramp ratios but does not
            # require FRC to be enabled.
            deep_anchor_loss = logits.new_zeros((), dtype=torch.float32)
            deep_anchor_diag = {
                "protected_block_count": logits.new_zeros(()),
                "protected_token_count": logits.new_zeros(()),
                "eligible_block_count": logits.new_zeros(()),
                "protected_block_fraction": logits.new_zeros(()),
                "protected_weight_mass": logits.new_zeros(()),
                "protected_mean_margin": logits.new_zeros(()),
            }
            effective_deep_anchor_alpha = 0.0
            deep_anchor_margin_floor = self.first_rejection_deep_anchor_margin
            deep_anchor_batch_floor = logits.new_full(
                (self.block_size,), self.first_rejection_deep_anchor_margin
            )
            deep_anchor_floor_count = logits.new_zeros((self.block_size,))
            if self.first_rejection_deep_anchor_alpha > 0:
                from specforge.core.acceptance_losses import (
                    deep_chain_dpace_anchor_loss,
                )

                effective_deep_anchor_alpha = (
                    self._effective_deep_anchor_alpha()
                )
                if self.first_rejection_deep_anchor_quantile > 0:
                    if self.training:
                        (
                            deep_anchor_batch_floor,
                            deep_anchor_floor_count,
                        ) = self._update_deep_anchor_margin_ema(
                            target_margin=first_rejection_margin_live,
                            valid_mask=first_rejection_valid_mask,
                            hard_accept=first_rejection_diag["hard_accept"],
                        )
                    deep_anchor_margin_floor = (
                        self.deep_anchor_margin_ema.detach()
                    )
                deep_anchor_loss, deep_anchor_diag = (
                    deep_chain_dpace_anchor_loss(
                        target_margin=first_rejection_margin_live,
                        dpace_weights=loss_weights.detach(),
                        valid_mask=first_rejection_valid_mask,
                        hard_accept=first_rejection_diag["hard_accept"],
                        min_depth=self.first_rejection_deep_anchor_min_depth,
                        margin_floor=deep_anchor_margin_floor,
                        temperature=(
                            self.first_rejection_deep_anchor_temperature
                        ),
                    )
                )
            weighted_deep_anchor_live = (
                effective_deep_anchor_alpha * deep_anchor_loss
            )

            # The historical path projects FRC only and adds DFAP normally.
            # A positive gradient budget activates joint FRC+DFAP surgery at
            # the compact verification-margin interface, then caps the joint
            # auxiliary norm relative to the D-PACE margin proxy.
            auxiliary_budget_scale = logits.new_ones(())
            auxiliary_grad_norm = logits.new_zeros(())
            primary_margin_grad_norm = logits.new_zeros(())
            gradient_context_active = (
                self.training
                and torch.is_grad_enabled()
                and effective_first_rejection_alpha > 0
            )
            primary_margin_proxy = None
            primary_grad = None
            if gradient_context_active and (
                self.first_rejection_gradient_mode == "margin-pcgrad"
                or self.first_rejection_auxiliary_gradient_budget > 0
            ):
                primary_margin_proxy = (
                    F.softplus(
                        -first_rejection_margin_live
                        / self.first_rejection_temperature
                    )
                    * loss_weights
                ).sum() / float(bsz)
                primary_grad = torch.autograd.grad(
                    primary_margin_proxy,
                    first_rejection_margin_live,
                    retain_graph=True,
                )[0].detach()

            if (
                gradient_context_active
                and self.first_rejection_auxiliary_gradient_budget > 0
            ):
                combined_auxiliary_live = (
                    weighted_first_rejection_live + weighted_deep_anchor_live
                )
                auxiliary_grad = torch.autograd.grad(
                    combined_auxiliary_live,
                    first_rejection_margin_live,
                    retain_graph=True,
                )[0].detach()
                from specforge.core.acceptance_losses import (
                    project_conflicting_gradient,
                )

                if self.first_rejection_gradient_mode == "margin-pcgrad":
                    (
                        auxiliary_grad,
                        first_rejection_grad_cosine,
                        first_rejection_gradient_conflict,
                    ) = project_conflicting_gradient(primary_grad, auxiliary_grad)
                else:
                    primary_fp32 = primary_grad.float()
                    auxiliary_fp32 = auxiliary_grad.float()
                    dot = (primary_fp32 * auxiliary_fp32).sum()
                    first_rejection_grad_cosine = dot / (
                        primary_fp32.square().sum().sqrt()
                        * auxiliary_fp32.square().sum().sqrt()
                    ).clamp_min(1e-12)
                    first_rejection_gradient_conflict = dot.lt(0).float()

                primary_margin_grad_norm = primary_grad.float().norm()
                auxiliary_grad_norm = auxiliary_grad.float().norm()
                auxiliary_budget_scale = torch.clamp(
                    self.first_rejection_auxiliary_gradient_budget
                    * primary_margin_grad_norm
                    / auxiliary_grad_norm.clamp_min(1e-12),
                    max=1.0,
                )
                budgeted_auxiliary_grad = (
                    auxiliary_grad * auxiliary_budget_scale
                )
                gradient_surrogate = (
                    first_rejection_margin_live * budgeted_auxiliary_grad
                ).sum() - (
                    first_rejection_margin_live.detach()
                    * budgeted_auxiliary_grad
                ).sum()
                loss = loss + combined_auxiliary_live.detach() + gradient_surrogate
            elif (
                gradient_context_active
                and self.first_rejection_gradient_mode == "margin-pcgrad"
            ):
                auxiliary_grad = torch.autograd.grad(
                    weighted_first_rejection_live,
                    first_rejection_margin_live,
                    retain_graph=True,
                )[0].detach()
                from specforge.core.acceptance_losses import (
                    project_conflicting_gradient,
                )

                (
                    projected_auxiliary_grad,
                    first_rejection_grad_cosine,
                    first_rejection_gradient_conflict,
                ) = project_conflicting_gradient(primary_grad, auxiliary_grad)
                gradient_surrogate = (
                    first_rejection_margin_live * projected_auxiliary_grad
                ).sum() - (
                    first_rejection_margin_live.detach()
                    * projected_auxiliary_grad
                ).sum()
                loss = (
                    loss
                    + weighted_first_rejection_live.detach()
                    + gradient_surrogate
                    + weighted_deep_anchor_live
                )
            else:
                loss = (
                    loss
                    + weighted_first_rejection_live
                    + weighted_deep_anchor_live
                )
            loss_components["first_rejection_loss"] = (
                first_rejection_loss.detach()
            )
            loss_components["first_rejection_alpha"] = torch.tensor(
                effective_first_rejection_alpha,
                device=device,
                dtype=torch.float32,
            )
            loss_components["first_rejection_count"] = first_rejection_diag[
                "frontier_count"
            ]
            loss_components["first_rejection_mean_margin"] = (
                first_rejection_diag["mean_frontier_margin"]
            )
            loss_components["first_rejection_mean_continuation"] = (
                first_rejection_diag["mean_continuation"]
            )
            loss_components["primary_loss"] = primary_loss
            loss_components["dpace_weight_mass_per_sample"] = (
                dpace_reference_weight_mass.mean()
                if dpace_reference_weight_mass is not None
                else torch.tensor(0.0, device=device)
            )
            loss_components["first_rejection_credit_mass"] = (
                first_rejection_diag["credit_mass"]
            )
            loss_components["first_rejection_credit_mass_per_sample"] = (
                first_rejection_diag["credit_mass_per_sample"]
            )
            loss_components["first_rejection_mass_scale_mean"] = (
                first_rejection_diag["mass_scale_mean"]
            )
            loss_components["first_rejection_mass_scale_max"] = (
                first_rejection_diag["mass_scale_max"]
            )
            loss_components["first_rejection_normalized_loss"] = (
                first_rejection_diag["normalized_loss"]
            )
            weighted_first_rejection = weighted_first_rejection_live.detach()
            loss_components["first_rejection_weighted_contribution"] = (
                weighted_first_rejection
            )
            loss_components["first_rejection_to_primary_ratio"] = (
                weighted_first_rejection / primary_loss.abs().clamp_min(1e-6)
            )
            loss_components["first_rejection_grad_cosine"] = (
                first_rejection_grad_cosine.detach()
            )
            loss_components["first_rejection_gradient_conflict"] = (
                first_rejection_gradient_conflict.detach()
            )
            loss_components["first_rejection_max_depth"] = torch.tensor(
                self.first_rejection_max_depth,
                device=device,
                dtype=torch.float32,
            )
            loss_components["first_rejection_adaptive_scale_mean"] = (
                first_rejection_diag.get(
                    "adaptive_scale_mean", logits.new_ones(())
                )
            )
            loss_components["first_rejection_adaptive_scale_min"] = (
                first_rejection_diag.get(
                    "adaptive_scale_min", logits.new_ones(())
                )
            )
            loss_components["first_rejection_adaptive_scale_max"] = (
                first_rejection_diag.get(
                    "adaptive_scale_max", logits.new_ones(())
                )
            )
            loss_components["first_rejection_survival_ema_mean"] = (
                self.first_rejection_survival_ema[1:].mean().detach()
            )
            loss_components["first_rejection_survival_batch_mean"] = (
                survival_batch_rate[1:].mean().detach()
            )
            loss_components["deep_anchor_loss"] = deep_anchor_loss.detach()
            loss_components["deep_anchor_alpha"] = torch.tensor(
                effective_deep_anchor_alpha,
                device=device,
                dtype=torch.float32,
            )
            loss_components["deep_anchor_block_count"] = deep_anchor_diag[
                "protected_block_count"
            ]
            loss_components["deep_anchor_token_count"] = deep_anchor_diag[
                "protected_token_count"
            ]
            loss_components["deep_anchor_eligible_block_count"] = (
                deep_anchor_diag["eligible_block_count"]
            )
            loss_components["deep_anchor_block_fraction"] = deep_anchor_diag[
                "protected_block_fraction"
            ]
            loss_components["deep_anchor_weight_mass"] = deep_anchor_diag[
                "protected_weight_mass"
            ]
            loss_components["deep_anchor_mean_margin"] = deep_anchor_diag[
                "protected_mean_margin"
            ]
            loss_components["deep_anchor_margin_floor_mean"] = (
                self.deep_anchor_margin_ema[1:].mean().detach()
                if self.first_rejection_deep_anchor_quantile > 0
                else logits.new_tensor(self.first_rejection_deep_anchor_margin)
            )
            loss_components["deep_anchor_batch_floor_mean"] = (
                deep_anchor_batch_floor[1:].mean().detach()
            )
            loss_components["deep_anchor_floor_observation_count"] = (
                deep_anchor_floor_count.sum().detach()
            )
            weighted_deep_anchor = (
                effective_deep_anchor_alpha * deep_anchor_loss.detach()
            )
            loss_components["deep_anchor_weighted_contribution"] = (
                weighted_deep_anchor
            )
            loss_components["deep_anchor_to_primary_ratio"] = (
                weighted_deep_anchor / primary_loss.abs().clamp_min(1e-6)
            )
            loss_components["auxiliary_gradient_budget"] = torch.tensor(
                self.first_rejection_auxiliary_gradient_budget,
                device=device,
                dtype=torch.float32,
            )
            loss_components["auxiliary_gradient_budget_scale"] = (
                auxiliary_budget_scale.detach()
            )
            loss_components["auxiliary_margin_grad_norm"] = (
                auxiliary_grad_norm.detach()
            )
            loss_components["primary_margin_grad_norm"] = (
                primary_margin_grad_norm.detach()
            )

        # --- First-Rejection Boundary Optimization (FRBO) ---
        # FRBO is intentionally independent of D-PACE.  The four-machine
        # study uses loss_type=dflash, whose primary CE is normalized by valid
        # token mass.  Auxiliary losses are EMA-normalized to that primary
        # scale so their coefficients remain interpretable across batches.
        if self.frbo_survival_alpha > 0 or self.frbo_boundary_alpha > 0:
            from specforge.core.acceptance_losses import frbo_objective

            frbo_primary_loss = loss
            frbo_valid_mask = (weight_mask > 0).int()
            frbo_valid_mask[:, :, 0] = 1
            frbo_valid_mask = frbo_valid_mask.cumprod(dim=-1).float()
            frbo_valid_mask[:, :, 0] = 0.0
            frbo_logits = logits.view(bsz, n_blocks, self.block_size, -1)

            zero = logits.new_zeros((), dtype=torch.float32)
            (
                frbo_survival_loss,
                frbo_boundary_loss,
                frbo_margin,
                frbo_diag,
            ) = frbo_objective(
                logits=frbo_logits,
                target_ids=target_ids,
                valid_mask=frbo_valid_mask,
                temperature=self.frbo_temperature,
                focal_gamma=self.frbo_boundary_gamma,
            )

            # Initialize each EMA from its first observed value, then update
            # once per training forward.  All updates are detached buffers and
            # therefore never become part of the optimization graph.
            if self.training:
                with torch.no_grad():
                    first_update = self.frbo_ema_updates.eq(0)
                    decay = self.frbo_ema_decay
                    primary_value = frbo_primary_loss.detach().float().abs()
                    self.frbo_primary_ema.copy_(
                        torch.where(
                            first_update,
                            primary_value,
                            decay * self.frbo_primary_ema
                            + (1.0 - decay) * primary_value,
                        )
                    )
                    if self.frbo_survival_alpha > 0:
                        survival_value = (
                            frbo_survival_loss.detach().float().abs()
                        )
                        self.frbo_survival_ema.copy_(
                            torch.where(
                                first_update,
                                survival_value,
                                decay * self.frbo_survival_ema
                                + (1.0 - decay) * survival_value,
                            )
                        )
                    if self.frbo_boundary_alpha > 0:
                        boundary_value = (
                            frbo_boundary_loss.detach().float().abs()
                        )
                        self.frbo_boundary_ema.copy_(
                            torch.where(
                                first_update,
                                boundary_value,
                                decay * self.frbo_boundary_ema
                                + (1.0 - decay) * boundary_value,
                            )
                        )
                    self.frbo_ema_updates.add_(1)

            primary_scale = self.frbo_primary_ema.detach().clamp_min(1e-6)
            survival_scale = torch.ones_like(primary_scale)
            boundary_scale = torch.ones_like(primary_scale)
            if self.frbo_survival_alpha > 0:
                survival_scale = (
                    primary_scale
                    / self.frbo_survival_ema.detach().clamp_min(1e-6)
                ).clamp(self.frbo_scale_min, self.frbo_scale_max)
            if self.frbo_boundary_alpha > 0:
                boundary_scale = (
                    primary_scale
                    / self.frbo_boundary_ema.detach().clamp_min(1e-6)
                ).clamp(self.frbo_scale_min, self.frbo_scale_max)

            frbo_ramp = self._effective_frbo_ramp()
            effective_frbo_survival_alpha = (
                self.frbo_survival_alpha * frbo_ramp
            )
            effective_frbo_boundary_alpha = (
                self.frbo_boundary_alpha * frbo_ramp
            )
            frbo_aux_loss = (
                effective_frbo_survival_alpha
                * survival_scale
                * frbo_survival_loss
                + effective_frbo_boundary_alpha
                * boundary_scale
                * frbo_boundary_loss
            )

            frbo_grad_cosine = zero
            frbo_conflict = zero
            if (
                self.frbo_gradient_mode == "margin-pcgrad"
                and self.training
                and torch.is_grad_enabled()
                and frbo_ramp > 0
            ):
                primary_margin_proxy = (
                    F.softplus(-frbo_margin / self.frbo_temperature)
                    * frbo_valid_mask
                ).sum() / frbo_valid_mask.sum().clamp_min(1.0)
                primary_grad = torch.autograd.grad(
                    primary_margin_proxy, frbo_margin, retain_graph=True
                )[0].detach()
                auxiliary_grad = torch.autograd.grad(
                    frbo_aux_loss, frbo_margin, retain_graph=True
                )[0].detach()
                from specforge.core.acceptance_losses import (
                    project_conflicting_gradient,
                )

                (
                    projected_auxiliary_grad,
                    frbo_grad_cosine,
                    frbo_conflict,
                ) = project_conflicting_gradient(primary_grad, auxiliary_grad)
                # Zero-valued surrogate with the projected verification-margin
                # gradient.  The compact margin interface avoids retaining two
                # vocabulary-sized gradient tensors.
                gradient_surrogate = (
                    frbo_margin * projected_auxiliary_grad
                ).sum() - (
                    frbo_margin.detach() * projected_auxiliary_grad
                ).sum()
                loss = (
                    frbo_primary_loss
                    + frbo_aux_loss.detach()
                    + gradient_surrogate
                )
            else:
                loss = frbo_primary_loss + frbo_aux_loss

            loss_components["frbo_primary_loss"] = frbo_primary_loss.detach()
            loss_components["frbo_survival_loss"] = frbo_survival_loss.detach()
            loss_components["frbo_boundary_loss"] = frbo_boundary_loss.detach()
            loss_components["frbo_survival_scale"] = survival_scale.detach()
            loss_components["frbo_boundary_scale"] = boundary_scale.detach()
            loss_components["frbo_survival_alpha"] = torch.tensor(
                effective_frbo_survival_alpha, device=device, dtype=torch.float32
            )
            loss_components["frbo_boundary_alpha"] = torch.tensor(
                effective_frbo_boundary_alpha, device=device, dtype=torch.float32
            )
            loss_components["frbo_aux_to_primary_ratio"] = (
                frbo_aux_loss.detach().abs()
                / frbo_primary_loss.detach().abs().clamp_min(1e-6)
            )
            loss_components["frbo_grad_cosine"] = frbo_grad_cosine.detach()
            loss_components["frbo_gradient_conflict"] = frbo_conflict.detach()
            if self.frbo_survival_alpha > 0:
                loss_components["frbo_soft_accept_length"] = (
                    frbo_diag["soft_accept_length"]
                )
                loss_components["frbo_hard_accept_length"] = (
                    frbo_diag["hard_accept_length"]
                )
            if self.frbo_boundary_alpha > 0:
                loss_components["frbo_boundary_credit_mass"] = (
                    frbo_diag["boundary_credit_mass"]
                )
                loss_components["frbo_mean_soft_accept"] = (
                    frbo_diag["mean_soft_accept"]
                )

        # --- Accuracy & Scaffold Diagnostics ---
        with torch.no_grad():
            # Base accuracy (always computed).
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count
            base_accuracy = accuracy.clone()

            base_pred_4d = logits.view(
                bsz, n_blocks, self.block_size, -1
            ).argmax(dim=-1)
            valid_pos = weight_mask > 0.5
            base_correct_4d = (base_pred_4d == target_ids) & valid_pos
            accepted_or_invalid = base_correct_4d | (~valid_pos)
            alive_before = torch.ones_like(valid_pos)
            alive_before[..., 1:] = (
                accepted_or_invalid[..., :-1].int().cumprod(dim=-1).bool()
            )
            base_first_rejection = valid_pos & alive_before & (~base_correct_4d)
            if self.acceptance_replay_signals:
                survived = accepted_or_invalid.int().cumprod(dim=-1).bool()
                active_block = valid_pos.any(dim=-1)
                accepted_depth = (survived & valid_pos).float().sum(dim=-1)
                full_accept = active_block & accepted_or_invalid.all(dim=-1)
                rejected = active_block & (~full_accept)
                first_rejection_depth = accepted_depth + 1.0
                boundary_hard = rejected & first_rejection_depth.le(3.0)
                deep_hard = rejected & first_rejection_depth.gt(3.0)
                block_count = active_block.float().sum(dim=-1).clamp_min(1.0)
                self._last_acceptance_replay_signals = torch.stack(
                    [
                        boundary_hard.float().sum(dim=-1) / block_count,
                        deep_hard.float().sum(dim=-1) / block_count,
                        full_accept.float().sum(dim=-1) / block_count,
                    ],
                    dim=-1,
                ).detach()
            loss_components["base_accuracy"] = base_accuracy.detach()
            for depth in range(1, self.block_size):
                valid_d = valid_pos[..., depth]
                count_d = valid_d.float().sum() + 1e-6
                loss_components[f"base_accuracy_depth_{depth}"] = (
                    base_correct_4d[..., depth].float().sum() / count_d
                ).detach()
                loss_components[f"first_rejection_rate_depth_{depth}"] = (
                    base_first_rejection[..., depth].float().sum() / count_d
                ).detach()

            # Scaffold diagnostics.
            corrected_accuracy = torch.tensor(0.0, device=device)
            base_corrected_agreement = torch.tensor(0.0, device=device)
            rescue_rate = torch.tensor(0.0, device=device)
            harm_rate = torch.tensor(0.0, device=device)
            markov_bias_rms = torch.tensor(0.0, device=device)
            markov_bias_abs_max = torch.tensor(0.0, device=device)
            base_logits_rms = torch.tensor(0.0, device=device)
            markov_to_base_ratio = torch.tensor(0.0, device=device)

            if scaffold_active:
                # Corrected accuracy.
                corrected_pred_ids = corrected_logits_4d.argmax(dim=-1).view(-1)
                corrected_correct = (
                    (corrected_pred_ids == flat_targets)
                    & (binary_eval_mask > 0.5)
                )
                corrected_accuracy = (
                    corrected_correct.sum().float() / actual_token_count
                )

                # Top-1 agreement.
                corrected_pred_4d = corrected_logits_4d.argmax(dim=-1)
                agreement = (base_pred_4d == corrected_pred_4d) & valid_pos
                corrected_correct_4d = (corrected_pred_4d == target_ids) & valid_pos
                valid_count = valid_pos.sum().float() + 1e-6

                base_corrected_agreement = agreement.sum().float() / valid_count

                # Rescue: base wrong, corrected right.
                rescued = (~base_correct_4d) & corrected_correct_4d
                rescue_rate = rescued.sum().float() / valid_count

                # Harm: base right, corrected wrong.
                harmed = base_correct_4d & (~corrected_correct_4d)
                harm_rate = harmed.sum().float() / valid_count

                # Markov bias statistics.
                if markov_bias_4d is not None:
                    bias_valid = markov_bias_4d[valid_pos].float()
                    base_valid = (
                        logits.view(bsz, n_blocks, self.block_size, -1)[valid_pos]
                        .float()
                    )
                    markov_bias_rms = bias_valid.pow(2).mean().sqrt()
                    markov_bias_abs_max = bias_valid.abs().max()
                    base_logits_rms = base_valid.pow(2).mean().sqrt()
                    markov_to_base_ratio = markov_bias_rms / (
                        base_logits_rms + 1e-6
                    )

            # Scaffold-specific loss components.
            if scaffold_active:
                loss_components["scaffold_corrected_ce"] = corrected_ce_loss.detach()
                loss_components["scaffold_distill_loss"] = scaffold_kd_loss.detach()
                loss_components["scaffold_kd_token_fraction"] = (
                    scaffold_kd_fraction.detach()
                )
                loss_components["scaffold_positive_advantage_fraction"] = (
                    scaffold_positive_advantage_fraction.detach()
                )
                loss_components["scaffold_mean_positive_advantage"] = (
                    scaffold_mean_positive_advantage.detach()
                )
                loss_components["scaffold_first_rejection_fraction"] = (
                    scaffold_first_rejection_fraction.detach()
                )
                loss_components["scaffold_prefix_len"] = (
                    scaffold_prefix_lengths.float().mean().detach()
                    if scaffold_prefix_lengths is not None
                    else torch.tensor(0.0, device=device)
                )
                # AR ratio: fraction of positions using on-policy predecessor.
                if self.scaffold_on_policy and scaffold_prefix_lengths is not None:
                    pos_range = torch.arange(
                        self.block_size, device=device
                    ).view(1, 1, -1)
                    is_ar_pos = pos_range >= scaffold_prefix_lengths.unsqueeze(-1)
                    valid_ar = is_ar_pos & (weight_mask > 0.5)
                    loss_components["scaffold_ar_ratio"] = (
                        valid_ar.sum().float()
                        / (valid_pos.sum().float() + 1e-6)
                    ).detach()
                else:
                    loss_components["scaffold_ar_ratio"] = torch.tensor(
                        0.0, device=device
                    )

                loss_components["corrected_accuracy"] = (
                    corrected_accuracy.detach()
                )
                loss_components["base_corrected_top1_agreement"] = (
                    base_corrected_agreement.detach()
                )
                loss_components["scaffold_rescue_rate"] = rescue_rate.detach()
                loss_components["scaffold_harm_rate"] = harm_rate.detach()
                loss_components["markov_bias_rms"] = markov_bias_rms.detach()
                loss_components["markov_bias_abs_max"] = (
                    markov_bias_abs_max.detach()
                )
                loss_components["base_logits_rms"] = base_logits_rms.detach()
                loss_components["markov_to_base_ratio"] = (
                    markov_to_base_ratio.detach()
                )

                # Per-depth diagnostics expose where the scaffold helps.
                for depth in range(1, self.block_size):
                    valid_d = valid_pos[..., depth]
                    count_d = valid_d.float().sum() + 1e-6
                    loss_components[f"corrected_accuracy_depth_{depth}"] = (
                        corrected_correct_4d[..., depth].float().sum() / count_d
                    ).detach()
                    loss_components[f"scaffold_rescue_rate_depth_{depth}"] = (
                        rescued[..., depth].float().sum() / count_d
                    ).detach()
                    loss_components[f"scaffold_harm_rate_depth_{depth}"] = (
                        harmed[..., depth].float().sum() / count_d
                    ).detach()
                    loss_components[
                        f"scaffold_positive_advantage_fraction_depth_{depth}"
                    ] = scaffold_positive_fraction_by_depth[depth].detach()
                    loss_components[
                        f"scaffold_mean_positive_advantage_depth_{depth}"
                    ] = scaffold_mean_advantage_by_depth[depth].detach()
                    loss_components[
                        f"scaffold_adv_first_rejection_rate_depth_{depth}"
                    ] = scaffold_first_rejection_by_depth[depth].detach()

                # Report corrected accuracy for progress-bar display.
                accuracy = corrected_accuracy

        return loss, accuracy, loss_components
