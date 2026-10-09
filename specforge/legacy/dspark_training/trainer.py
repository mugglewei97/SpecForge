"""FSDP training loop for the original DSpark recipe."""

import argparse
import contextlib
import functools
import logging
import math
import os
import time
import warnings

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset
from specforge.bv_loss import validate_bv_args, validate_bv_resume
from specforge.core.dspark import OnlineDSparkModel
from specforge.core.multi_teacher_oracle import (
    MultiTeacherOracleCache,
    MultiTeacherTrajectoryWriter,
)
from specforge.data.preprocessing import build_eagle3_dataset
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.fixed_prefix_reference import (
    FixedPrefixReference,
    validate_reference_args,
)
from specforge.legacy.checkpoint import get_last_checkpoint
from specforge.legacy.data_utils import prepare_dp_dataloaders
from specforge.legacy.dspark import DSparkDraftModel
from specforge.legacy.optimizer import (
    build_bf16_optimizer,
    load_fsdp_optimizer_state,
    save_fsdp_optimizer_state,
    validate_optimizer_resume,
)
from specforge.microbatch import (
    forward_backward_microbatches,
    validate_microbatch_resume,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.muon import configure_fsdp_optimizer_precision, validate_muon_options
from specforge.netprefix import (
    GreedyPrefixHook,
    NetPrefixRuntime,
    restore_rng,
    rng_state,
    validate_netprefix_args,
)
from specforge.netprefix_stream import TrainingStreamSource
from specforge.predictive_auxiliary import (
    prepare_semantic_codebook,
    validate_predictive_auxiliary,
    validate_predictive_resume,
)
from specforge.tracker import create_tracker
from specforge.utils import get_local_device, print_on_rank0, print_with_rank

from .checkpoint import save_checkpoint
from .initialization import (
    initialize_draft_weights,
    prepare_warm_start,
    resolve_mask_token_id,
)
from .objective import (
    configure_objective,
    validate_objective_args,
    validate_objective_resume,
)
from .setup import (
    _apply_trainable_parameter_scope,
    _build_dialogue_token_class_lookup,
    _load_parallel_refiner_teacher,
    _patch_flex_attention_triton_backend,
    build_dataloader,
    build_models,
)


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    components: dict,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()
        if hasattr(optimizer, "get_learning_rates"):
            for name, value in optimizer.get_learning_rates().items():
                logdict[f"train/lr_{name}"] = value

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    for key, value in components.items():
        logdict[f"{mode}/{key}"] = value

    comp_str = " ".join(f"{k}={v:.4f}" for k, v in components.items())
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}"
        f"[{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?],"
        f" Loss: {loss:.4f}, Acc: {accuracy:.4f}, {comp_str}"
    )

    tracker.log(logdict, step=global_step)


def _forward_dspark_data_batch(
    data,
    args,
    device,
    target_model,
    dspark_model,
    needs_target_hidden,
    global_step,
    micro_index=0,
):
    # Slice on CPU before calling this function: neither target nor drafter
    # materializes the whole logical batch on GPU in microbatch mode.
    input_ids = data["input_ids"].to(device, non_blocking=True)
    attention_mask = data["attention_mask"].to(device, non_blocking=True)
    loss_mask = data["loss_mask"].to(device, non_blocking=True)
    sample_ids = data["sample_id"].to(device, non_blocking=True)
    source_ids = data["source_id"].to(device, non_blocking=True)
    rollout_seed = (
        args.seed * 1_000_003
        + global_step
        + dist.get_rank() * 10_000_019
        + micro_index * 100_000_007
    )
    if args.multi_teacher_oracle_export_dir or args.multi_teacher_oracle_cache:
        torch.manual_seed(rollout_seed)
    target_output = target_model.generate_dflash_data(
        input_ids, attention_mask, loss_mask
    )
    hidden_states = target_output.hidden_states.to(device, non_blocking=True)
    last_hidden_states = target_output.last_hidden_states
    if last_hidden_states is not None:
        last_hidden_states = last_hidden_states.to(device, non_blocking=True)
    elif needs_target_hidden:
        raise RuntimeError(
            "The selected DSpark objective requires target final hidden states, "
            "but the target backend "
            f"({args.target_model_backend}) did not surface last_hidden_states. "
            "Use a backend that returns final hidden states, such as "
            "--target-model-backend hf."
        )
    if args.step_seeded_rollouts:
        torch.manual_seed(rollout_seed)
    forward_context = (
        torch.no_grad()
        if args.multi_teacher_oracle_export_dir
        else contextlib.nullcontext()
    )
    with forward_context:
        return dspark_model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            last_hidden_states=last_hidden_states,
            attention_mask=attention_mask,
            sample_ids=sample_ids,
            source_ids=source_ids,
        )


def run_training(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    validate_objective_args(args)
    validate_bv_args(args)
    validate_netprefix_args(args)
    validate_reference_args(args)
    if args.on_policy_start_step is not None and args.on_policy_start_step < 1:
        raise ValueError("--on-policy-start-step must be positive")
    if args.on_policy_ramp_steps < 0:
        raise ValueError("--on-policy-ramp-steps must be non-negative")
    if not 0 <= args.carh_sampled_prefix_memory_start_ratio <= 1:
        raise ValueError("--carh-sampled-prefix-memory-start-ratio must be in [0, 1]")
    if args.carh_sampled_prefix_memory_ramp_ratio < 0:
        raise ValueError("--carh-sampled-prefix-memory-ramp-ratio must be non-negative")
    if (
        args.carh_sampled_prefix_memory_start_ratio
        + args.carh_sampled_prefix_memory_ramp_ratio
        > 1
    ):
        raise ValueError("sampled-prefix memory start+ramp ratios must be <= 1")
    if args.micro_batch_size < 0 or args.micro_batch_size > args.batch_size:
        raise ValueError(
            "micro-batch-size must be 0 (disabled) or between 1 and batch-size"
        )
    if args.micro_batch_size:
        if args.accumulation_steps != 1:
            raise ValueError(
                "Physical microbatch splitting requires accumulation-steps=1"
            )
        if args.multi_teacher_oracle_export_dir or args.multi_teacher_oracle_cache:
            raise ValueError(
                "Physical microbatch mode does not support multi-teacher export/cache"
            )
        logging.warning(
            "Physical microbatch=%d, logical batch=%d: data/update steps stay unchanged; "
            "loss normalization, random draws and batch-dependent statistics may change.",
            args.micro_batch_size,
            args.batch_size,
        )
    if args.optimizer == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "--optimizer muon requires torch.optim.Muon; use repository PyTorch 2.11"
            )
        validate_muon_options(
            lr=args.learning_rate,
            muon_lr=10 * args.learning_rate if args.muon_lr is None else args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.weight_decay,
            muon_weight_decay=args.muon_weight_decay,
            ns_steps=args.muon_ns_steps,
            adjust_lr_fn=args.muon_adjust_lr_fn,
        )
    if args.multi_teacher_oracle_export_dir and args.multi_teacher_oracle_cache:
        raise ValueError("multi-teacher export and distillation are separate phases")
    if args.multi_teacher_oracle_mode != "none" and not args.multi_teacher_oracle_cache:
        raise ValueError(
            "multi-teacher oracle mode requires --multi-teacher-oracle-cache"
        )
    if args.multi_teacher_oracle_export_dir and not args.init_draft_model_path:
        raise ValueError("multi-teacher export requires --init-draft-model-path")
    if args.multi_teacher_export_top_k < 2:
        raise ValueError("multi-teacher export top-k must be at least 2")
    prepare_warm_start(args)
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    _patch_flex_attention_triton_backend()

    device = get_local_device()
    device_type = device.type

    needs_target_hidden = (
        (args.dspark_loss_type == "tv-acceptance")
        or (args.offline_acceptance_objective != "none")
        or (args.bv_loss_alpha > 0)
        or (args.carh_predecessor_diagnostics_interval > 0)
        or (args.l1_loss_alpha > 0)
        or (args.enable_confidence_head and args.confidence_head_alpha > 0)
        or args.pace_mode in {"overlap", "hybrid"}
        or (args.elastic_horizon_enabled and args.elastic_projective_alpha > 0)
    )

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(
            draft_model_last_checkpoint, "config.json"
        )
        if os.path.exists(checkpoint_config_path):
            print(f"Loading draft config from checkpoint: {checkpoint_config_path}")
            args.draft_config_path = checkpoint_config_path
    elif args.init_draft_model_path:
        print(
            f"Loading draft config from initialization checkpoint: {args.draft_config_path}"
        )

    target_model, draft_model = build_models(args, device)
    validate_predictive_auxiliary(args, draft_model)

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DSparkDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            validate_optimizer_resume(resume_state, args.optimizer)
            validate_predictive_resume(resume_state.get("args", {}), args)
            validate_bv_resume(resume_state.get("args", {}), args)
            validate_objective_resume(resume_state.get("args", {}), args)
            if int(getattr(draft_model.config, "carh_sampled_prefix_memory_rank", 0)):
                saved_args = resume_state.get("args", {})
                if isinstance(saved_args, argparse.Namespace):
                    saved_args = vars(saved_args)
                for key in (
                    "carh_sampled_prefix_memory_start_ratio",
                    "carh_sampled_prefix_memory_ramp_ratio",
                ):
                    if saved_args.get(key) != getattr(args, key):
                        raise ValueError(
                            f"Sampled-prefix memory scheduled resume setting changed: {key}"
                        )
            if args.on_policy_start_step is not None:
                saved_args = resume_state.get("args", {})
                if isinstance(saved_args, argparse.Namespace):
                    saved_args = vars(saved_args)
                for key in (
                    "on_policy_start_step",
                    "on_policy_ramp_steps",
                    "max_steps",
                    "num_epochs",
                    "batch_size",
                    "accumulation_steps",
                    "seed",
                    "train_data_path",
                    "on_policy_repair_value",
                    "on_policy_repair_value_horizon",
                    "fixed_prefix_reference_alpha",
                    "fixed_prefix_reference_batches",
                    "fixed_prefix_reference_interval",
                ):
                    # Pre-D/E checkpoints have no new fields; preserve C resumes.
                    if (
                        key not in saved_args
                        and key
                        in (
                            "on_policy_repair_value",
                            "on_policy_repair_value_horizon",
                            "fixed_prefix_reference_alpha",
                            "fixed_prefix_reference_batches",
                            "fixed_prefix_reference_interval",
                        )
                        and not args.on_policy_repair_value
                        and not args.fixed_prefix_reference_alpha
                    ):
                        continue
                    if saved_args.get(key) != getattr(args, key):
                        raise ValueError(
                            f"On-policy scheduled resume setting changed: {key}"
                        )
            if args.micro_batch_size:
                validate_microbatch_resume(resume_state.get("args"), args)
            print(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )
    elif args.init_draft_model_path:
        initialize_draft_weights(draft_model, args)
        print(
            f"Post-training initialized from {args.init_draft_model_path}; "
            "optimizer, scheduler and training progress start fresh."
        )

    if args.optimizer == "muon" and args.resume and resume_state is None:
        raise ValueError(
            "Muon --resume requires a training_state.pt checkpoint; use weight-only init otherwise"
        )

    if args.parallel_refiner_teacher_path:
        _load_parallel_refiner_teacher(
            draft_model,
            args.parallel_refiner_teacher_path,
            args.freeze_parallel_refiner_teacher,
        )

    if args.freeze_dynamic_conv_during_transition:
        frozen_conv_parameters = 0
        for layer in draft_model.layers:
            for conv in (
                getattr(layer, "attention_conv", None),
                getattr(layer, "mlp_conv", None),
            ):
                if conv is not None:
                    for parameter in conv.parameters():
                        parameter.requires_grad_(False)
                        frozen_conv_parameters += parameter.numel()
        print_on_rank0(
            "Local transition migration: froze "
            f"{frozen_conv_parameters:,} DynamicConv parameters."
        )

    trainable_parameters, scope_frozen_parameters = _apply_trainable_parameter_scope(
        draft_model, args.trainable_parameter_scope
    )
    print_on_rank0(
        f"Trainable parameter scope={args.trainable_parameter_scope}: "
        f"trainable={trainable_parameters:,}, frozen={scope_frozen_parameters:,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    mask_token_id = resolve_mask_token_id(args, draft_model, tokenizer)
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    dialogue_token_class_lookup = None
    if (
        args.on_policy_anchor_sampling == "dialogue-frontier"
        or args.dialogue_occupancy_diagnostics
        or args.dialogue_prefix_phase_weighting
        or args.dialogue_t2cm_boundary_weighting
        or args.dialogue_dsg_adaptive_margin
        or args.on_policy_language_hazard_credit
    ):
        dialogue_token_class_lookup = _build_dialogue_token_class_lookup(tokenizer)
        class_counts = torch.bincount(
            dialogue_token_class_lookup.long(), minlength=6
        ).tolist()
        print_on_rank0(
            "Dialogue token classes other/Han/Latin/number/punctuation/special="
            f"{class_counts}"
        )

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    print_on_rank0(f"Total training steps: {total_steps}")
    if args.bv_loss_alpha > 0:
        draft_model.config.bv_training = dict(
            objective="vocabulary-integrated-block-log-bv",
            alpha=args.bv_loss_alpha,
            temperature=args.bv_temperature,
            anneal_ratio=args.bv_anneal_ratio,
            block_chunk_size=args.bv_block_chunk_size,
            log_score_floor=-80.0,
            normalization="valid-block-mean",
            replaces="l1",
            corrected_proposal=True,
        )
        print_on_rank0(
            f"BV loss replacement (training only): {draft_model.config.bv_training}"
        )
        print_on_rank0(
            f"BV beta 0->1 over first {round(total_steps * args.bv_anneal_ratio)} optimizer steps; "
            "CE/confidence/prefix auxiliaries retained. No D-PACE/depth weights inside BV. "
            "Target-path interpretation requires matching regen sampling settings; "
            "BV diagnostics are NOT measured SGLang acceptance lengths."
        )
    if args.offline_acceptance_objective != "none":
        cold_steps = round(total_steps * args.offline_acceptance_cold_start_ratio)
        transition_steps = round(total_steps * args.offline_acceptance_transition_ratio)
        print_on_rank0(
            "Offline acceptance objective: "
            f"{args.offline_acceptance_objective}; "
            f"LK steps=0..{cold_steps}, "
            f"transition steps={cold_steps}..{cold_steps + transition_steps}, "
            f"E2E steps={cold_steps + transition_steps}..{total_steps}; "
            f"T={args.offline_acceptance_temperature}, "
            "rollout=disabled by objective design"
        )

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device_type,
        trust_remote_code=args.trust_remote_code,
    )

    semantic_codebook = None
    if args.conv_source_semantic_alpha > 0:
        print_on_rank0(
            "Preparing fixed semantic codebook (chunked CPU PCA, rank zero only)..."
        )
        semantic_codebook = prepare_semantic_codebook(
            args,
            target_components.lm_head.weight,
            draft_model.layers[0].mlp_conv.source_rank,
        )
        print_on_rank0(
            "Semantic codebook ready; artifact: "
            + os.path.join(args.output_dir, "source_semantic_codebook.pt")
        )
    draft_model.config.predictive_auxiliary = dict(
        conv_source_semantic_alpha=args.conv_source_semantic_alpha,
        carh_reference_calibration_alpha=args.carh_reference_calibration_alpha,
        warmup_ratio=args.predictive_aux_warmup_ratio,
        ramp_ratio=args.predictive_aux_ramp_ratio,
        source_semantic_codebook_seed=args.source_semantic_codebook_seed,
        semantic_scope="mlp-source-only",
        reference_scope="valid-in-block-predecessors",
    )
    if args.conv_source_semantic_alpha > 0 or args.carh_reference_calibration_alpha > 0:
        print_on_rank0(
            f"Training-only predictive auxiliaries: {draft_model.config.predictive_auxiliary}"
        )

    dspark_model = OnlineDSparkModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        conv_source_semantic_alpha=args.conv_source_semantic_alpha,
        carh_reference_calibration_alpha=args.carh_reference_calibration_alpha,
        predictive_aux_warmup_ratio=args.predictive_aux_warmup_ratio,
        predictive_aux_ramp_ratio=args.predictive_aux_ramp_ratio,
        source_semantic_codebook=semantic_codebook,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        ce_loss_alpha=args.ce_loss_alpha,
        l1_loss_alpha=args.l1_loss_alpha,
        bv_loss_alpha=args.bv_loss_alpha,
        bv_temperature=args.bv_temperature,
        bv_anneal_ratio=args.bv_anneal_ratio,
        bv_block_chunk_size=args.bv_block_chunk_size,
        offline_acceptance_objective=args.offline_acceptance_objective,
        offline_acceptance_greedy_weight=(args.offline_acceptance_greedy_weight),
        offline_acceptance_stochastic_weight=(
            args.offline_acceptance_stochastic_weight
        ),
        offline_acceptance_temperature=args.offline_acceptance_temperature,
        offline_acceptance_lk_eta=args.offline_acceptance_lk_eta,
        offline_acceptance_dpace_rho=args.offline_acceptance_dpace_rho,
        offline_acceptance_cold_start_ratio=(args.offline_acceptance_cold_start_ratio),
        offline_acceptance_transition_ratio=(args.offline_acceptance_transition_ratio),
        confidence_head_alpha=args.confidence_head_alpha,
        confidence_target_mode=args.confidence_target_mode,
        confidence_detach_backbone=args.confidence_detach_backbone,
        carh_predecessor_diagnostics_interval=args.carh_predecessor_diagnostics_interval,
        carh_sampled_prefix_memory_start_ratio=(
            args.carh_sampled_prefix_memory_start_ratio
        ),
        carh_sampled_prefix_memory_ramp_ratio=(
            args.carh_sampled_prefix_memory_ramp_ratio
        ),
        recall_correction_alpha=args.recall_correction_alpha,
        recall_partition_top_k=args.recall_partition_top_k,
        recall_correction_loss_budget=args.recall_correction_loss_budget,
        pace_mode=args.pace_mode,
        pace_alpha=args.pace_alpha,
        pace_hybrid_beta=args.pace_hybrid_beta,
        pace_apply_to=args.pace_apply_to,
        pace_blend_max=args.pace_blend_max,
        pace_warmup_ratio=args.pace_warmup_ratio,
        pace_ramp_ratio=args.pace_ramp_ratio,
        pace_decay_start_ratio=args.pace_decay_start_ratio,
        pace_blend_final=args.pace_blend_final,
        pace_residual_beta=args.pace_residual_beta,
        pace_residual_min=args.pace_residual_min,
        pace_residual_max=args.pace_residual_max,
        prefix_credit_mode=args.prefix_credit_mode,
        prefix_credit_alpha=args.prefix_credit_alpha,
        prefix_credit_warmup_ratio=args.prefix_credit_warmup_ratio,
        prefix_credit_ramp_ratio=args.prefix_credit_ramp_ratio,
        prefix_credit_backbone_grad_scale=(args.prefix_credit_backbone_grad_scale),
        shallow_frc_alpha=args.shallow_frc_alpha,
        shallow_frc_max_depth=args.shallow_frc_max_depth,
        shallow_frc_margin=args.shallow_frc_margin,
        shallow_frc_temperature=args.shallow_frc_temperature,
        dfap_alpha=args.dfap_alpha,
        dfap_min_depth=args.dfap_min_depth,
        dfap_margin=args.dfap_margin,
        dfap_temperature=args.dfap_temperature,
        state_credit_partition=args.state_credit_partition,
        conv_gate_loss_alpha=args.conv_gate_loss_alpha,
        block_summary_counterfactual_alpha=args.block_summary_counterfactual_alpha,
        transition_credit_alpha=args.transition_credit_alpha,
        transition_credit_max_depth=args.transition_credit_max_depth,
        transition_credit_margin=args.transition_credit_margin,
        transition_credit_temperature=args.transition_credit_temperature,
        transition_credit_loss_budget=args.transition_credit_loss_budget,
        transition2_margin_alpha=args.transition2_margin_alpha,
        transition2_margin_floor=args.transition2_margin_floor,
        transition2_margin_temperature=args.transition2_margin_temperature,
        transition2_margin_loss_budget=args.transition2_margin_loss_budget,
        prefix_bottleneck_alpha=args.prefix_bottleneck_alpha,
        prefix_bottleneck_max_depth=args.prefix_bottleneck_max_depth,
        prefix_bottleneck_margin_floor=args.prefix_bottleneck_margin_floor,
        prefix_bottleneck_softmin_temperature=(
            args.prefix_bottleneck_softmin_temperature
        ),
        prefix_bottleneck_loss_temperature=(args.prefix_bottleneck_loss_temperature),
        prefix_bottleneck_loss_budget=args.prefix_bottleneck_loss_budget,
        deep_survival_guard_alpha=args.deep_survival_guard_alpha,
        deep_survival_guard_min_prefix=args.deep_survival_guard_min_prefix,
        deep_survival_guard_start_depth=args.deep_survival_guard_start_depth,
        deep_survival_guard_margin_floor=(args.deep_survival_guard_margin_floor),
        deep_survival_guard_temperature=args.deep_survival_guard_temperature,
        deep_survival_guard_loss_budget=(args.deep_survival_guard_loss_budget),
        carh_recoverability_top_k=args.carh_recoverability_top_k,
        carh_recovery_loss_alpha=args.carh_recovery_loss_alpha,
        carh_preservation_loss_alpha=args.carh_preservation_loss_alpha,
        carh_noop_loss_alpha=args.carh_noop_loss_alpha,
        carh_preservation_margin=args.carh_preservation_margin,
        carh_preservation_temperature=args.carh_preservation_temperature,
        carh_recoverability_loss_budget=(args.carh_recoverability_loss_budget),
        carh_gate_calibration_alpha=args.carh_gate_calibration_alpha,
        carh_gate_noop_alpha=args.carh_gate_noop_alpha,
        carh_gate_margin_threshold=args.carh_gate_margin_threshold,
        carh_gate_calibration_warmup_ratio=(args.carh_gate_calibration_warmup_ratio),
        selector_loss_alpha=args.selector_loss_alpha,
        selector_rollout_ratio=args.selector_rollout_ratio,
        selector_training_mode=args.selector_training_mode,
        selector_distill_alpha=args.selector_distill_alpha,
        selector_preservation_alpha=args.selector_preservation_alpha,
        selector_distill_temperature=args.selector_distill_temperature,
        selector_advantage_margin=args.selector_advantage_margin,
        selector_teacher_warmup_ratio=args.selector_teacher_warmup_ratio,
        selector_distill_start_ratio=args.selector_distill_start_ratio,
        selector_consolidation_ratio=args.selector_consolidation_ratio,
        selector_loss_budget=args.selector_loss_budget,
        selector_margin_threshold=float(draft_model.selector_margin_threshold),
        selector_tree_loss_alpha=args.selector_tree_loss_alpha,
        selector_tree_branch_width=args.selector_tree_branch_width,
        selector_tree_margin=args.selector_tree_margin,
        selector_tree_temperature=args.selector_tree_temperature,
        on_policy_survival_alpha=args.on_policy_survival_alpha,
        on_policy_full_alpha=args.on_policy_full_alpha,
        on_policy_first_rejection_max_depth=(args.on_policy_first_rejection_max_depth),
        on_policy_boundary_max_depth=args.on_policy_boundary_max_depth,
        on_policy_deep_alpha=args.on_policy_deep_alpha,
        on_policy_deep_start_depth=args.on_policy_deep_start_depth,
        on_policy_margin_floor=args.on_policy_margin_floor,
        on_policy_temperature=args.on_policy_temperature,
        on_policy_rollout_temperatures=args.on_policy_rollout_temperatures,
        on_policy_rollout_temperature_probs=(args.on_policy_rollout_temperature_probs),
        on_policy_anchor_sampling=args.on_policy_anchor_sampling,
        on_policy_hazard_power=args.on_policy_hazard_power,
        on_policy_flatness_power_max=args.on_policy_flatness_power_max,
        on_policy_flatness_warmup_ratio=(args.on_policy_flatness_warmup_ratio),
        on_policy_uniform_exploration=args.on_policy_uniform_exploration,
        dialogue_phase_masses=args.dialogue_phase_masses,
        dialogue_boundary_boost=args.dialogue_boundary_boost,
        dialogue_script_transition_boost=(args.dialogue_script_transition_boost),
        dialogue_occupancy_diagnostics=args.dialogue_occupancy_diagnostics,
        dialogue_num_sources=1 + len(args.additional_train_data_path),
        dialogue_token_class_lookup=dialogue_token_class_lookup,
        dialogue_prefix_phase_weighting=(args.dialogue_prefix_phase_weighting),
        dialogue_t2cm_boundary_weighting=(args.dialogue_t2cm_boundary_weighting),
        dialogue_dsg_adaptive_margin=args.dialogue_dsg_adaptive_margin,
        dialogue_dsg_late_margin_bonus=(args.dialogue_dsg_late_margin_bonus),
        dialogue_dsg_boundary_margin_bonus=(args.dialogue_dsg_boundary_margin_bonus),
        on_policy_distributional_top_k=args.on_policy_distributional_top_k,
        on_policy_distributional_min_top_k=(args.on_policy_distributional_min_top_k),
        on_policy_distributional_mass_threshold=(
            args.on_policy_distributional_mass_threshold
        ),
        on_policy_distributional_alpha=args.on_policy_distributional_alpha,
        on_policy_distributional_deep_alpha=(args.on_policy_distributional_deep_alpha),
        on_policy_distributional_cold_start_ratio=(
            args.on_policy_distributional_cold_start_ratio
        ),
        on_policy_rejection_aligned_alpha=(args.on_policy_rejection_aligned_alpha),
        on_policy_rejection_aligned_survival_blend=(
            args.on_policy_rejection_aligned_survival_blend
        ),
        on_policy_temperature_exclusive_routing=(
            args.on_policy_temperature_exclusive_routing
        ),
        on_policy_mixed_kl_alpha=args.on_policy_mixed_kl_alpha,
        on_policy_mixed_kl_accepted_weight=(args.on_policy_mixed_kl_accepted_weight),
        on_policy_mixed_kl_rejected_weight=(args.on_policy_mixed_kl_rejected_weight),
        on_policy_mixed_kl_rejection_decay=(args.on_policy_mixed_kl_rejection_decay),
        on_policy_clipped_rkl_alpha=args.on_policy_clipped_rkl_alpha,
        on_policy_clipped_rkl_clip=args.on_policy_clipped_rkl_clip,
        on_policy_clipped_rkl_temperature_weights=(
            args.on_policy_clipped_rkl_temperature_weights
        ),
        on_policy_target_distribution_temperature_floor=(
            args.on_policy_target_distribution_temperature_floor
        ),
        on_policy_target_margin_alpha=args.on_policy_target_margin_alpha,
        on_policy_target_margin_scale=args.on_policy_target_margin_scale,
        on_policy_target_margin_offset=args.on_policy_target_margin_offset,
        on_policy_target_margin_min=args.on_policy_target_margin_min,
        on_policy_target_margin_max=args.on_policy_target_margin_max,
        on_policy_target_margin_max_depth=(args.on_policy_target_margin_max_depth),
        on_policy_deployment_top_k=args.on_policy_deployment_top_k,
        on_policy_deployment_top_p=args.on_policy_deployment_top_p,
        on_policy_credit_partition=args.on_policy_credit_partition,
        on_policy_middle_max_depth=args.on_policy_middle_max_depth,
        on_policy_middle_alpha=args.on_policy_middle_alpha,
        on_policy_temperature_hazard_credit=(args.on_policy_temperature_hazard_credit),
        on_policy_hazard_ema_decay=args.on_policy_hazard_ema_decay,
        on_policy_hazard_weight_min=args.on_policy_hazard_weight_min,
        on_policy_hazard_weight_max=args.on_policy_hazard_weight_max,
        on_policy_marginal_value_credit=(args.on_policy_marginal_value_credit),
        on_policy_marginal_value_temperature=(
            args.on_policy_marginal_value_temperature
        ),
        on_policy_marginal_value_weight_min=(args.on_policy_marginal_value_weight_min),
        on_policy_marginal_value_weight_max=(args.on_policy_marginal_value_weight_max),
        on_policy_language_hazard_credit=(args.on_policy_language_hazard_credit),
        on_policy_language_han_threshold=(args.on_policy_language_han_threshold),
        on_policy_language_hazard_ema_decay=(args.on_policy_language_hazard_ema_decay),
        on_policy_language_hazard_weight_min=(
            args.on_policy_language_hazard_weight_min
        ),
        on_policy_language_hazard_weight_max=(
            args.on_policy_language_hazard_weight_max
        ),
        on_policy_mix_ratio_max=args.on_policy_mix_ratio_max,
        on_policy_start_step=args.on_policy_start_step,
        on_policy_ramp_steps=args.on_policy_ramp_steps,
        on_policy_repair_value=args.on_policy_repair_value,
        on_policy_repair_value_horizon=args.on_policy_repair_value_horizon,
        on_policy_interval=args.on_policy_interval,
        on_policy_loss_budget=args.on_policy_loss_budget,
        on_policy_preservation_alpha=args.on_policy_preservation_alpha,
        on_policy_regression_alpha=args.on_policy_regression_alpha,
        on_policy_regression_margin=args.on_policy_regression_margin,
        on_policy_greedy_preservation_alpha=(args.on_policy_greedy_preservation_alpha),
        on_policy_greedy_preservation_margin_floor=(
            args.on_policy_greedy_preservation_margin_floor
        ),
        on_policy_greedy_preservation_temperature=(
            args.on_policy_greedy_preservation_temperature
        ),
        on_policy_reset_replay_alpha=args.on_policy_reset_replay_alpha,
        on_policy_reset_replay_horizon=args.on_policy_reset_replay_horizon,
        on_policy_reset_replay_max_rejection_depth=(
            args.on_policy_reset_replay_max_rejection_depth
        ),
        on_policy_reset_replay_start_ratio=(args.on_policy_reset_replay_start_ratio),
        on_policy_reset_replay_ramp_ratio=(args.on_policy_reset_replay_ramp_ratio),
        on_policy_reset_replay_loss_budget=(args.on_policy_reset_replay_loss_budget),
        on_policy_reset_replay_advantage_threshold=(
            args.on_policy_reset_replay_advantage_threshold
        ),
        on_policy_reset_replay_value_clip=(args.on_policy_reset_replay_value_clip),
        on_policy_preference_alpha=args.on_policy_preference_alpha,
        on_policy_preference_value_gap=args.on_policy_preference_value_gap,
        on_policy_preference_temperature=args.on_policy_preference_temperature,
        on_policy_preference_loss_budget=args.on_policy_preference_loss_budget,
        on_policy_pareto_credit=args.on_policy_pareto_credit,
        on_policy_pareto_ema_decay=args.on_policy_pareto_ema_decay,
        on_policy_pareto_temperature=args.on_policy_pareto_temperature,
        on_policy_pareto_weight_min=args.on_policy_pareto_weight_min,
        on_policy_pareto_weight_max=args.on_policy_pareto_weight_max,
        parallel_refiner_hazard_alpha=args.parallel_refiner_hazard_alpha,
        parallel_refiner_recovery_alpha=args.parallel_refiner_recovery_alpha,
        parallel_refiner_preservation_alpha=(args.parallel_refiner_preservation_alpha),
        parallel_refiner_margin_floor=args.parallel_refiner_margin_floor,
        parallel_refiner_temperature=args.parallel_refiner_temperature,
        parallel_refiner_loss_budget=args.parallel_refiner_loss_budget,
        refiner_advantage_mode=args.refiner_advantage_mode,
        refiner_advantage_threshold=args.refiner_advantage_threshold,
        refiner_advantage_gate_alpha=args.refiner_advantage_gate_alpha,
        refiner_advantage_regression_alpha=(args.refiner_advantage_regression_alpha),
        refiner_advantage_distill_alpha=args.refiner_advantage_distill_alpha,
        refiner_advantage_preservation_alpha=(
            args.refiner_advantage_preservation_alpha
        ),
        refiner_advantage_greedy_preservation_alpha=(
            args.refiner_advantage_greedy_preservation_alpha
        ),
        refiner_advantage_greedy_margin_floor=(
            args.refiner_advantage_greedy_margin_floor
        ),
        refiner_advantage_gate_distill_min_probability=(
            args.refiner_advantage_gate_distill_min_probability
        ),
        refiner_advantage_distill_top_k=(args.refiner_advantage_distill_top_k),
        refiner_advantage_distill_temperature=(
            args.refiner_advantage_distill_temperature
        ),
        refiner_advantage_value_clip=args.refiner_advantage_value_clip,
        refiner_advantage_distill_start_ratio=(
            args.refiner_advantage_distill_start_ratio
        ),
        refiner_advantage_consolidation_ratio=(
            args.refiner_advantage_consolidation_ratio
        ),
        refiner_advantage_loss_budget=args.refiner_advantage_loss_budget,
        vat_enabled=args.vat_enabled,
        vat_hard_loss_alpha=args.vat_hard_loss_alpha,
        vat_soft_loss_alpha=args.vat_soft_loss_alpha,
        vat_verification_head_alpha=args.vat_verification_head_alpha,
        vat_post_rejection_decay_gamma=(args.vat_post_rejection_decay_gamma),
        vat_simulation_temperature=args.vat_simulation_temperature,
        branch_value_mode=args.branch_value_mode,
        branch_value_top_m=args.branch_value_top_m,
        branch_value_horizon=args.branch_value_horizon,
        branch_value_alpha=args.branch_value_alpha,
        branch_value_temperature=args.branch_value_temperature,
        branch_value_loss_budget=args.branch_value_loss_budget,
        branch_value_warmup_ratio=args.branch_value_warmup_ratio,
        branch_value_consolidation_ratio=(args.branch_value_consolidation_ratio),
        elastic_horizon_enabled=args.elastic_horizon_enabled,
        elastic_short_horizon=args.elastic_short_horizon,
        elastic_long_horizon=args.elastic_long_horizon,
        elastic_warmup_ratio=args.elastic_warmup_ratio,
        elastic_late_ratio=args.elastic_late_ratio,
        elastic_consolidation_ratio=args.elastic_consolidation_ratio,
        elastic_middle_long_prob=args.elastic_middle_long_prob,
        elastic_late_long_prob=args.elastic_late_long_prob,
        elastic_pair_probability=args.elastic_pair_probability,
        elastic_projective_num_anchors=args.elastic_projective_num_anchors,
        elastic_projective_alpha=args.elastic_projective_alpha,
        elastic_projective_top_k=args.elastic_projective_top_k,
        elastic_overlap_gap=args.elastic_overlap_gap,
        elastic_margin_gap=args.elastic_margin_gap,
        elastic_loss_budget=args.elastic_loss_budget,
    )
    dspark_model.configure_multi_teacher_oracle(
        mode=args.multi_teacher_oracle_mode,
        alpha=args.multi_teacher_oracle_alpha,
        advantage_threshold=args.multi_teacher_oracle_advantage_threshold,
        value_clip=args.multi_teacher_oracle_value_clip,
        loss_budget=args.multi_teacher_oracle_loss_budget,
    )
    configure_objective(dspark_model, tokenizer, args)

    trajectory_writer = None
    if args.multi_teacher_oracle_export_dir:
        trajectory_writer = MultiTeacherTrajectoryWriter(
            output_dir=args.multi_teacher_oracle_export_dir,
            teacher_name=args.multi_teacher_name,
            rank=dist.get_rank(),
            top_k=args.multi_teacher_export_top_k,
            flush_records=args.multi_teacher_export_flush_records,
        )
        dspark_model.multi_teacher_export_sink = trajectory_writer
    if args.multi_teacher_oracle_cache:
        oracle_cache = MultiTeacherOracleCache(args.multi_teacher_oracle_cache)
        dspark_model.multi_teacher_oracle_provider = oracle_cache.lookup

    if (
        args.netprefix_mode in ("fixed-greedy", "h1-greedy")
        or args.branch_value_mode != "none"
        or args.refiner_advantage_mode != "none"
        or args.multi_teacher_oracle_export_dir is not None
        or args.multi_teacher_oracle_cache is not None
        or max(
            args.on_policy_survival_alpha,
            args.on_policy_full_alpha,
            args.on_policy_deep_alpha,
            args.on_policy_distributional_alpha,
            args.on_policy_distributional_deep_alpha,
            args.on_policy_rejection_aligned_alpha,
            args.on_policy_mixed_kl_alpha,
            args.on_policy_clipped_rkl_alpha,
            args.on_policy_target_margin_alpha,
            args.on_policy_preservation_alpha,
            args.on_policy_regression_alpha,
            args.on_policy_greedy_preservation_alpha,
            args.on_policy_reset_replay_alpha,
            args.on_policy_preference_alpha,
            args.carh_gate_calibration_alpha,
            args.carh_gate_noop_alpha,
            args.branch_value_alpha,
            args.parallel_refiner_hazard_alpha,
            args.parallel_refiner_recovery_alpha,
            args.parallel_refiner_preservation_alpha,
        )
        > 0
    ):

        @torch.no_grad()
        def score_on_policy_proposals(
            input_ids,
            attention_mask,
            anchors,
            proposals,
            active_rows,
            rollout_temperatures=None,
        ):
            """Write proposal paths into causal prefixes and score them.

            Multi-branch objectives (VAGRD and Branch Value) are deliberately
            evaluated one branch at a time. Flattening branches into the batch
            makes SGLang materialize a correspondingly larger full-vocabulary
            FP32 logits tensor and can add more than 10 GiB of peak memory.
            Serial scoring leaves labels unchanged while keeping the target
            forward at the same batch size as ordinary SR-OPSC.
            """
            if proposals.dim() == 3:
                branch_results = [
                    score_on_policy_proposals(
                        input_ids,
                        attention_mask,
                        anchors,
                        proposals[:, branch_index],
                        active_rows,
                        rollout_temperatures,
                    )
                    for branch_index in range(proposals.size(1))
                ]
                if isinstance(branch_results[0], tuple):
                    return tuple(
                        torch.stack(
                            [result[item_index] for result in branch_results],
                            dim=1,
                        )
                        for item_index in range(len(branch_results[0]))
                    )
                return torch.stack(branch_results, dim=1)
            if proposals.dim() != 2:
                raise ValueError("on-policy proposals must have shape [B,K] or [B,N,K]")
            scorer_ids = input_ids.clone()
            seq_len = scorer_ids.size(1)
            depth_count = proposals.size(-1)
            depth = torch.arange(depth_count, device=input_ids.device).view(1, -1)
            write_pos = anchors.unsqueeze(-1) + depth + 1
            in_bounds = write_pos.lt(seq_len) & active_rows.unsqueeze(-1)
            rows = torch.arange(input_ids.size(0), device=input_ids.device).view(-1, 1)
            scorer_ids[rows.expand_as(write_pos)[in_bounds], write_pos[in_bounds]] = (
                proposals[in_bounds]
            )
            score_pos = (anchors.unsqueeze(-1) + depth).clamp(max=seq_len - 1)
            if args.on_policy_distributional_top_k > 0:
                target_next_ids, target_topk_ids, target_topk_probs = (
                    target_model.score_proposal_distribution(
                        scorer_ids,
                        attention_mask,
                        args.on_policy_distributional_top_k,
                        temperatures=rollout_temperatures,
                        sampling_top_k=args.on_policy_deployment_top_k,
                        sampling_top_p=args.on_policy_deployment_top_p,
                        score_positions=score_pos,
                    )
                )
                result = (
                    target_next_ids,
                    target_topk_ids,
                    target_topk_probs,
                )
                return result
            target_next_ids = target_model.score_proposal_tokens(
                scorer_ids, attention_mask
            ).to(input_ids.device)
            result = target_next_ids.gather(1, score_pos)
            return result

        dspark_model.on_policy_scorer = score_on_policy_proposals

    netprefix_hook = None
    if args.on_policy_repair_value or args.fixed_prefix_reference_alpha:
        if draft_model.markov_head is None:
            raise ValueError("D/E requires CARH recurrence")
        for name in (
            "recall_correction",
            "parallel_refiner",
            "candidate_selector",
            "prefix_state_mixer",
        ):
            if getattr(draft_model, name, None) is not None:
                raise ValueError(f"D/E v1 does not support {name}")
    if args.netprefix_mode != "off":
        if draft_model.markov_head is None:
            raise ValueError("NetPrefix v1 requires the CARH/Markov serving path")
        for name in (
            "recall_correction",
            "parallel_refiner",
            "candidate_selector",
            "prefix_state_mixer",
        ):
            if getattr(draft_model, name, None) is not None:
                raise ValueError(f"NetPrefix v1 does not support {name}")
        if args.on_policy_distributional_top_k > 0:
            raise ValueError(
                "NetPrefix v1 requires the exact greedy ID scorer, not top-k distribution scoring"
            )
        netprefix_hook = GreedyPrefixHook(dspark_model)
        dspark_model.netprefix_hook = netprefix_hook

    # Capture full (unsharded) parameter shapes BEFORE FSDP wrapping so
    # the optimizer can convert saved state dicts between rank-agnostic
    # and FSDP-sharded formats.
    draft_full_param_shapes = [
        p.shape for p in draft_model.parameters() if p.requires_grad
    ]

    # Wrap each transformer block as its own FSDP unit (compute/comm overlap).
    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=configure_fsdp_optimizer_precision(
            dspark_model, args.optimizer
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in dspark_model.modules() if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_classes,
        )
    else:
        print_with_rank(
            "No _no_split_modules on draft model; falling back to single-unit "
            "FSDP wrap (no compute-comm overlap)."
        )
    dspark_model = FSDP(dspark_model, **fsdp_kwargs)
    print_with_rank("Initialized FSDP")

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]

    optimizer = build_bf16_optimizer(
        draft_model,
        optimizer_name=args.optimizer,
        fsdp_model=dspark_model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )
    optimizer.set_full_param_shapes(draft_full_param_shapes)

    if args.resume_extension_start_step is not None:
        boundary = args.resume_extension_start_step
        if (
            not args.resume
            or resume_state is None
            or args.optimizer != "adamw"
            or not resume_state.get("_fsdp_full_param_format")
            or args.accumulation_steps != 1
            or args.num_epochs != 10
            or boundary != 6 * len(train_dataloader)
            or not boundary <= resume_state["global_step"] < total_steps
            or not 0 < args.resume_extension_lr < float("inf")
        ):
            raise ValueError(
                "6-to-10 extension requires AdamW resume, unchanged batches/epoch, accumulation1, valid LR and step in [6 epochs, 10 epochs)"
            )
        # Restore Adam below, but reconstruct the extension scheduler explicitly.
        # Do not load an old completed cosine schedule or a saved LambdaLR into it.
        resume_state["scheduler_state_dict"] = optimizer.scheduler.state_dict()

    if resume_state is not None:
        # Detect whether the saved optimizer state is in the new
        # rank-agnostic (full-param) format or the legacy FSDP-sharded
        # format.  The new format is identified by the presence of a
        # "_fsdp_full_param_format" marker; lacking that, we fall back
        # to the old load_state_dict for backwards compatibility and
        # then immediately re-save in the new format.
        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if resume_state.get("_fsdp_full_param_format"):
            load_fsdp_optimizer_state(
                optimizer,
                dspark_model,
                training_state_path,
                restore_scheduler=args.resume_extension_start_step is None,
            )
        else:
            # Legacy checkpoint: optimizer state was saved in
            # FSDP-sharded (rank-dependent) format.  We can only safely
            # load it on the rank that saved it (rank 0).  For other
            # ranks the param shapes differ, causing a crash in Adam.
            # Fall back to loading only the scheduler (which has no
            # per-param tensors) and let Adam re-initialise on other
            # ranks — the loss of a few steps of momentum is acceptable.
            if dist.get_rank() == 0:
                optimizer.load_state_dict(resume_state)
            else:
                # Only restore the scheduler (lr schedule) on non-zero
                # ranks.  The optimizer Adam state (exp_avg / exp_avg_sq)
                # will start from scratch on these ranks — mild warm-up
                # penalty, but avoids the crash.
                optimizer.scheduler.load_state_dict(
                    resume_state["scheduler_state_dict"]
                )
                print_on_rank0(
                    "Legacy FSDP-sharded optimizer checkpoint: only "
                    "scheduler restored on non-rank-0; Adam state "
                    "re-initialised."
                )
            # Re-save in the new rank-agnostic format so future
            # resumes work correctly on all ranks.
            save_fsdp_optimizer_state(
                optimizer,
                dspark_model,
                training_state_path,
            )
            if dist.get_rank() == 0:
                saved = torch.load(
                    training_state_path, map_location="cpu", weights_only=False
                )
                saved["epoch"] = resume_state["epoch"]
                saved["global_step"] = resume_state["global_step"]
                saved["args"] = resume_state["args"]
                torch.save(saved, training_state_path)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Restored optimizer/scheduler state: "
            f"epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}"
        )

    if args.resume_extension_start_step is not None:
        from specforge.resume_extension import extension_lr_factor

        boundary = args.resume_extension_start_step
        elapsed = global_step - boundary
        duration = total_steps - boundary
        if len(optimizer.optimizer.param_groups) != 1:
            raise ValueError(
                "Extension currently supports one AdamW parameter group only"
            )
        for group in optimizer.optimizer.param_groups:
            group["lr"] = args.resume_extension_lr
            group["initial_lr"] = args.resume_extension_lr
        optimizer.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer.optimizer,
            lambda step: extension_lr_factor(step, duration),
            last_epoch=elapsed - 1,
        )
        print_on_rank0(
            f"Extension schedule: start={boundary}, end={total_steps}, elapsed={elapsed}, peak_lr={args.resume_extension_lr}, current_lr={optimizer.get_learning_rate()}"
        )

    # Derive the data cursor from consumed batches, including a mid-epoch pilot stop.
    if args.netprefix_mode != "off" or args.stop_after_steps is not None:
        start_epoch, skip_steps = divmod(global_step, len(train_dataloader))
    else:
        skip_steps = global_step - start_epoch * len(train_dataloader)

    reference_runtime = None
    if args.fixed_prefix_reference_alpha:
        reference_runtime = FixedPrefixReference(
            args,
            dspark_model,
            train_dataloader,
            lambda batch, seed: _forward_dspark_data_batch(
                batch,
                args,
                device,
                target_model,
                dspark_model,
                needs_target_hidden,
                seed,
            ),
        )
        dspark_model.fixed_prefix_reference_runtime = reference_runtime
        if args.resume:
            if not draft_model_last_checkpoint:
                raise ValueError("E resume requires a complete same-arm checkpoint")
            reference_runtime.load(draft_model_last_checkpoint, global_step)

    netprefix_runtime = None
    if args.netprefix_mode != "off":
        setup_rng = rng_state()
        control_batches, audit_loader = [], None
        if args.netprefix_mode in ("fixed-greedy", "h1-greedy"):

            def heldout_loader(path):
                dataset = load_dataset("json", data_files=path)["train"]
                processed = build_eagle3_dataset(
                    dataset=dataset,
                    tokenizer=tokenizer,
                    chat_template=args.chat_template,
                    max_length=args.max_length,
                    is_preformatted=args.is_preformatted,
                )
                processed = processed.add_column(
                    "__sample_id", list(range(len(processed)))
                )
                # Avoid DistributedSampler padding duplicated audit examples.
                multiple = dist.get_world_size(get_dp_group()) * args.batch_size
                usable = len(processed) // multiple * multiple
                if usable == 0:
                    raise ValueError(
                        f"NetPrefix pool {path} has no complete global batch"
                    )
                processed = processed.select(range(usable))
                return prepare_dp_dataloaders(
                    processed,
                    args.batch_size,
                    num_workers=0,
                    shuffle=False,
                    process_group=get_dp_group(),
                )

            if args.netprefix_data_source == "train-stream":
                audit_loader = TrainingStreamSource(
                    train_dataloader.dataset,
                    train_dataloader.collate_fn,
                    args.batch_size,
                    dist.get_rank(),
                    dist.get_world_size(),
                    args.netprefix_control_batches,
                    args.seed + 1700000,
                )
                control_batches = audit_loader.control_batches()
            else:
                control_loader = heldout_loader(args.netprefix_control_data_path)
                audit_loader = heldout_loader(args.netprefix_audit_data_path)
                import itertools

                control_batches = list(
                    itertools.islice(control_loader, args.netprefix_control_batches)
                )
                if (
                    len(control_batches) != args.netprefix_control_batches
                    or len(audit_loader) == 0
                ):
                    raise ValueError("NetPrefix held-out datasets are too small")
        netprefix_runtime = NetPrefixRuntime(
            args,
            dspark_model,
            optimizer,
            netprefix_hook,
            lambda batch, seed: _forward_dspark_data_batch(
                batch,
                args,
                device,
                target_model,
                dspark_model,
                needs_target_hidden,
                seed,
            ),
            control_batches,
            audit_loader,
        )
        dspark_model.netprefix_runtime = netprefix_runtime
        restore_rng(setup_rng)
        if args.resume:
            if not draft_model_last_checkpoint:
                raise ValueError("NetPrefix --resume did not find a checkpoint")
            netprefix_runtime.load(draft_model_last_checkpoint, global_step)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")
    stop = (
        args.stop_after_steps is not None and global_step >= args.stop_after_steps
    ) or (args.max_steps is not None and global_step >= args.max_steps)

    for epoch in range(start_epoch, args.num_epochs):
        if stop:
            break
        train_dataloader.sampler.set_epoch(epoch)
        if args.multi_teacher_oracle_export_dir:
            draft_model.eval()
        else:
            draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            # PACE mixture scheduling uses optimizer-step units, matching the
            # optimizer warmup and DFlash auxiliary schedules.
            optimizer_step = (
                global_step + args.accumulation_steps - 1
            ) // args.accumulation_steps
            dspark_model.set_training_progress(optimizer_step, total_steps)
            transition_conv_scale = draft_model.set_local_transition_progress(
                optimizer_step, total_steps
            )

            if netprefix_runtime is not None:
                (
                    loss,
                    accuracy,
                    loss_per_position,
                    acc_per_position,
                    count_per_position,
                    loss_components,
                ) = netprefix_runtime.step(data, global_step)
                loss_components.update(netprefix_runtime.metrics(loss.device))
            elif args.micro_batch_size:
                loss, accuracy, loss_components = forward_backward_microbatches(
                    data,
                    args.micro_batch_size,
                    lambda micro, index: _forward_dspark_data_batch(
                        micro,
                        args,
                        device,
                        target_model,
                        dspark_model,
                        needs_target_hidden,
                        global_step,
                        micro_index=index,
                    ),
                )
            else:
                if reference_runtime is not None:
                    reference_runtime.before_step(global_step)
                (
                    loss,
                    accuracy,
                    loss_per_position,
                    acc_per_position,
                    count_per_position,
                    loss_components,
                ) = _forward_dspark_data_batch(
                    data,
                    args,
                    device,
                    target_model,
                    dspark_model,
                    needs_target_hidden,
                    global_step,
                )

            if args.multi_teacher_oracle_export_dir:
                if global_step % args.log_interval == 0:
                    print_on_rank0(
                        f"Multi-teacher export step {global_step}: "
                        f"records/rank={trajectory_writer.count}"
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break
                continue

            if netprefix_runtime is None and not args.micro_batch_size:
                (loss / args.accumulation_steps).backward()
                if reference_runtime is not None:
                    loss_components.update(
                        reference_runtime.backward(data, global_step)
                    )

            if netprefix_runtime is None and global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                # Average over the DP group (not the full world, which includes
                # TP ranks that hold redundant copies).
                dp_group = get_dp_group()
                dp_size = dist.get_world_size(dp_group)

                loss_log = loss.clone()
                acc_log = accuracy.clone()
                if dp_group is not None:
                    dist.all_reduce(loss_log, group=dp_group)
                    dist.all_reduce(acc_log, group=dp_group)
                else:
                    dist.all_reduce(loss_log)
                    dist.all_reduce(acc_log)
                loss_log = loss_log / dp_size
                acc_log = acc_log / dp_size

                # Optional objectives can legitimately emit rank-local
                # diagnostics (for example, one rank samples a useful
                # multi-rollout preference while another does not).  Reducing
                # each rank's raw dict directly gives NCCL a different number
                # of collectives and deadlocks the following FSDP forward.
                # Agree on a sorted union first, then contribute zero for a
                # locally absent diagnostic so every rank executes the exact
                # same collective sequence.
                local_component_keys = sorted(loss_components)
                gathered_component_keys = [None] * dp_size
                dist.all_gather_object(
                    gathered_component_keys,
                    local_component_keys,
                    group=dp_group,
                )
                component_keys = sorted(
                    {key for rank_keys in gathered_component_keys for key in rank_keys}
                )
                comp_log = {}
                for key in component_keys:
                    value = loss_components.get(key)
                    v = (
                        value.clone().float()
                        if value is not None
                        else torch.zeros((), device=loss.device, dtype=torch.float32)
                    )
                    if dp_group is not None:
                        dist.all_reduce(v, group=dp_group)
                    else:
                        dist.all_reduce(v)
                    comp_log[key] = (v / dp_size).item()
                if draft_model.local_transition_enabled:
                    comp_log["local_transition_conv_scale"] = transition_conv_scale

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    comp_log,
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, dspark_model, draft_model, optimizer
                )

            if (
                args.stop_after_steps is not None
                and global_step >= args.stop_after_steps
            ):
                print_on_rank0(
                    f"Reached pilot stop={args.stop_after_steps}; full schedule unchanged."
                )
                stop = True
                break
            if args.max_steps is not None and global_step >= args.max_steps:
                print_on_rank0(f"Reached max_steps={args.max_steps}; stopping.")
                stop = True
                break

    if trajectory_writer is not None:
        trajectory_writer.close()
        dist.barrier()
        print_on_rank0(
            "Multi-teacher trajectory export complete: "
            f"{args.multi_teacher_oracle_export_dir}"
        )
    else:
        save_checkpoint(
            args,
            (
                global_step // len(train_dataloader)
                if args.netprefix_mode != "off" or args.stop_after_steps is not None
                else args.num_epochs
            ),
            global_step,
            dspark_model,
            draft_model,
            optimizer,
        )

    tracker.close()
    destroy_distributed()
