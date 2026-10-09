"""CLI, resume invariants and model wiring for the TV acceptance objective."""

import math
from collections.abc import Mapping

OBJECTIVE_DEFAULTS = {
    "dspark_loss_type": "ce-l1",
    "tv_sampling_temperature": 1.0,
    "tv_objective_chunk_blocks": 8,
    "tv_verification_batch_size": 4,
}


def add_objective_args(group):
    group.add_argument(
        "--dspark-loss-type",
        choices=["ce-l1", "tv-acceptance"],
        default=OBJECTIVE_DEFAULTS["dspark_loss_type"],
        help="tv-acceptance: normalized accepted-length proxy on draft-sampled prefixes.",
    )
    for name in (
        "tv_sampling_temperature",
        "tv_objective_chunk_blocks",
        "tv_verification_batch_size",
    ):
        default = OBJECTIVE_DEFAULTS[name]
        group.add_argument(
            "--" + name.replace("_", "-"), type=type(default), default=default
        )


def validate_objective_args(args):
    if args.dspark_loss_type != "tv-acceptance":
        return
    if (
        not math.isfinite(args.tv_sampling_temperature)
        or args.tv_sampling_temperature <= 0
    ):
        raise ValueError("--tv-sampling-temperature must be finite and positive")
    if min(args.tv_objective_chunk_blocks, args.tv_verification_batch_size) < 1:
        raise ValueError("TV objective and verification chunk sizes must be positive")
    if args.tp_size != 1:
        raise ValueError("tv-acceptance currently requires --tp-size 1")
    if args.accumulation_steps != 1 or args.micro_batch_size:
        raise ValueError(
            "tv-acceptance requires accumulation-steps=1 and micro-batch-size=0 "
            "for exact block normalization"
        )
    if (
        args.offline_acceptance_objective != "none"
        or args.netprefix_mode != "off"
        or args.fixed_prefix_reference_alpha
        or args.elastic_horizon_enabled
        or args.trainable_parameter_scope != "all"
        or args.vat_enabled
        or args.multi_teacher_oracle_export_dir
        or args.multi_teacher_oracle_cache
    ):
        raise ValueError(
            "tv-acceptance is a standalone objective; disable auxiliary training modes"
        )


def validate_objective_resume(saved_args, args):
    saved = saved_args if isinstance(saved_args, Mapping) else vars(saved_args)
    for name, default in OBJECTIVE_DEFAULTS.items():
        if saved.get(name, default) != getattr(args, name):
            raise ValueError(
                f"Cannot resume with changed {name}; "
                "use --init-draft-model-path for a new objective"
            )


def configure_objective(model, target_model, tokenizer, args):
    """Attach the live verifier and freeze unused heads before FSDP wrapping."""
    if args.dspark_loss_type != "tv-acceptance":
        return
    from transformers import AutoConfig

    from specforge.core.tv_acceptance import configure_tv_acceptance
    from specforge.inference.target_engine.candidate_verifier import (
        CandidatePrefixVerifier,
    )
    from specforge.utils import print_on_rank0

    config = AutoConfig.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    eos_ids = getattr(config, "eos_token_id", tokenizer.eos_token_id)
    if eos_ids is None:
        eos_ids = ()
    elif isinstance(eos_ids, int):
        eos_ids = (eos_ids,)
    configure_tv_acceptance(
        model,
        temperature=args.tv_sampling_temperature,
        chunk_blocks=args.tv_objective_chunk_blocks,
        eos_token_ids=eos_ids,
        verifier=CandidatePrefixVerifier(
            target_model,
            batch_size=args.tv_verification_batch_size,
            pad_token_id=tokenizer.pad_token_id or 0,
        ),
    )
    print_on_rank0(
        "TV acceptance: fixed sampled prefixes, full-vocabulary softmax, "
        "loss=mean_blocks(1-sum(cumprod(1-TV))/K). "
        "CE/L1/confidence weights, PACE weights and loss-decay-gamma are unused."
    )
