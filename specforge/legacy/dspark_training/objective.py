"""CLI, resume invariants and model wiring for the TV acceptance objective."""

import math
from collections.abc import Mapping

OBJECTIVE_DEFAULTS = {
    "dspark_loss_type": "ce-l1",
    "tv_sampling_temperature": 1.0,
    "tv_objective_chunk_blocks": 8,
}
TV_PREFIX_MODE = "teacher-forced"


def add_objective_args(group):
    # Saved with training args to distinguish old sampled-prefix checkpoints.
    group.set_defaults(tv_prefix_mode=TV_PREFIX_MODE)
    group.add_argument(
        "--dspark-loss-type",
        choices=["ce-l1", "tv-acceptance"],
        default=OBJECTIVE_DEFAULTS["dspark_loss_type"],
        help="tv-acceptance: accepted-length proxy on ground-truth prefixes; reuse one target forward.",
    )
    group.add_argument(
        "--tv-temperature",
        "--tv-sampling-temperature",
        dest="tv_sampling_temperature",
        type=float,
        default=OBJECTIVE_DEFAULTS["tv_sampling_temperature"],
        help="Shared target/draft softmax temperature. The old sampling name is an alias; no tokens are sampled.",
    )
    group.add_argument(
        "--tv-objective-chunk-blocks",
        type=int,
        default=OBJECTIVE_DEFAULTS["tv_objective_chunk_blocks"],
        help="Blocks per checkpointed vocabulary-loss chunk; bounds peak memory.",
    )
    group.add_argument(
        "--tv-verification-batch-size",
        type=int,
        default=None,
        help="Deprecated and ignored: teacher-forced TV does not run candidate verification.",
    )


def validate_objective_args(args):
    if args.dspark_loss_type != "tv-acceptance":
        return
    if (
        not math.isfinite(args.tv_sampling_temperature)
        or args.tv_sampling_temperature <= 0
    ):
        raise ValueError("--tv-temperature must be finite and positive")
    if args.tv_objective_chunk_blocks < 1:
        raise ValueError("TV objective chunk size must be positive")
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
    if (
        args.dspark_loss_type == "tv-acceptance"
        and saved.get("dspark_loss_type") == "tv-acceptance"
        and saved.get("tv_prefix_mode", "sampled") != TV_PREFIX_MODE
    ):
        raise ValueError(
            "Cannot resume TV with changed tv_prefix_mode: this checkpoint uses "
            "sampled or unknown prefixes, but training now uses teacher-forced "
            "prefixes. Use --init-draft-model-path with a new output directory."
        )
    # Chunk size and the obsolete verification batch size do not change the
    # teacher-forced objective or consume sampling RNG, so they may be retuned.
    for name in ("dspark_loss_type", "tv_sampling_temperature"):
        default = OBJECTIVE_DEFAULTS[name]
        if saved.get(name, default) != getattr(args, name):
            raise ValueError(
                f"Cannot resume with changed {name}; "
                "use --init-draft-model-path for a new objective"
            )


def configure_objective(model, tokenizer, args):
    """Enable target-state reuse and freeze unused heads before FSDP wrapping."""
    if args.dspark_loss_type != "tv-acceptance":
        return
    from transformers import AutoConfig

    from specforge.core.tv_acceptance import configure_tv_acceptance
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
    )
    if args.tv_verification_batch_size is not None:
        print_on_rank0(
            "--tv-verification-batch-size is deprecated and ignored: "
            "teacher-forced TV reuses the original target forward."
        )
    print_on_rank0(
        "TV acceptance: teacher-forced ground-truth prefixes, "
        "one target forward reused across anchors, no candidate sampling/verification, "
        "full-vocabulary softmax, "
        "loss=mean_blocks(1-sum(cumprod(1-TV))/K). "
        "CE/L1/confidence weights, PACE weights and loss-decay-gamma are unused."
    )
