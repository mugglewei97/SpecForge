"""CLI, resume invariants and model wiring for the TV acceptance objective."""

import math
from collections.abc import Mapping

OBJECTIVE_DEFAULTS = {
    "dspark_loss_type": "ce-l1",
    "tv_sampling_temperature": 1.0,
    "tv_objective_chunk_blocks": 8,
}
TV_PREFIX_MODE = "teacher-forced"
TV_BETA_PREFIX_MODE = "sampled-beta-v1"
TV_OBJECTIVES = ("tv-acceptance", "tv-acceptance-beta")


def add_objective_args(group):
    # Saved with training args to distinguish old sampled-prefix checkpoints.
    group.set_defaults(tv_prefix_mode=TV_PREFIX_MODE)
    group.add_argument(
        "--dspark-loss-type",
        choices=["ce-l1", *TV_OBJECTIVES],
        default=OBJECTIVE_DEFAULTS["dspark_loss_type"],
        help="tv-acceptance: ground-truth-prefix proxy; tv-acceptance-beta: sampled-prefix single-round accepted length.",
    )
    group.add_argument(
        "--tv-temperature",
        "--tv-sampling-temperature",
        dest="tv_sampling_temperature",
        type=float,
        default=OBJECTIVE_DEFAULTS["tv_sampling_temperature"],
        help="Shared target/draft softmax temperature; beta uses it for both proposal sampling and replay.",
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
        help="Beta candidate verification batch size (default: 64 with SGLang prefix KV, 4 otherwise); ignored by teacher-forced TV.",
    )


def validate_objective_args(args):
    if args.dspark_loss_type not in TV_OBJECTIVES:
        return
    if args.dspark_loss_type == "tv-acceptance-beta":
        args.tv_prefix_mode = TV_BETA_PREFIX_MODE
        if (
            args.tv_verification_batch_size is not None
            and args.tv_verification_batch_size < 1
        ):
            raise ValueError("TV beta verification batch size must be positive")
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
        args.dspark_loss_type in TV_OBJECTIVES
        and saved.get("dspark_loss_type") == args.dspark_loss_type
        and saved.get("tv_prefix_mode", "sampled")
        != (
            TV_BETA_PREFIX_MODE
            if args.dspark_loss_type == "tv-acceptance-beta"
            else TV_PREFIX_MODE
        )
    ):
        raise ValueError(
            "Cannot resume TV with changed tv_prefix_mode: this checkpoint uses "
            "incompatible or unknown prefix semantics. "
            "Use --init-draft-model-path with a new output directory."
        )
    # Memory knobs do not change the objective and may be retuned. Beta chunk
    # size can change RNG ordering, so this does not promise bitwise replay.
    for name in ("dspark_loss_type", "tv_sampling_temperature"):
        default = OBJECTIVE_DEFAULTS[name]
        if saved.get(name, default) != getattr(args, name):
            raise ValueError(
                f"Cannot resume with changed {name}; "
                "use --init-draft-model-path for a new objective"
            )


def configure_objective(model, tokenizer, args):
    """Enable target-state reuse and freeze unused heads before FSDP wrapping."""
    if args.dspark_loss_type not in TV_OBJECTIVES:
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
    from .config import is_dflash2_config

    if is_dflash2_config(model.draft_model.config):
        if args.dspark_loss_type != "tv-acceptance-beta":
            raise ValueError("DFlash2 requires tv-acceptance-beta in this entrypoint")
        model.configure_beta(
            temperature=args.tv_sampling_temperature,
            chunk_blocks=args.tv_objective_chunk_blocks,
            eos_token_ids=eos_ids,
            verification_batch_size=args.tv_verification_batch_size,
        )
        print_on_rank0(
            "DFlash2 TV beta: 8 native slots, 7 proposals; strict unary top-k "
            "selector sampling/replay, full-vocabulary target normalization."
        )
        return
    configure_tv_acceptance(
        model,
        temperature=args.tv_sampling_temperature,
        chunk_blocks=args.tv_objective_chunk_blocks,
        eos_token_ids=eos_ids,
        beta=args.dspark_loss_type == "tv-acceptance-beta",
        verification_batch_size=args.tv_verification_batch_size,
    )
    if args.dspark_loss_type == "tv-acceptance-beta":
        print_on_rank0(
            "TV acceptance beta: actual proposal prefixes, same-version rollout/replay, "
            "differentiable min(p(d),q(d))/stopgrad(q_old(d)) prefix factors, "
            "loss=mean_blocks(1-(a1+r1*a2+r1*r2*a3+...)/7). "
            "Fixed anchors; excludes bonus tokens and future round-start gradients. "
            "Reuse target hidden and full-attention Qwen3 prefix KV when available."
        )
        return
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
