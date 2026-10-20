"""CLI, resume invariants and model wiring for standalone training objectives."""

import math
from collections.abc import Mapping

OBJECTIVE_DEFAULTS = {
    "dspark_loss_type": "ce-l1",
    "tv_sampling_temperature": 1.0,
    "tv_objective_chunk_blocks": 8,
    "tv_anchor_mode": "random",
    "tv_rollout_max_tokens": 256,
}
TV_PREFIX_MODE = "teacher-forced"
TV_BETA_PREFIX_MODE = "sampled-beta-v1"
TV_OBJECTIVES = ("tv-acceptance", "tv-acceptance-beta")
DFLASH2_NATIVE_OBJECTIVES = ("dflash2", "dpace")


def add_objective_args(group):
    # Saved with training args to distinguish old sampled-prefix checkpoints.
    group.set_defaults(tv_prefix_mode=TV_PREFIX_MODE)
    group.add_argument(
        "--dspark-loss-type",
        choices=["ce-l1", *DFLASH2_NATIVE_OBJECTIVES, *TV_OBJECTIVES],
        default=OBJECTIVE_DEFAULTS["dspark_loss_type"],
        help="dflash2: native DFlash2 CE + selector; dpace: native DFlash2 D-PACE + selector; tv-acceptance: ground-truth-prefix proxy; tv-acceptance-beta: sampled-prefix single-round accepted length.",
    )
    group.add_argument(
        "--dpace-alpha",
        type=float,
        default=0.5,
        help="Probability smoothing floor for native DFlash2 --dspark-loss-type dpace, in [0, 1].",
    )
    group.add_argument(
        "--tv-anchor-mode",
        choices=["random", "on-policy"],
        default=OBJECTIVE_DEFAULTS["tv_anchor_mode"],
        help="Beta anchor selection: random supervised positions, or actual speculative rollout round starts (DFlash2 only).",
    )
    group.add_argument(
        "--tv-rollout-max-tokens",
        type=int,
        default=OBJECTIVE_DEFAULTS["tv_rollout_max_tokens"],
        help="On-policy beta: maximum new committed tokens per prompt; --num-anchors also caps rollout rounds per prompt.",
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
        help="Random-anchor beta verification batch size (default: 64 with SGLang prefix KV, 4 otherwise); ignored by teacher-forced TV and on-policy beta.",
    )


def validate_objective_args(args):
    if args.tv_rollout_max_tokens < 1:
        raise ValueError("--tv-rollout-max-tokens must be positive")
    if args.tv_anchor_mode == "on-policy":
        if args.dspark_loss_type != "tv-acceptance-beta":
            raise ValueError("--tv-anchor-mode on-policy requires tv-acceptance-beta")
        if args.num_anchors < 1:
            raise ValueError(
                "On-policy --num-anchors must be a positive rollout-round limit"
            )
        if args.max_length < 2:
            raise ValueError(
                "On-policy --max-length must leave room for a prefix and generation"
            )
        if args.target_model_backend not in ("hf", "sglang"):
            raise ValueError(
                "On-policy TV beta requires the HF or SGLang target backend"
            )
    if args.dspark_loss_type in DFLASH2_NATIVE_OBJECTIVES:
        if args.pace_mode != "none":
            raise ValueError(
                "--pace-mode is a DSpark option; for DFlash2 use "
                "--dspark-loss-type dpace and --dpace-alpha"
            )
        if args.dspark_loss_type == "dpace" and not (
            math.isfinite(args.dpace_alpha) and 0.0 <= args.dpace_alpha <= 1.0
        ):
            raise ValueError("--dpace-alpha must be finite and in [0, 1]")
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
    invariants = ["dspark_loss_type", "tv_sampling_temperature", "tv_anchor_mode"]
    if args.tv_anchor_mode == "on-policy":
        invariants.extend(("tv_rollout_max_tokens", "num_anchors", "max_length"))
    for name in invariants:
        default = OBJECTIVE_DEFAULTS.get(name)
        if (args.tv_anchor_mode == "on-policy" and name not in saved) or saved.get(
            name, default
        ) != getattr(args, name):
            raise ValueError(
                f"Cannot resume with changed {name}; "
                "use --init-draft-model-path for a new objective"
            )


def configure_objective(model, tokenizer, args):
    """Enable target-state reuse and freeze unused heads before FSDP wrapping."""
    if args.dspark_loss_type == "dpace":
        from specforge.utils import print_on_rank0

        print_on_rank0(
            f"DFlash2 D-PACE + selector: dpace_alpha={model.dpace_alpha}; "
            "dynamic position weights replace loss-decay-gamma."
        )
        return
    if args.dspark_loss_type not in TV_OBJECTIVES:
        return
    from .config import is_dflash2_config

    is_dflash2 = is_dflash2_config(model.draft_model.config)
    if args.tv_anchor_mode == "on-policy" and not is_dflash2:
        raise ValueError(
            "--tv-anchor-mode on-policy requires a native DFlash2 draft; "
            "DSpark is unsupported"
        )
    from transformers import AutoConfig

    from specforge.core.tv_acceptance import configure_tv_acceptance
    from specforge.utils import print_on_rank0

    config = AutoConfig.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    if args.tv_anchor_mode == "on-policy":
        from specforge.inference.target_engine.proposal_verification import (
            supports_qwen3_prefix_cache,
        )

        if not supports_qwen3_prefix_cache(config):
            raise ValueError(
                "On-policy TV beta currently requires a full-attention Qwen3 target"
            )
    eos_ids = getattr(config, "eos_token_id", tokenizer.eos_token_id)
    if eos_ids is None:
        eos_ids = ()
    elif isinstance(eos_ids, int):
        eos_ids = (eos_ids,)
    if is_dflash2:
        if args.dspark_loss_type != "tv-acceptance-beta":
            raise ValueError("DFlash2 requires tv-acceptance-beta in this entrypoint")
        model.configure_beta(
            temperature=args.tv_sampling_temperature,
            chunk_blocks=args.tv_objective_chunk_blocks,
            eos_token_ids=eos_ids,
            verification_batch_size=args.tv_verification_batch_size,
            anchor_mode=args.tv_anchor_mode,
            rollout_max_tokens=args.tv_rollout_max_tokens,
            max_sequence_length=args.max_length,
        )
        print_on_rank0(
            "DFlash2 TV beta: 8 native slots, 7 proposals; strict unary top-k "
            "selector sampling/replay, full-vocabulary target normalization."
        )
        if args.tv_anchor_mode == "on-policy":
            print_on_rank0(
                "On-policy anchors: actual sampled speculative round starts; "
                f"at most {args.tv_rollout_max_tokens} new tokens and "
                f"{args.num_anchors} rounds per prompt. Replay retains rejected "
                "proposal suffixes; fixed denominator 7. Conditional single-round "
                "gradient only; no gradient through the anchor visitation distribution."
            )
            if args.tv_verification_batch_size is not None:
                print_on_rank0(
                    "--tv-verification-batch-size is ignored by on-policy beta: "
                    "each round verifies its active prompt batch."
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
