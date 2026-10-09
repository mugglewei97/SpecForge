"""Weight-only initialization for DSpark post-training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from specforge.legacy.dspark import DSparkDraftModel


def prepare_warm_start(args):
    """Resolve a local exported checkpoint before allocating distributed models."""
    if not args.init_draft_model_path:
        return
    if args.resume:
        raise ValueError("--resume and --init-draft-model-path are mutually exclusive")
    source = Path(args.init_draft_model_path).expanduser().resolve()
    if not (source / "config.json").is_file():
        raise FileNotFoundError(
            f"Initialization checkpoint has no config.json: {source}"
        )
    if not any(
        (source / name).is_file()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    ):
        raise FileNotFoundError(
            f"Initialization checkpoint has no model weights: {source}"
        )
    if Path(args.output_dir).expanduser().resolve() == source:
        raise ValueError(
            "Post-training requires an output-dir different from the initialization checkpoint"
        )
    args.init_draft_model_path = str(source)
    # Architecture and capture layers come from the learned drafter, including
    # when the launch command also supplies a generic scratch config.
    args.draft_config_path = str(source / "config.json")


def load_draft_config(path):
    """Support native DSpark configs and older Qwen3-based DSpark exports."""
    from transformers import AutoConfig, PretrainedConfig

    from specforge.legacy.dspark import DSparkConfig

    config_dict, _ = PretrainedConfig.get_config_dict(path)
    if config_dict.get("model_type") == "dspark":
        return DSparkConfig.from_dict(config_dict)
    return AutoConfig.from_pretrained(path)


def load_checkpoint_model(path, *, dtype=None):
    """Load complete draft weights; optimizer/scheduler/progress are never read."""
    import torch

    from specforge.legacy.dspark import DSparkDraftModel

    model, info = DSparkDraftModel.from_pretrained(
        path,
        config=load_draft_config(path),
        torch_dtype=torch.bfloat16 if dtype is None else dtype,
        local_files_only=True,
        weights_only=True,
        output_loading_info=True,
    )
    errors = {
        key: info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(key)
    }
    if errors:
        raise ValueError(
            f"Incomplete or incompatible DSpark checkpoint {path}: {errors}"
        )
    return model


def resolve_mask_token_id(args, draft_model, tokenizer):
    """Prefer an explicit override, then the trained model's mask token."""
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        mask_token_id = draft_model.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None or not 0 <= mask_token_id < draft_model.config.vocab_size:
        raise ValueError(
            "mask_token_id must be within the draft/target embedding vocabulary"
        )
    return mask_token_id


def initialize_draft_weights(draft_model, args):
    """Overlay learned weights, retaining explicitly requested legacy expansions."""
    loaded_model = load_checkpoint_model(
        args.init_draft_model_path, dtype=next(draft_model.parameters()).dtype
    )
    _load_draft_state_allowing_selector_expansion(
        draft_model,
        loaded_model,
        allow_conv_position_pruning=args.allow_conv_position_pruning,
    )


def _load_draft_state_allowing_selector_expansion(
    draft_model: DSparkDraftModel,
    loaded_model: DSparkDraftModel,
    allow_conv_position_pruning: bool = False,
) -> None:
    """Load a warm start that may predate optional research modules.

    All shared weights remain strict. Newly constructed research modules may be
    missing, and DynamicConv-only parameters may be dropped when an SPSM run
    explicitly replaces that backbone component.
    """
    import torch

    from specforge.utils import print_on_rank0

    loaded_state = loaded_model.state_dict()
    replacing_dynamic_conv = (
        getattr(draft_model, "prefix_state_mixer", None) is not None
        and getattr(loaded_model, "dynamic_conv_enabled", False)
        and not getattr(draft_model, "dynamic_conv_enabled", False)
    )
    dropped_dynamic_conv = []
    if replacing_dynamic_conv:
        dropped_dynamic_conv = [
            key
            for key in loaded_state
            if ".attention_conv." in key or ".mlp_conv." in key
        ]
        loaded_state = {
            key: value
            for key, value in loaded_state.items()
            if key not in dropped_dynamic_conv
        }
    target_state = draft_model.state_dict()
    if allow_conv_position_pruning:
        removed = [
            key
            for key in loaded_state
            if key not in target_state
            and (".attention_conv." in key or ".mlp_conv." in key)
        ]
        for key in removed:
            del loaded_state[key]
        print_on_rank0(f"Explicit conv-position pruning: removed {removed}")
    expanded_depth_embeddings = []
    for key, value in list(loaded_state.items()):
        target_value = target_state.get(key)
        if (
            target_value is not None
            and key.endswith("depth_embedding.weight")
            and value.ndim == target_value.ndim == 2
            and value.shape[1:] == target_value.shape[1:]
            and value.shape[0] < target_value.shape[0]
        ):
            expanded = value.new_empty(target_value.shape)
            expanded[: value.shape[0]].copy_(value)
            # New depths start close to the deepest learned K=7 state while
            # retaining a tiny symmetry-breaking perturbation.
            tail = value[-1:].expand(target_value.shape[0] - value.shape[0], -1)
            expanded[value.shape[0] :].copy_(tail)
            expanded[value.shape[0] :].add_(
                torch.randn_like(expanded[value.shape[0] :]) * 1e-4
            )
            loaded_state[key] = expanded
            expanded_depth_embeddings.append(key)
    incompatible = draft_model.load_state_dict(loaded_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    adding_second_predecessor = (
        getattr(draft_model.markov_head, "predecessor_count", 1) >= 2
        and getattr(loaded_model.markov_head, "predecessor_count", 1) == 1
        and getattr(loaded_model.markov_head, "markov_head_type", "") == "carh"
    )
    adding_third_predecessor = (
        getattr(draft_model.markov_head, "predecessor_count", 1) == 3
        and getattr(loaded_model.markov_head, "predecessor_count", 1) < 3
        and getattr(loaded_model.markov_head, "markov_head_type", "") == "carh"
    )
    adding_sampled_prefix_memory = (
        int(getattr(draft_model.markov_head, "sampled_prefix_memory_rank", 0)) > 0
        and int(getattr(loaded_model.markov_head, "sampled_prefix_memory_rank", 0)) == 0
        and getattr(loaded_model.markov_head, "markov_head_type", "") == "carh"
    )
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(
            (
                "candidate_selector.",
                "recall_correction.",
                "target_layer_fusion_weights",
                "target_layer_fusion_proj.",
                "prefix_state_mixer.",
                "block_summary.",
                "parallel_refiner.",
            )
        )
        and ".local_transition_attention." not in key
        and not (
            adding_second_predecessor and key == "markov_head.second_prev_proj.weight"
        )
        and not (
            adding_third_predecessor and key == "markov_head.third_prev_proj.weight"
        )
        and not (
            adding_sampled_prefix_memory
            and key.startswith("markov_head.sampled_prefix_memory.")
        )
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Warm-start state is incompatible: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    if incompatible.missing_keys:
        print_on_rank0(
            "Warm start predates optional research modules; initialized the "
            f"following parameters from scratch: {incompatible.missing_keys}"
        )
    if dropped_dynamic_conv:
        print_on_rank0(
            "SPSM replacement warm start: discarded DynamicConv-only "
            f"parameters: {dropped_dynamic_conv}"
        )
    if expanded_depth_embeddings:
        print_on_rank0(
            "Elastic warm start: expanded depth embeddings for K_max: "
            f"{expanded_depth_embeddings}"
        )
