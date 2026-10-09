"""Build legacy DSpark models and data, and apply warm-start compatibility."""

import os
import unicodedata
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from datasets import concatenate_datasets, load_dataset
from specforge.args import SGLangBackendArgs
from specforge.data.preprocessing import build_eagle3_dataset
from specforge.distributed import get_dp_group
from specforge.inference.target_engine.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.legacy.data_utils import (
    build_source_mixture_weights,
    prepare_dp_dataloaders,
)
from specforge.legacy.dspark import DSparkDraftModel
from specforge.utils import print_on_rank0

from .config import _apply_dspark_config


def _build_dialogue_token_class_lookup(tokenizer) -> torch.Tensor:
    """Classify vocabulary entries for cheap Chinese/English boundary routing.

    Classes are other/Han/Latin/number/punctuation/special.  The lookup is a
    non-persistent uint8 buffer, so it adds neither trainable parameters nor
    checkpoint size.
    """
    vocab_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)
    classes = torch.zeros(vocab_size, dtype=torch.uint8)
    # Decoding avoids misclassifying byte-level BPE spellings of Han tokens as
    # Latin. Chunking keeps the temporary singleton lists bounded.
    for start in range(0, vocab_size, 4096):
        stop = min(start + 4096, vocab_size)
        texts = tokenizer.batch_decode(
            [[token_id] for token_id in range(start, stop)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id, decoded in zip(range(start, stop), texts):
            if token_id in special_ids:
                classes[token_id] = 5
                continue
            text = decoded.strip()
            if not text:
                classes[token_id] = 4
                continue
            if any("\u4e00" <= char <= "\u9fff" for char in text):
                classes[token_id] = 1
            elif any(
                ("a" <= char.lower() <= "z") or "LATIN" in unicodedata.name(char, "")
                for char in text
            ):
                classes[token_id] = 2
            elif any(char.isdigit() for char in text):
                classes[token_id] = 3
            elif all(unicodedata.category(char)[0] in {"P", "S", "Z"} for char in text):
                classes[token_id] = 4
    return classes


def _apply_trainable_parameter_scope(
    draft_model: DSparkDraftModel, scope: str
) -> Tuple[int, int]:
    """Apply a named training scope before FSDP and optimizer construction."""
    if scope == "all":
        trainable = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        return trainable, 0
    if scope == "selector-only":
        if getattr(draft_model, "candidate_selector", None) is None:
            raise ValueError(
                "selector-only scope requires --selector-rank and --selector-top-k"
            )
        prefix = "candidate_selector."
    elif scope == "carh-only":
        if getattr(draft_model, "markov_head", None) is None:
            raise ValueError("carh-only scope requires an initialized Markov/CARH head")
        prefix = "markov_head."
    else:
        raise ValueError(f"unsupported trainable parameter scope: {scope}")

    trainable = 0
    frozen = 0
    for name, parameter in draft_model.named_parameters():
        enabled = name.startswith(prefix)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if trainable == 0:
        raise RuntimeError(f"{scope} scope selected no trainable parameters")
    return trainable, frozen


def build_models(args, device) -> Tuple[DFlashTargetModel, DSparkDraftModel]:
    """Build target model (backend wrapper) and DSpark draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using "
        f"{args.target_model_backend} backend"
    )

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    device_type = device.type

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=device_type if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
        if (
            hasattr(draft_config, "block_size")
            and draft_config.block_size != args.block_size
        ):
            print_on_rank0(
                f"Warning: config block_size ({draft_config.block_size}) differs from "
                f"command-line arg ({args.block_size}). Using config value."
            )
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    if args.elastic_horizon_enabled:
        original_block_size = int(draft_config.block_size)
        draft_config.block_size = int(args.elastic_long_horizon)
        print_on_rank0(
            "Elastic horizon: expanded fixed model block size from "
            f"{original_block_size} to K_max={draft_config.block_size}; "
            f"active horizons={args.elastic_short_horizon},"
            f"{args.elastic_long_horizon}."
        )

    _apply_dspark_config(draft_config, args)
    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DSparkDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}, "
        f"markov_rank={getattr(draft_config, 'markov_rank', 0)}, "
        f"markov_head_type={getattr(draft_config, 'markov_head_type', None)}, "
        f"selector_rank={getattr(draft_config, 'selector_rank', 0)}, "
        f"selector_top_k={getattr(draft_config, 'selector_top_k', 0)}, "
        "selector_margin_threshold="
        f"{getattr(draft_config, 'selector_margin_threshold', 0.0)}, "
        "selector_runtime_enabled="
        f"{getattr(draft_config, 'selector_runtime_enabled', True)}, "
        "conv_kernel_size="
        f"{draft_config.dflash_config.get('conv_kernel_size', 0)}, "
        "conv_group_size="
        f"{draft_config.dflash_config.get('conv_group_size', 0)}, "
        "conv_identity_init="
        f"{draft_config.dflash_config.get('conv_identity_init', True)}, "
        "conv_mode="
        f"{draft_config.dflash_config.get('conv_mode', 'legacy')}, "
        "conv_kernel_conditioning="
        f"{draft_config.dflash_config.get('conv_kernel_conditioning', 'input')}, "
        "conv_source_rank="
        f"{draft_config.dflash_config.get('conv_source_rank', 32)}, "
        "conv_apply_to="
        f"{draft_config.dflash_config.get('conv_apply_to', 'attention-mlp')}, "
        "conv_last_n_layers="
        f"{draft_config.dflash_config.get('conv_last_n_layers', 0)}, "
        "prefix_state_mixer_mode="
        f"{getattr(draft_config, 'prefix_state_mixer_mode', 'none')}, "
        "prefix_state_rank="
        f"{getattr(draft_config, 'prefix_state_rank', 0)}, "
        "block_summary_rank="
        f"{getattr(draft_config, 'block_summary_rank', 0)}, "
        "parallel_refiner_rank="
        f"{getattr(draft_config, 'parallel_refiner_rank', 0)}, "
        "parallel_refiner_steps="
        f"{getattr(draft_config, 'parallel_refiner_steps', 0)}, "
        f"enable_confidence_head={getattr(draft_config, 'enable_confidence_head', False)}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return target_model, draft_model


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    train_paths = [args.train_data_path, *args.additional_train_data_path]
    mixture_weights = args.train_data_mixture_weights
    if mixture_weights is None:
        mixture_weights = tuple(1.0 / len(train_paths) for _ in train_paths)
    if len(mixture_weights) != len(train_paths):
        raise ValueError(
            "--train-data-mixture-weights must contain one value for the primary "
            "and every additional training source"
        )
    if any(weight < 0 for weight in mixture_weights) or sum(mixture_weights) <= 0:
        raise ValueError("training data mixture weights must have positive mass")
    mixture_weights = tuple(weight / sum(mixture_weights) for weight in mixture_weights)
    source_languages = args.dialogue_source_languages
    if source_languages is None:
        source_languages = tuple(f"source-{index}" for index in range(len(train_paths)))
    if len(source_languages) != len(train_paths):
        raise ValueError(
            "--dialogue-source-languages must contain one label for the primary "
            "and every additional training source"
        )
    cache_params_string = (
        f"{train_paths}-{mixture_weights}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    min_loss_tokens = 2 * args.block_size
    processed_sources = []
    source_sizes = []
    for source_index, path in enumerate(train_paths):
        train_dataset = load_dataset("json", data_files=path)["train"]
        source_cache_key = f"{cache_key}-source-{source_index}"
        processed = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=source_cache_key,
            num_proc=args.build_dataset_num_proc,
        )
        original_size = len(processed)
        processed = processed.filter(lambda x: x["loss_mask"].sum() >= min_loss_tokens)
        if len(processed) == 0:
            raise ValueError(f"training source is empty after filtering: {path}")
        if "__source_id" in processed.column_names:
            processed = processed.remove_columns("__source_id")
        processed = processed.add_column("__source_id", [source_index] * len(processed))
        processed_sources.append(processed)
        source_sizes.append(len(processed))
        print_on_rank0(
            f"Filtered train source {source_index}: {original_size} -> "
            f"{len(processed)} samples; mass={mixture_weights[source_index]:.4f}; "
            f"language={source_languages[source_index]}; path={path}"
        )
    train_eagle3_dataset = (
        processed_sources[0]
        if len(processed_sources) == 1
        else concatenate_datasets(processed_sources)
    )
    # Stable IDs survive distributed sampling and allow independently exported
    # teacher trajectories to be joined without relying on dataloader order.
    train_eagle3_dataset = train_eagle3_dataset.add_column(
        "__sample_id", list(range(len(train_eagle3_dataset)))
    )
    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        weighted_sampling=len(processed_sources) > 1,
        sampler_seed=args.seed,
    )
    if len(processed_sources) > 1:
        initial_weights = build_source_mixture_weights(source_sizes, mixture_weights)
        train_dataloader.sampler.update_weights(initial_weights)

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


def _patch_flex_attention_triton_backend() -> None:
    """Force transformer :func:`flex_attention` onto the general Triton kernel
    instead of the Inductor "flex_decoding" path.

    Inductor's flex-decoding kernel derives its autotune block size from
    ``BLOCK_M = next_pow2(query_len * gqa_shared_heads)`` and rejects every
    candidate that does not divide the sparse mask block size
    (``SPARSE_Q_BLOCK_SIZE % BLOCK_M != 0``).  For some batches (query length
    roughly in (16, 128) with GQA) this filters out *all* candidates, and
    because the decode path has no ATen fallback, training crashes in
    ``autotune_select_algorithm`` with ``NoValidChoicesError``.  Forcing
    ``BACKEND="TRITON"`` routes attention to the general Triton kernel, which
    has no such divisibility requirement and always has a valid configuration.
    """
    import transformers.integrations.flex_attention as _tfa

    if getattr(_tfa, "_specforge_force_flex_triton", False):
        return
    _orig = _tfa.compile_friendly_flex_attention

    def _force_triton(query, key, value, training=False, **kwargs):
        # Transformers may pass kernel_options=None (the Qwen3 path does), in
        # which case torch's flex_attention defaults BACKEND to AUTO internally.
        # Always inject an explicit BACKEND so the general Triton kernel (which
        # has no divisibility requirement) is selected.
        ko = dict(kwargs["kernel_options"]) if kwargs.get("kernel_options") else {}
        ko["BACKEND"] = "TRITON"
        kwargs["kernel_options"] = ko
        return _orig(query, key, value, training=training, **kwargs)

    _tfa.compile_friendly_flex_attention = _force_triton
    _tfa._specforge_force_flex_triton = True
    print_on_rank0(
        "Flex attention: forcing BACKEND=TRITON (avoids Inductor "
        "flex-decoding NoValidChoicesError on dynamic short-query batches)."
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


def _load_parallel_refiner_teacher(
    draft_model: DSparkDraftModel, checkpoint_path: str, freeze: bool
) -> None:
    """Overlay only HAPR weights, preserving the stronger CARH student."""
    if getattr(draft_model, "parallel_refiner", None) is None:
        raise ValueError("parallel refiner teacher path requires an enabled refiner")
    teacher_model = DSparkDraftModel.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16
    )
    teacher_refiner = getattr(teacher_model, "parallel_refiner", None)
    if teacher_refiner is None:
        raise ValueError(
            f"teacher checkpoint has no parallel refiner: {checkpoint_path}"
        )
    incompatible = draft_model.parallel_refiner.load_state_dict(
        teacher_refiner.state_dict(), strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(
            (
                "advantage_class_proj.",
                "advantage_value_proj.",
                "advantage_temperature_proj.",
            )
        )
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Parallel refiner teacher is incompatible: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    del teacher_model
    if freeze:
        for name, parameter in draft_model.parallel_refiner.named_parameters():
            parameter.requires_grad = name.startswith(
                (
                    "advantage_class_proj.",
                    "advantage_value_proj.",
                    "advantage_temperature_proj.",
                )
            )
    print_on_rank0(
        "Loaded training-only parallel refiner teacher from "
        f"{checkpoint_path}; freeze={freeze}"
    )
