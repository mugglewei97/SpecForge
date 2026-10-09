"""Resolve CLI overrides while retaining values stored in model checkpoints."""

# These options override the checkpoint only when explicitly supplied on CLI.
HEAD_DEFAULTS = {
    "markov_head_type": "vanilla",
    "carh_predecessor_count": 1,
    "carh_predecessor_context_mode": "none",
    "carh_sampled_prefix_memory_rank": 0,
    "selector_rank": 0,
    "selector_top_k": 0,
    "selector_runtime_enabled": True,
    "selector_margin_threshold": 0.0,
    "recall_correction_rank": 0,
    "recall_correction_gate_bias": -2.0,
    "prefix_state_mixer_mode": "none",
    "prefix_state_rank": 128,
    "prefix_state_retention_bias": 2.0,
    "prefix_state_update_bias": -1.0,
    "prefix_state_gate_bias": -2.0,
    "prefix_state_residual_scale": 0.1,
    "block_summary_rank": 0,
    "block_summary_source": "anchor",
    "block_summary_gate_mode": "position",
    "parallel_refiner_rank": 0,
    "parallel_refiner_steps": 0,
    "parallel_refiner_gate_bias": -1.0,
    "parallel_refiner_second_step_bias": -1.0,
    "parallel_refiner_residual_scale": 0.1,
    "parallel_refiner_runtime_enabled": True,
    "refiner_advantage_temperature_conditioned": False,
}

LOCAL_TRANSITION_DEFAULTS = {
    "local_transition_heads": 4,
    "local_transition_window": 2,
    "local_transition_residual_scale": 0.1,
    "local_transition_gate_bias": -1.0,
    "local_transition_conv_decay_start_ratio": 0.10,
    "local_transition_conv_decay_end_ratio": 0.70,
    "local_transition_final_conv_scale": 0.0,
}

METHOD_OVERRIDES = (
    "target_layer_fusion_mode",
    "target_layer_fusion_rank",
    "target_layer_fusion_source_dropout",
    "target_layer_fusion_residual_scale",
    "conv_kernel_size",
    "conv_group_size",
    "local_transition_rank",
    *LOCAL_TRANSITION_DEFAULTS,
    "conv_mode",
    "conv_kernel_conditioning",
    "conv_source_rank",
    "conv_apply_to",
    "conv_last_n_layers",
    "conv_residual_scale",
    "conv_gate_bias",
    "conv_freeze_identity",
)


def _apply_dspark_config(draft_config, args) -> None:
    """Apply defaults, explicit CLI overrides, and nested DFlash options."""
    for name in (
        "markov_rank",
        "enable_confidence_head",
        "confidence_head_with_markov",
        "carh_gate_bias",
    ):
        if getattr(draft_config, name, None) is None:
            setattr(draft_config, name, getattr(args, name))
    draft_config.elastic_horizon_enabled = bool(args.elastic_horizon_enabled)
    draft_config.elastic_short_horizon = int(args.elastic_short_horizon)
    draft_config.elastic_long_horizon = int(args.elastic_long_horizon)
    for name, default in HEAD_DEFAULTS.items():
        value = getattr(args, name, None)
        if value is not None:
            setattr(draft_config, name, value)
        elif not hasattr(draft_config, name):
            setattr(draft_config, name, default)

    method_config = dict(getattr(draft_config, "dflash_config", None) or {})
    for name in METHOD_OVERRIDES:
        value = getattr(args, name)
        if value is not None:
            method_config[name] = value
    if int(method_config.get("local_transition_rank", 0) or 0) > 0:
        for name, default in LOCAL_TRANSITION_DEFAULTS.items():
            method_config.setdefault(name, default)
        # A newly expanded checkpoint starts with the full convolution teacher.
        method_config["local_transition_conv_scale"] = 1.0
    if bool(method_config.get("conv_kernel_size", 0)) != bool(
        method_config.get("conv_group_size", 0)
    ):
        raise ValueError(
            "--conv-kernel-size and --conv-group-size must be enabled or "
            "disabled together"
        )
    method_config["conv_identity_init"] = bool(args.conv_identity_init)
    conditioning = method_config.get("conv_kernel_conditioning", "input")
    if conditioning not in {
        "input",
        "output",
        "source-aware",
        "grouped16",
        "grouped64",
        "low-rank64",
        "static",
    }:
        raise ValueError(f"Unsupported conv_kernel_conditioning={conditioning!r}")
    if conditioning != "input" and (
        method_config.get("conv_mode", "legacy") != "legacy"
        or not method_config.get("conv_kernel_size", 0)
    ):
        raise ValueError(
            "Alternative kernel conditioning requires enabled legacy convolution"
        )
    if int(method_config.get("conv_source_rank", 32)) < 1:
        raise ValueError("conv_source_rank must be positive")
    draft_config.dflash_config = method_config
