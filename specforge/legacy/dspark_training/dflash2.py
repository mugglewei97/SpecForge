"""Native DFlash2 wiring for the standalone TV-beta training lifecycle."""


def prepare_dflash2_config(config, args):
    if args.dspark_loss_type != "tv-acceptance-beta":
        raise ValueError(
            "This entrypoint supports DFlash2 only with --dspark-loss-type tv-acceptance-beta"
        )
    method = dict(getattr(config, "dflash_config", None) or {})
    top_level = getattr(config, "block_size", None)
    nested = method.get("block_size")
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError("DFlash2 top-level and nested block_size disagree")
    block_size = top_level if top_level is not None else nested
    if block_size != 8:
        raise ValueError(
            "DFlash2 tv-acceptance-beta requires block_size=8 (anchor + 7 proposals)"
        )
    if method.get("shift_label", False) or method.get("pure_draft_prefix_len", 0):
        raise ValueError(
            "DFlash2 TV beta requires the native same-position anchor layout"
        )
    # These are architecture switches shared with the legacy CLI, but DFlash2
    # stores them together in dflash_config, not in DSpark's top-level fields.
    for name in (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ):
        value = getattr(args, name, None)
        if value is not None:
            method[name] = value
    for name in (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ):
        value = method.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"DFlash2 requires a positive integer dflash_config.{name}"
            )

    # The native wrapper implements pure beta. Reject active DSpark-only
    # extensions rather than silently creating a different experiment.
    unsupported = []
    for name, value in vars(args).items():
        if (
            name.endswith("_alpha")
            and name.startswith(
                (
                    "on_policy_",
                    "carh_",
                    "parallel_refiner_",
                    "conv_source_",
                    "branch_value_",
                )
            )
            and value
        ):
            unsupported.append(name)
    for name in (
        "branch_value_mode",
        "refiner_advantage_mode",
        "multi_teacher_oracle_mode",
    ):
        if getattr(args, name, "none") != "none":
            unsupported.append(name)
    for name in (
        "parallel_refiner_teacher_path",
        "freeze_dynamic_conv_during_transition",
        "local_transition_rank",
        "bv_loss_alpha",
        "on_policy_repair_value",
        "conv_source_semantic_alpha",
        "carh_reference_calibration_alpha",
    ):
        if getattr(args, name, None):
            unsupported.append(name)
    if unsupported:
        raise ValueError(
            f"DFlash2 TV beta does not support DSpark auxiliary modes: {sorted(set(unsupported))}"
        )
    config.block_size = block_size
    method["block_size"] = block_size
    config.dflash_config = method
    config._attn_implementation = args.attention_backend
    args.block_size = block_size


def build_dflash2_training_model(args, draft_model, target_components, mask_token_id):
    from specforge.core.dflash2_tv_acceptance import OnlineDFlash2TVBetaModel

    return OnlineDFlash2TVBetaModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
    )


def dflash2_beta_resume_contract(draft_model):
    """Persist non-tensor proposal semantics as well as the parameter layout."""
    return {
        "policy": "strict-topk-selector-beta-v1",
        "block_size": draft_model.block_size,
        "proposal_length": 7,
        "dflash_config": dict(draft_model.config.dflash_config),
    }


def validate_dflash2_beta_resume(state, draft_model):
    if state.get("dflash2_tv_beta") != dflash2_beta_resume_contract(draft_model):
        raise ValueError(
            "Cannot resume DFlash2 TV beta with changed or missing proposal semantics; "
            "use --init-draft-model-path with a new output directory"
        )
