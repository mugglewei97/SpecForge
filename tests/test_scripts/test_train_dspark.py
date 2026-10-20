"""CLI/configuration contracts for the standalone compatibility entrypoint."""

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from specforge.legacy.dspark_training.arguments import parse_args
from specforge.legacy.dspark_training.config import _apply_dspark_config
from specforge.legacy.dspark_training.initialization import (
    prepare_warm_start,
    resolve_mask_token_id,
)
from specforge.legacy.dspark_training.objective import (
    validate_objective_args,
    validate_objective_resume,
)


def training_args(*extra):
    return parse_args(
        [
            "--target-model-path",
            "target",
            "--train-data-path",
            "data.jsonl",
            "--output-dir",
            "output",
            *extra,
        ]
    )


def test_entrypoint_help_needs_no_training_dependencies():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import runpy, sys
for name in ('torch', 'sglang', 'transformers', 'datasets', 'accelerate'):
    sys.modules[name] = None
sys.argv = ['scripts/train_dspark.py', '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
""",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--tv-sampling-temperature" in result.stdout
    assert "--tv-temperature" in result.stdout
    assert "--sglang-mem-fraction-static" in result.stdout
    assert "--dpace-alpha" in result.stdout
    assert "--tv-anchor-mode" in result.stdout
    assert "--tv-rollout-max-tokens" in result.stdout


def test_native_dpace_cli_defaults_and_override():
    args = training_args("--dspark-loss-type", "dpace")
    validate_objective_args(args)
    assert args.dpace_alpha == 0.5
    for alpha in (0.0, 0.25, 1.0):
        args = training_args("--dspark-loss-type", "dpace", "--dpace-alpha", str(alpha))
        validate_objective_args(args)
        assert args.dpace_alpha == alpha


@pytest.mark.parametrize("alpha", ["nan", "inf", "-inf", "-0.1", "1.1"])
def test_native_dpace_rejects_invalid_alpha(alpha):
    args = training_args("--dspark-loss-type", "dpace", f"--dpace-alpha={alpha}")
    with pytest.raises(ValueError, match="dpace-alpha"):
        validate_objective_args(args)


@pytest.mark.parametrize("objective", ["dflash2", "dpace"])
def test_native_dflash2_rejects_dspark_pace_mode(objective):
    args = training_args("--dspark-loss-type", objective, "--pace-mode", "token")
    with pytest.raises(ValueError, match="--pace-mode is a DSpark option"):
        validate_objective_args(args)


def test_training_cli_and_objective_defaults_remain_compatible():
    legacy = training_args()
    assert legacy.dspark_loss_type == "ce-l1"
    validate_objective_resume({}, legacy)
    args = training_args(
        "--dspark-loss-type",
        "tv-acceptance",
        "--block-size",
        "7",
        "--num-anchors",
        "512",
        "--batch-size",
        "4",
        "--target-model-backend",
        "sglang",
        "--sglang-mem-fraction-static",
        "0.3",
    )
    validate_objective_args(args)
    assert (args.block_size, args.num_anchors, args.batch_size) == (7, 512, 4)
    assert args.tv_sampling_temperature == 1.0
    assert args.tv_objective_chunk_blocks == 8
    assert args.tv_verification_batch_size is None
    assert args.tv_prefix_mode == "teacher-forced"
    assert args.tv_anchor_mode == "random"
    assert args.tv_rollout_max_tokens == 256


@pytest.mark.parametrize("flag", ["--tv-temperature", "--tv-sampling-temperature"])
def test_temperature_alias_and_obsolete_verification_option(flag):
    args = training_args(
        "--dspark-loss-type",
        "tv-acceptance",
        flag,
        "0.7",
        "--tv-verification-batch-size",
        "4",
    )
    validate_objective_args(args)
    assert args.tv_sampling_temperature == 0.7


@pytest.mark.parametrize(
    "flag,value,match",
    [
        ("--tv-sampling-temperature", "nan", "temperature"),
        ("--tv-sampling-temperature", "0", "temperature"),
        ("--tv-objective-chunk-blocks", "0", "chunk"),
        ("--tv-temperature", "-1", "temperature"),
        ("--tp-size", "2", "tp-size"),
        ("--accumulation-steps", "2", "normalization"),
        ("--micro-batch-size", "1", "normalization"),
        ("--netprefix-mode", "baseline", "standalone"),
    ],
)
def test_objective_rejects_incompatible_options(flag, value, match):
    args = training_args("--dspark-loss-type", "tv-acceptance", flag, value)
    with pytest.raises(ValueError, match=match):
        validate_objective_args(args)


@pytest.mark.parametrize("as_namespace", [False, True])
def test_resume_accepts_saved_args_and_rejects_objective_changes(as_namespace):
    args = training_args("--dspark-loss-type", "tv-acceptance")
    saved = vars(args).copy()
    wrap = (lambda values: argparse.Namespace(**values)) if as_namespace else dict
    validate_objective_resume(wrap(saved), args)
    for key, value in (
        ("dspark_loss_type", "ce-l1"),
        ("tv_sampling_temperature", 0.7),
        ("tv_prefix_mode", "sampled"),
    ):
        with pytest.raises(ValueError, match=key):
            validate_objective_resume(wrap({**saved, key: value}), args)
    saved.pop("tv_prefix_mode")
    with pytest.raises(ValueError, match="tv_prefix_mode"):
        validate_objective_resume(wrap(saved), args)


def test_teacher_forced_resume_allows_memory_tuning():
    args = training_args("--dspark-loss-type", "tv-acceptance")
    saved = {
        **vars(args),
        "tv_objective_chunk_blocks": 32,
        "tv_verification_batch_size": 4,
    }
    validate_objective_resume(saved, args)


def test_beta_cli_resume_semantics_and_memory_tuning():
    args = training_args("--dspark-loss-type", "tv-acceptance-beta")
    validate_objective_args(args)
    assert args.tv_prefix_mode == "sampled-beta-v1"
    saved = {
        **vars(args),
        "tv_objective_chunk_blocks": 32,
        "tv_verification_batch_size": 64,
    }
    validate_objective_resume(saved, args)
    for key, value in (
        ("dspark_loss_type", "tv-acceptance"),
        ("tv_sampling_temperature", 0.7),
        ("tv_prefix_mode", "teacher-forced"),
    ):
        with pytest.raises(ValueError, match=key):
            validate_objective_resume({**saved, key: value}, args)
    saved.pop("tv_prefix_mode")
    with pytest.raises(ValueError, match="tv_prefix_mode"):
        validate_objective_resume(saved, args)


@pytest.mark.parametrize(
    "flag,value,match",
    [
        ("--tv-verification-batch-size", "0", "verification"),
        ("--tv-verification-batch-size", "-1", "verification"),
        ("--tv-temperature", "nan", "temperature"),
        ("--tp-size", "2", "tp-size"),
        ("--accumulation-steps", "2", "normalization"),
        ("--netprefix-mode", "baseline", "standalone"),
    ],
)
def test_beta_cli_rejects_unsupported_options(flag, value, match):
    with pytest.raises(ValueError, match=match):
        validate_objective_args(
            training_args("--dspark-loss-type", "tv-acceptance-beta", flag, value)
        )


def on_policy_args(*extra):
    return training_args(
        "--dspark-loss-type",
        "tv-acceptance-beta",
        "--tv-anchor-mode",
        "on-policy",
        *extra,
    )


@pytest.mark.parametrize("backend", ["hf", "sglang"])
def test_on_policy_cli_preserves_beta_and_limits_rollout(backend):
    args = on_policy_args("--target-model-backend", backend)
    validate_objective_args(args)
    assert args.tv_prefix_mode == "sampled-beta-v1"
    assert args.tv_sampling_temperature == 1.0
    assert args.tv_rollout_max_tokens == 256
    assert args.num_anchors == 512


@pytest.mark.parametrize(
    "flag,value,match",
    [
        ("--tv-rollout-max-tokens", "0", "rollout-max-tokens"),
        ("--tv-rollout-max-tokens", "-1", "rollout-max-tokens"),
        ("--num-anchors", "0", "num-anchors"),
        ("--max-length", "1", "max-length"),
        ("--dspark-loss-type", "dflash2", "tv-acceptance-beta"),
        ("--dspark-loss-type", "tv-acceptance", "tv-acceptance-beta"),
    ],
)
def test_on_policy_cli_rejects_invalid_scope_and_budget(flag, value, match):
    with pytest.raises(ValueError, match=match):
        validate_objective_args(on_policy_args(flag, value))


@pytest.mark.parametrize("as_namespace", [False, True])
def test_on_policy_resume_preserves_sampling_and_rollout_contract(as_namespace):
    args = on_policy_args()
    validate_objective_args(args)
    saved = vars(args).copy()
    wrap = argparse.Namespace if as_namespace else dict
    validate_objective_resume(wrap(**saved), args)
    for key, value in (
        ("tv_anchor_mode", "random"),
        ("tv_sampling_temperature", 0.7),
        ("tv_rollout_max_tokens", 128),
        ("num_anchors", 256),
        ("max_length", 2048),
    ):
        with pytest.raises(ValueError, match=key):
            validate_objective_resume(wrap(**{**saved, key: value}), args)
    for key in ("tv_anchor_mode", "tv_rollout_max_tokens", "num_anchors", "max_length"):
        incomplete = saved.copy()
        incomplete.pop(key)
        with pytest.raises(ValueError, match=key):
            validate_objective_resume(wrap(**incomplete), args)
    random = training_args("--dspark-loss-type", "tv-acceptance-beta")
    validate_objective_args(random)
    with pytest.raises(ValueError, match="tv_anchor_mode"):
        validate_objective_resume(vars(args), random)


def test_old_random_beta_checkpoint_remains_compatible():
    args = training_args("--dspark-loss-type", "tv-acceptance-beta")
    validate_objective_args(args)
    saved = vars(args).copy()
    saved.pop("tv_anchor_mode")
    saved.pop("tv_rollout_max_tokens")
    validate_objective_resume(saved, args)


def test_on_policy_resume_allows_memory_tuning():
    args = on_policy_args()
    validate_objective_args(args)
    validate_objective_resume(
        {
            **vars(args),
            "tv_objective_chunk_blocks": 32,
            "tv_verification_batch_size": 2,
        },
        args,
    )


@pytest.mark.parametrize("architecture", ["DSparkDraftModel", "DFlash2DraftModel"])
@pytest.mark.parametrize(
    "model_type,layer_type,supported",
    [
        ("qwen3", "full_attention", True),
        ("qwen3", "sliding_attention", False),
        ("llama", "full_attention", False),
    ],
)
def test_on_policy_wiring_requires_full_attention_qwen3(
    monkeypatch, architecture, model_type, layer_type, supported
):
    import transformers

    from specforge.legacy.dspark_training.objective import configure_objective

    target_config = SimpleNamespace(
        model_type=model_type,
        eos_token_id=[1, 2],
        layer_types=[layer_type],
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: target_config
    )
    monkeypatch.setattr("specforge.utils.print_on_rank0", lambda *args: None)
    captured = {}
    model = SimpleNamespace(
        draft_model=SimpleNamespace(
            config=SimpleNamespace(architectures=[architecture])
        ),
        configure_beta=lambda **kwargs: captured.update(kwargs),
    )

    def configure_dspark(actual_model, **kwargs):
        assert actual_model is model
        captured.update(kwargs)

    monkeypatch.setattr(
        "specforge.core.tv_acceptance.configure_tv_acceptance", configure_dspark
    )
    args = on_policy_args("--max-length", "1024", "--tv-rollout-max-tokens", "128")
    if not supported:
        with pytest.raises(ValueError, match="full-attention Qwen3"):
            configure_objective(model, SimpleNamespace(eos_token_id=1), args)
        assert not captured
        return
    configure_objective(model, SimpleNamespace(eos_token_id=1), args)
    expected = {
        "temperature": 1.0,
        "chunk_blocks": 8,
        "eos_token_ids": [1, 2],
        "verification_batch_size": None,
        "anchor_mode": "on-policy",
        "rollout_max_tokens": 128,
        "max_sequence_length": 1024,
    }
    if architecture == "DSparkDraftModel":
        expected["beta"] = True
    assert captured == expected


def test_on_policy_dflash2_config_keeps_native_eight_slots():
    from specforge.legacy.dspark_training.dflash2 import prepare_dflash2_config

    def config(block_size):
        return SimpleNamespace(
            block_size=block_size,
            dflash_config={
                "block_size": block_size,
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": 4,
                "selector_top_k": 8,
            },
        )

    args = on_policy_args()
    draft_config = config(8)
    prepare_dflash2_config(draft_config, args)
    assert args.block_size == draft_config.block_size == 8
    with pytest.raises(ValueError, match="block_size=8"):
        prepare_dflash2_config(config(7), on_policy_args())
    with pytest.raises(ValueError, match="tv-acceptance-beta"):
        prepare_dflash2_config(
            config(8), on_policy_args("--dspark-loss-type", "dflash2")
        )


def test_config_preserves_checkpoint_values_and_explicit_cli_overrides():
    config = SimpleNamespace(
        markov_rank=32,
        selector_rank=64,
        selector_runtime_enabled=True,
        prefix_state_rank=256,
        dflash_config={"target_layer_fusion_rank": 16},
    )
    args = training_args("--selector-rank", "0", "--no-selector-runtime-enabled")
    _apply_dspark_config(config, args)
    assert config.markov_rank == 32
    assert config.selector_rank == 0
    assert config.selector_runtime_enabled is False
    assert config.prefix_state_rank == 256
    assert config.markov_head_type == "vanilla"
    assert config.dflash_config["target_layer_fusion_rank"] == 16


def test_config_retains_local_transition_defaults_and_convolution_checks():
    config = SimpleNamespace()
    _apply_dspark_config(config, training_args("--local-transition-rank", "8"))
    assert config.dflash_config["local_transition_heads"] == 4
    assert config.dflash_config["local_transition_conv_scale"] == 1.0
    with pytest.raises(ValueError, match="enabled or disabled together"):
        _apply_dspark_config(
            SimpleNamespace(), training_args("--conv-kernel-size", "3")
        )


def test_checkpoint_exports_models_relative_to_relocated_module(tmp_path, monkeypatch):
    import torch

    from specforge.legacy.dspark_training import checkpoint

    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)
    monkeypatch.setattr(
        checkpoint.FSDP, "state_dict_type", lambda *args: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        checkpoint,
        "save_fsdp_optimizer_state",
        lambda optimizer, model, path: torch.save({}, path),
    )
    monkeypatch.setattr(checkpoint, "print_on_rank0", lambda *args: None)

    def save_model(directory, state_dict):
        assert set(state_dict) == {"weight"}
        (Path(directory) / "config.json").write_text(
            json.dumps({"architectures": ["DSparkDraftModel"]})
        )

    model = SimpleNamespace(state_dict=lambda: {"draft_model.weight": torch.ones(1)})
    checkpoint.save_checkpoint(
        argparse.Namespace(output_dir=str(tmp_path)),
        1,
        3,
        model,
        SimpleNamespace(save_pretrained=save_model),
        None,
    )
    saved = tmp_path / "epoch_1_step_3"
    source = Path(checkpoint.__file__).resolve().parents[1]
    for filename in (
        "dspark.py",
        "dflash.py",
        "dflash2.py",
        "neighbor_residual.py",
        "light_conv_kernel.py",
    ):
        assert (saved / filename).read_bytes() == (source / filename).read_bytes()
    assert json.loads((saved / "config.json").read_text())["architectures"] == [
        "Qwen3DSparkModel"
    ]


def test_warm_start_selects_checkpoint_config_and_rejects_resume(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    args = training_args("--init-draft-model-path", str(source))
    with pytest.raises(FileNotFoundError, match="no model weights"):
        prepare_warm_start(args)
    (source / "model.safetensors").touch()
    prepare_warm_start(args)
    assert args.draft_config_path == str(source / "config.json")
    args.resume = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_warm_start(args)
    args.resume = False
    args.output_dir = str(source)
    with pytest.raises(ValueError, match="different"):
        prepare_warm_start(args)


@pytest.mark.parametrize(
    "explicit,saved,expected", [(None, 13, 13), (7, 13, 7), (None, None, 9)]
)
def test_mask_token_preserves_checkpoint_unless_explicitly_overridden(
    explicit, saved, expected
):
    args = SimpleNamespace(mask_token_id=explicit)
    model = SimpleNamespace(mask_token_id=saved, config=SimpleNamespace(vocab_size=16))
    assert (
        resolve_mask_token_id(args, model, SimpleNamespace(mask_token_id=9)) == expected
    )
    args.mask_token_id = 16
    with pytest.raises(ValueError, match="vocabulary"):
        resolve_mask_token_id(args, model, SimpleNamespace(mask_token_id=9))


@pytest.mark.parametrize(
    "recipe,block_size",
    [
        ("run_qwen3_8b_dspark_tv_acceptance.sh", 7),
        ("run_qwen3_8b_dflash2_tv_acceptance_beta.sh", 8),
    ],
)
def test_example_forwards_post_training_options_without_starting_training(
    tmp_path, recipe, block_size
):
    root = Path(__file__).resolve().parents[2]
    # Replace the executable for this subprocess only; capture argv without GPUs.
    executable = tmp_path / "torchrun"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            f"examples/{recipe}",
            "8",
            "sdpa",
            "hf",
            "--init-draft-model-path",
            "/checkpoint with spaces",
            "--num-epochs",
            "1",
        ],
        cwd=root,
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    forwarded = result.stdout.splitlines()
    args = parse_args(forwarded[forwarded.index("scripts/train_dspark.py") + 1 :])
    assert args.init_draft_model_path == "/checkpoint with spaces"
    assert args.num_epochs == 1
    assert args.attention_backend == "sdpa"
    assert args.block_size == block_size
    if block_size == 8:
        assert args.dspark_loss_type == "tv-acceptance-beta"
        assert args.draft_config_path.endswith("qwen3-8b-dflash2.json")


@pytest.mark.parametrize("override", [False, True])
def test_on_policy_recipe_preserves_server_settings_without_starting_training(
    tmp_path, override
):
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "torchrun"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
    }
    for name in (
        "TARGET_MODEL_PATH",
        "INIT_DRAFT_MODEL_PATH",
        "TRAIN_DATA_PATH",
        "OUTPUT_DIR",
        "NUM_EPOCHS",
        "TV_ROLLOUT_MAX_TOKENS",
        "TV_TEMPERATURE",
        "SPECFORGE_DATA_NUM_PROC",
    ):
        environment.pop(name, None)
    if override:
        environment.update(
            TARGET_MODEL_PATH="/target with spaces; literal",
            INIT_DRAFT_MODEL_PATH="/checkpoint with spaces",
            TRAIN_DATA_PATH="/dataset with spaces.jsonl",
            OUTPUT_DIR="/output with spaces",
            TV_ROLLOUT_MAX_TOKENS="64",
        )
    result = subprocess.run(
        ["bash", "examples/run_qwen3_8b_dflash2_tv_beta_on_policy.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    forwarded = result.stdout.splitlines()
    assert forwarded[:4] == [
        "--standalone",
        "--nproc_per_node",
        "8",
        "scripts/train_dspark.py",
    ]
    args = parse_args(forwarded[4:])
    validate_objective_args(args)
    expected = {
        "dspark_loss_type": "tv-acceptance-beta",
        "tv_anchor_mode": "on-policy",
        "tv_rollout_max_tokens": 64 if override else 256,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.04,
        "max_grad_norm": 1.0,
        "max_length": 4096,
        "block_size": 8,
        "num_anchors": 512,
        "build_dataset_num_proc": 100,
        "sglang_mem_fraction_static": 0.3,
        "attention_backend": "flex_attention",
        "target_model_backend": "sglang",
        "report_to": "wandb",
        "wandb_offline": True,
        "save_interval": 1000,
        "log_interval": 10,
    }
    for name, value in expected.items():
        assert getattr(args, name) == value
    if override:
        assert args.target_model_path == "/target with spaces; literal"
        assert args.init_draft_model_path == "/checkpoint with spaces"
        assert args.train_data_path == "/dataset with spaces.jsonl"
        assert args.output_dir == "/output with spaces"
    else:
        assert args.init_draft_model_path.endswith("/epoch_5_step_322000")
        assert args.output_dir.endswith("-onpolicy")
        assert args.train_data_path.endswith("_20260929_regenerate_nothink.jsonl")


@pytest.mark.parametrize("rollout_budget", [None, "4096"])
def test_dspark_on_policy_recipe_forwards_checkpoint_and_rollout_budget(
    tmp_path, rollout_budget
):
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "torchrun"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "TARGET_MODEL_PATH": "/target with spaces; literal",
        "INIT_DRAFT_MODEL_PATH": "/original DSpark checkpoint",
        "TRAIN_DATA_PATH": "/training data.jsonl",
        "OUTPUT_DIR": "/new on-policy output",
    }
    for name in ("TV_ROLLOUT_MAX_TOKENS", "TV_TEMPERATURE", "NUM_EPOCHS"):
        environment.pop(name, None)
    if rollout_budget is not None:
        environment["TV_ROLLOUT_MAX_TOKENS"] = rollout_budget
    result = subprocess.run(
        [
            "bash",
            "examples/run_qwen3_8b_dspark_tv_beta_on_policy.sh",
            "8",
            "sdpa",
            "hf",
            "--batch-size",
            "2",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    forwarded = result.stdout.splitlines()
    assert forwarded[:4] == [
        "--standalone",
        "--nproc_per_node",
        "8",
        "scripts/train_dspark.py",
    ]
    args = parse_args(forwarded[4:])
    validate_objective_args(args)
    assert args.dspark_loss_type == "tv-acceptance-beta"
    assert args.tv_anchor_mode == "on-policy"
    assert args.tv_rollout_max_tokens == int(rollout_budget or "256")
    assert args.max_length == 4096 and args.num_anchors == 512
    assert args.block_size == 7 and args.batch_size == 2
    assert args.chat_template == "qwen3-instruct"
    assert args.attention_backend == "sdpa" and args.target_model_backend == "hf"
    assert args.draft_config_path is None
    assert args.target_model_path == environment["TARGET_MODEL_PATH"]
    assert args.init_draft_model_path == environment["INIT_DRAFT_MODEL_PATH"]
    assert args.train_data_path == environment["TRAIN_DATA_PATH"]
    assert args.output_dir == environment["OUTPUT_DIR"]
    assert not args.resume


def test_dspark_on_policy_recipe_requires_original_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "torchrun"
    executable.write_text('#!/bin/sh\nprintf "unexpected launch\\n"\n')
    executable.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "TARGET_MODEL_PATH": "/target",
        "TRAIN_DATA_PATH": "/training.jsonl",
    }
    environment.pop("INIT_DRAFT_MODEL_PATH", None)
    result = subprocess.run(
        ["bash", "examples/run_qwen3_8b_dspark_tv_beta_on_policy.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Set INIT_DRAFT_MODEL_PATH to the original DSpark checkpoint" in result.stderr
    assert "unexpected launch" not in result.stdout
