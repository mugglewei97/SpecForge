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
    assert "--sglang-mem-fraction-static" in result.stdout


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
    assert args.tv_verification_batch_size == 4


@pytest.mark.parametrize(
    "flag,value,match",
    [
        ("--tv-sampling-temperature", "nan", "temperature"),
        ("--tv-sampling-temperature", "0", "temperature"),
        ("--tv-objective-chunk-blocks", "0", "chunk"),
        ("--tv-verification-batch-size", "-1", "chunk"),
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
        ("tv_objective_chunk_blocks", 2),
        ("tv_verification_batch_size", 2),
    ):
        with pytest.raises(ValueError, match=key):
            validate_objective_resume(wrap({**saved, key: value}), args)


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
