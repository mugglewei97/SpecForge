"""Native DFlash2 compatibility with the original colocated training recipe."""

import copy
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from specforge.legacy.dspark_training.arguments import parse_args
from specforge.legacy.dspark_training.dflash2 import (
    build_dflash2_training_model,
    dflash2_native_resume_contract,
    forward_dflash2_native,
    prepare_dflash2_config,
    validate_dflash2_resume,
)
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from tests.test_scripts.test_train_dspark import training_args
from tests.test_utils.test_dflash2_tv_acceptance_beta import tiny_config


def test_server_recipe_preserves_training_settings(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    recipe = (root / "examples/run_qwen3_8b_dflash2.sh").read_text()
    # Parse only: the server cleanup commands must never execute in tests.
    command = recipe[recipe.index("torchrun --standalone") :].replace("\\\n", " ")
    command = command.replace("$ATTENTION_BACKEND", "flex_attention")
    command = command.replace("$TARGET_BACKEND", "sglang")
    monkeypatch.setenv("SPECFORGE_DATA_NUM_PROC", "32")
    argv = shlex.split(command)
    assert argv[:5] == [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "8",
        "scripts/train_dspark.py",
    ]
    args = parse_args(argv[5:])
    assert args.draft_config_path == "./configs/qwen3-8b-dflash2.json"
    assert args.dspark_loss_type == "dflash2"
    assert args.output_dir.endswith("/qwen3-8b-dflash2-mix/")
    expected = {
        "target_model_path": "/mnt/amed-s1/common/ckpt/gaochang/Qwen3-8B",
        "train_data_path": "/mnt/amed-s1/common/data/gaochang/eagle-data/qwen3-8b-regen-mix-70w.jsonl",
        "num_epochs": 6,
        "batch_size": 4,
        "learning_rate": 6e-4,
        "warmup_ratio": 0.04,
        "max_grad_norm": 1.0,
        "max_length": 4096,
        "chat_template": "qwen",
        "attention_backend": "flex_attention",
        "target_model_backend": "sglang",
        "loss_decay_gamma": 4.0,
        "log_interval": 50,
        "save_interval": 2000,
        "block_size": 7,
        "num_anchors": 512,
        "sglang_mem_fraction_static": 0.3,
        "build_dataset_num_proc": 32,
    }
    for name, value in expected.items():
        assert getattr(args, name) == value
    assert "ATTENTION_BACKEND=${2:-flex_attention}" in recipe
    assert "TARGET_BACKEND=${3:-sglang}" in recipe


def make_native():
    args = training_args(
        "--dspark-loss-type", "dflash2",
        "--block-size", "7",
        "--attention-backend", "sdpa",
        "--num-anchors", "2",
        "--loss-decay-gamma", "4.0",
    )
    config = tiny_config()
    config.block_size = 8
    prepare_dflash2_config(config, args)
    assert config.block_size == config.dflash_config["block_size"] == 7
    assert args.block_size == 7
    model = build_dflash2_training_model(
        args,
        DFlash2DraftModel(config),
        SimpleNamespace(
            lm_head=nn.Linear(16, 16, bias=False),
            embed_tokens=nn.Embedding(16, 16),
        ),
        mask_token_id=15,
    )
    return model, args


def test_native_adapter_matches_objective_and_backpropagates():
    torch.manual_seed(17)
    model, _ = make_native()
    assert model.loss_decay_gamma == 4.0
    assert model.selector_loss_alpha == 1.0
    inputs = torch.randint(0, 15, (2, 20))
    hidden = torch.randn(2, 20, 16)
    mask = torch.ones_like(inputs)
    attention = torch.ones_like(mask)
    attention[:, -3:] = 0
    torch.manual_seed(19)
    expected = model(inputs, hidden, mask * attention)
    torch.manual_seed(19)
    actual = forward_dflash2_native(model, inputs, hidden, mask, attention, None)
    assert len(actual) == 6
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert all(
        value.ndim == 0 and not value.requires_grad for value in actual[-1].values()
    )
    actual[0].backward()
    conv_grad = model.draft_model.layers[0].attention_conv.kernel_projection.weight.grad
    assert torch.isfinite(conv_grad).all() and conv_grad.count_nonzero() > 0
    selector_grads = [p.grad for p in model.draft_model.candidate_selector.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in selector_grads)
    assert any(g.count_nonzero() > 0 for g in selector_grads)
    assert model.lm_head.weight.grad is None
    assert model.embed_tokens.weight.grad is None


def test_native_resume_rejects_changed_decay_or_block_size():
    model, args = make_native()
    state = {
        "dflash2_native": copy.deepcopy(
            dflash2_native_resume_contract(model.draft_model, args)
        )
    }
    validate_dflash2_resume(state, model.draft_model, args)
    args.loss_decay_gamma = 7.0
    with pytest.raises(ValueError, match="objective settings"):
        validate_dflash2_resume(state, model.draft_model, args)
    args.loss_decay_gamma = 4.0
    model.draft_model.block_size = 8
    with pytest.raises(ValueError, match="objective settings"):
        validate_dflash2_resume(state, model.draft_model, args)
