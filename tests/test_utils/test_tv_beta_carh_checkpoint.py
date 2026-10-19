"""Checkpoint-driven CARH/low-rank DynamicConv TV-beta post-training."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from specforge.core.tv_acceptance import configure_tv_acceptance
from specforge.inference.target_engine.proposal_verification import (
    ProposalVerificationSession,
)
from specforge.legacy.dspark_training.arguments import parse_args
from specforge.legacy.dspark_training.config import _apply_dspark_config
from specforge.legacy.dspark_training.initialization import (
    initialize_draft_weights,
    load_checkpoint_model,
    load_draft_config,
    prepare_warm_start,
)

from .test_tv_acceptance import RecordingTarget


CHECKPOINT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "dspark_carh_low_rank64_checkpoint_config.json"
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
            "--dspark-loss-type",
            "tv-acceptance-beta",
            *extra,
        ]
    )


def test_original_checkpoint_config_survives_loading_and_cli_defaults():
    """Inspect the full 4096-wide config without allocating its parameters."""
    original = json.loads(CHECKPOINT_CONFIG.read_text())
    config = load_draft_config(str(CHECKPOINT_CONFIG))
    _apply_dspark_config(config, training_args())
    for name, expected in original.items():
        if name in {"dtype", "transformers_version"}:
            # HF normalizes dtype and records the installed serializer version.
            continue
        assert getattr(config, name) == expected, name
    assert config.dtype == torch.bfloat16


def tiny_checkpoint_config():
    config = load_draft_config(str(CHECKPOINT_CONFIG))
    # Keep all five draft/capture layers, CARH rank 256 and the exact conv
    # topology; shrink only backbone/vocabulary sizes for an ordinary CPU test.
    config.hidden_size = 32
    config.intermediate_size = 64
    config.head_dim = 8
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.vocab_size = 32
    config.bos_token_id = 28
    config.eos_token_id = 29
    config.dflash_config["mask_token_id"] = 31
    config._attn_implementation = "sdpa"
    return config


class MultiCaptureTarget(RecordingTarget):
    """Causal verifier with five capture tensors and one final hidden tensor."""

    def __init__(self, hidden_size, capture_count, dtype):
        super().__init__(hidden_size)
        self.capture_count = capture_count
        self.dtype = dtype

    def generate_dflash_data(self, input_ids, attention_mask, loss_mask):
        output = super().generate_dflash_data(input_ids, attention_mask, loss_mask)
        output.last_hidden_states = output.last_hidden_states.to(self.dtype)
        output.hidden_states = output.last_hidden_states.repeat(
            1, 1, self.capture_count
        )
        return output


@pytest.mark.parametrize(
    "dtype,sharded", [(torch.float32, False), (torch.bfloat16, True)]
)
def test_carh_conv_checkpoint_warm_start_and_beta_optimizer_step(
    tmp_path, dtype, sharded
):
    from specforge.core.dspark import OnlineDSparkModel
    from specforge.legacy.dspark import DSparkDraftModel

    torch.manual_seed(41)
    source = DSparkDraftModel(tiny_checkpoint_config()).to(dtype=dtype)
    # Nonzero learned projections detect accidental identity reinitialization
    # and let gradients reach CARH conditioning and both conv factors.
    with torch.no_grad():
        source.markov_head.markov_w2.weight.normal_(std=0.025)
        for layer in source.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                conv.kernel_projection.weight.normal_(std=0.025)
                conv.base_kernel.add_(torch.randn_like(conv.base_kernel) * 0.01)
    checkpoint = tmp_path / "source"
    source.save_pretrained(
        checkpoint, max_shard_size="50KB" if sharded else "1GB"
    )
    config_path = checkpoint / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config["architectures"] = ["Qwen3DSparkModel"]
    config_path.write_text(json.dumps(saved_config))
    # Weight-only initialization must never deserialize optimizer state or
    # depend on the external dspark.py referenced by auto_map.
    (checkpoint / "training_state.pt").write_bytes(b"not an optimizer checkpoint")
    assert not (checkpoint / "dspark.py").exists()
    if sharded:
        assert (checkpoint / "model.safetensors.index.json").is_file()

    args = training_args(
        "--init-draft-model-path",
        str(checkpoint),
        "--output-dir",
        str(tmp_path / "new-run"),
    )
    prepare_warm_start(args)
    config = load_draft_config(args.draft_config_path)
    _apply_dspark_config(config, args)
    config._attn_implementation = "sdpa"
    draft = DSparkDraftModel(config).to(dtype=dtype)
    loaded = load_checkpoint_model(str(checkpoint), dtype=dtype)
    initialize_draft_weights(draft, args)
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(draft.state_dict()[name], expected, rtol=0, atol=0)
    assert draft.target_layer_ids == [1, 9, 17, 25, 33]
    assert draft.markov_head.markov_rank == 256
    assert draft.markov_head.predecessor_count == 1
    assert draft.mask_token_id == 31
    assert draft.layers[0].attention_conv.kernel_projection.down.out_features == 64
    assert draft.layers[0].attention_conv.source_down is None

    model = OnlineDSparkModel(
        draft,
        nn.Linear(32, 32, bias=False).to(dtype=dtype),
        nn.Embedding(32, 32).to(dtype=dtype),
        mask_token_id=31,
        block_size=7,
        attention_backend="sdpa",
        num_anchors=2,
    )
    model.embed_tokens.requires_grad_(False)
    configure_tv_acceptance(
        model,
        temperature=0.8,
        chunk_blocks=1,
        eos_token_ids=[],
        beta=True,
        verification_batch_size=2,
    )
    confidence_before = {
        name: p.detach().clone() for name, p in draft.confidence_head.named_parameters()
    }
    assert all(not p.requires_grad for p in draft.confidence_head.parameters())
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.01
    )
    assert not optimizer.state
    target = MultiCaptureTarget(32, 5, dtype)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    attention = torch.ones_like(ids)
    loss_mask = torch.ones_like(ids)
    with ProposalVerificationSession(target) as session:
        capture = session.capture(ids, attention, loss_mask)
        loss, *_, components = model(
            input_ids=ids,
            hidden_states=capture.hidden_states,
            last_hidden_states=capture.last_hidden_states,
            attention_mask=attention,
            loss_mask=loss_mask,
            tv_verification_session=session,
        )
    requests_after_forward = len(target.requests)
    with patch("torch.multinomial", side_effect=AssertionError("backward resampled")):
        loss.backward()
    assert session.closed
    assert len(target.requests) == requests_after_forward
    assert torch.isfinite(loss)
    assert "tv_acceptance_length_estimate" in components
    trainable = dict(draft.named_parameters())
    checked_names = [
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "markov_head.hidden_proj.weight",
        "markov_head.gate_proj.weight",
        "markov_head.depth_embedding.weight",
        "layers.0.attention_conv.base_kernel",
        "layers.0.attention_conv.kernel_projection.down.weight",
        "layers.0.attention_conv.kernel_projection.weight",
        "layers.0.mlp_conv.kernel_projection.weight",
        "layers.4.mlp_conv.kernel_projection.weight",
    ]
    before_step = {}
    for name in checked_names:
        parameter = trainable[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.count_nonzero() > 0, name
        before_step[name] = parameter.detach().clone()
    optimizer.step()
    for name, before in before_step.items():
        assert not torch.equal(trainable[name], before), name
    for name, parameter in draft.confidence_head.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, confidence_before[name], rtol=0, atol=0)
    assert model.lm_head.weight.grad is None
    assert model.embed_tokens.weight.grad is None
