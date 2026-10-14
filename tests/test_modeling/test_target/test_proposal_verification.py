"""Real HF cache equivalence, including ragged prefixes and cache isolation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from specforge.inference.target_engine.dflash_target_model import HFDFlashTargetEngine
from specforge.inference.target_engine.proposal_verification import (
    HFProposalVerificationSession,
    ProposalVerificationSession,
    supports_qwen3_prefix_cache,
)


def make_target():
    torch.manual_seed(18)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 2,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    engine = HFDFlashTargetEngine(model)
    engine.set_capture_layers([0])
    return engine


@pytest.mark.parametrize("left_padding", [False, True])
def test_hf_cached_suffix_matches_full_prefix_and_never_projects_vocab(left_padding):
    engine = make_target()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 0, 0, 0]])
    mask = torch.tensor([[1] * 9, [1] * 6 + [0] * 3])
    if left_padding:
        ids[0, :2] = 0
        mask[0, :2] = 0
    rows, prefixes = [0, 1, 0, 1], [4, 5, 7, 3]
    suffixes = [
        torch.tensor([21, 22, 23]),
        torch.tensor([24, 25]),
        torch.tensor([26]),
        torch.tensor([27, 28, 29]),
    ]
    expected = []
    for row, prefix, suffix in zip(rows, prefixes, suffixes):
        full = torch.cat((ids[row, :prefix], suffix))[None]
        full_mask = torch.cat((mask[row, :prefix], torch.ones_like(suffix)))[None]
        output = engine.model(
            input_ids=full,
            attention_mask=full_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        expected.append(output.hidden_states[-1][0, prefix:])
    original = engine.model(
        input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False
    )
    with patch.object(
        engine.model.lm_head,
        "forward",
        side_effect=AssertionError("unneeded vocabulary projection"),
    ):
        with engine.proposal_verification() as session:
            assert isinstance(session, HFProposalVerificationSession)
            captured = session.capture(ids, mask, mask.float())
            torch.testing.assert_close(
                captured.last_hidden_states, original.hidden_states[-1]
            )
            torch.testing.assert_close(
                captured.hidden_states, original.hidden_states[1]
            )
            snapshot = [
                (kv[0].clone(), kv[1].clone()) for kv in captured.past_key_values
            ]
            # Exercise multiple candidate batches sharing the same original KV.
            for _ in range(2):
                result = session.score_suffixes(rows, prefixes, suffixes)
                for actual, reference in zip(result, expected):
                    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
            for kv, saved in zip(captured.past_key_values, snapshot):
                torch.testing.assert_close(kv[0], saved[0], rtol=0, atol=0)
                torch.testing.assert_close(kv[1], saved[1], rtol=0, atol=0)
        assert captured.past_key_values is None and session.closed
        # The ordinary capture path also avoids unused Qwen3 logits.
        engine.generate_dflash_data(ids, mask, mask.float())


def test_hf_cache_released_when_verification_fails():
    engine = make_target()
    ids = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(RuntimeError, match="test failure"):
        with engine.proposal_verification() as session:
            captured = session.capture(ids, torch.ones_like(ids), torch.ones_like(ids))
            with patch.object(
                engine.model.model, "forward", side_effect=RuntimeError("test failure")
            ):
                session.score_suffixes([0], [2], [torch.tensor([5])])
    assert captured.past_key_values is None and session.closed
    with pytest.raises(RuntimeError, match="active captured"):
        session.score_suffixes([0], [2], [torch.tensor([5])])


@pytest.mark.parametrize(
    "model_type,layers",
    [
        ("qwen3", ["sliding_attention"]),
        ("qwen3_moe", ["full_attention"]),
        ("qwen3_5", ["linear_attention", "full_attention"]),
    ],
)
def test_other_architectures_use_full_candidate_fallback(model_type, layers):
    config = SimpleNamespace(model_type=model_type, layer_types=layers)
    assert not supports_qwen3_prefix_cache(config)
    engine = HFDFlashTargetEngine(SimpleNamespace(config=config))
    assert type(engine.proposal_verification()) is ProposalVerificationSession


def test_sglang_full_candidate_fallback_also_rejects_noncontiguous_mask():
    engine = SimpleNamespace(backend="sglang")
    with ProposalVerificationSession(engine) as session:
        with pytest.raises(ValueError, match="right-padded"):
            session.capture(
                torch.ones(1, 3).long(), torch.tensor([[1, 0, 1]]), torch.ones(1, 3)
            )
