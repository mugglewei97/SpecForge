"""Real-model rollout/cache/replay equivalence and no target work in backward."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from specforge.core.dflash2_on_policy import (
    DFlashRolloutCache,
    collect_dflash2_rollout,
    replay_dflash2_blocks,
)
from specforge.core.dflash2_tv_acceptance import dflash2_proposal_logits
from specforge.inference.target_engine.dflash_target_model import HFDFlashTargetEngine
from tests.test_utils.test_dflash2_tv_acceptance_beta import make_dflash2


def make_on_policy_models(*, budget=12, rounds=8, eos=()):
    model, _ = make_dflash2()
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 3,
    )
    config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    engine = HFDFlashTargetEngine(target)
    engine.set_capture_layers([1])
    model.lm_head = target.lm_head
    model.embed_tokens = target.model.embed_tokens
    model.num_anchors = rounds
    model.configure_beta(
        temperature=0.8,
        chunk_blocks=2,
        eos_token_ids=eos,
        verification_batch_size=2,
        anchor_mode="on-policy",
        rollout_max_tokens=budget,
        max_sequence_length=64,
    )
    return model, engine


def _batch():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    loss = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0]])
    return ids, mask, loss


@pytest.mark.parametrize("backend", ["sdpa", "flex-mask"])
def test_draft_incremental_cache_matches_dense_blocks_with_ragged_advances(
    backend, monkeypatch
):
    torch.manual_seed(271)
    model, engine = make_on_policy_models()
    if backend == "flex-mask":
        # Exercise the production BlockMask without requiring a CUDA kernel.
        # Both incremental and packed masks are evaluated at token resolution.
        def cpu_flex(q, k, v, *, block_mask, enable_gqa, scale, **kwargs):
            device = q.device
            b = torch.arange(q.size(0), device=device)[:, None, None, None]
            h = torch.arange(q.size(1), device=device)[None, :, None, None]
            qi = torch.arange(q.size(2), device=device)[None, None, :, None]
            ki = torch.arange(k.size(2), device=device)[None, None, None, :]
            mask = block_mask.mask_mod(b, h, qi, ki)
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=enable_gqa, scale=scale
            )

        monkeypatch.setattr(
            "specforge.modeling.draft.dflash.compile_friendly_flex_attention",
            cpu_flex,
        )
        model.attention_backend = "flex_attention"
        model.draft_model.config._attn_implementation = "flex_attention"
    cache = DFlashRolloutCache()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    with torch.no_grad():
        for lengths, active in [
            ([4, 2], [True, True]),
            ([6, 3], [True, False]),
            ([8, 5], [True, True]),
        ]:
            mask = torch.arange(8)[None] < torch.tensor(lengths)[:, None]
            captured = engine.generate_dflash_data(ids, mask.long(), mask.float())
            output = SimpleNamespace(
                input_ids=ids, attention_mask=mask, hidden_states=captured.hidden_states
            )
            active = torch.tensor(active)
            anchors, actual = cache.forward(model, output, active)
            _, _, dense = model._forward_draft_blocks(
                ids,
                captured.hidden_states,
                mask,
                anchor_positions=anchors[:, None],
                block_keep_mask=active[:, None],
            )
            torch.testing.assert_close(
                actual[active], dense[:, 1:][active], atol=2e-6, rtol=2e-5
            )
            assert cache.kv.get_seq_length() == cache.valid_keys.size(1)
            assert cache.valid_keys.sum(-1).tolist() == [x - 1 for x in lengths]


def test_real_rollout_records_actual_anchors_and_replays_without_teacher_calls():
    torch.manual_seed(149)
    model, engine = make_on_policy_models()
    ids, mask, loss = _batch()
    with (
        engine.on_policy_rollout() as session,
        patch.object(
            engine.model.model, "forward", wraps=engine.model.model.forward
        ) as target_forward,
    ):
        replay = collect_dflash2_rollout(model, ids, mask, loss, session)
        calls = target_forward.call_count
        assert calls == 1 + replay.anchors.size(1)
        assert replay.anchors[:, 0].tolist() == [3, 2]
        assert (replay.anchors[:, 1:] > replay.anchors[:, :-1])[
            replay.keep[:, 1:]
        ].all()
        assert torch.equal(replay.valid.any(-1), replay.keep)
        assert replay.statistics[0] < replay.statistics[1], (
            "fixture must exercise rejections"
        )
        assert replay.statistics[3] <= 24
        # Replaying all anchors together must yield the same behavior law as
        # incremental drafting, with no future-context or cross-block leakage.
        _, _, hidden = model._forward_draft_blocks(
            replay.input_ids,
            replay.target_hidden,
            replay.attention_mask,
            anchor_positions=replay.anchors,
            block_keep_mask=replay.keep,
        )
        hidden = hidden.reshape(2, -1, 8, 16)[:, :, 1:]
        logits = dflash2_proposal_logits(
            model, hidden, replay.previous, replay.candidate_ids
        )
        log_q = (logits.float() / model.tv_temperature).log_softmax(-1)
        recorded = log_q.gather(-1, replay.tokens[..., None]).squeeze(-1)
        torch.testing.assert_close(
            recorded[replay.valid],
            replay.behavior_log_probs[replay.valid],
            atol=2e-6,
            rtol=2e-5,
        )
        session.close()
        with patch.object(
            engine.model.model,
            "forward",
            side_effect=AssertionError("teacher replayed"),
        ):
            result = replay_dflash2_blocks(model, replay)
            with patch(
                "torch.multinomial", side_effect=AssertionError("backward resampled")
            ):
                result[0].backward()
        assert target_forward.call_count == calls
    conv = model.draft_model.layers[0].attention_conv.kernel_projection.weight
    assert (
        conv.grad is not None
        and torch.isfinite(conv.grad).all()
        and conv.grad.count_nonzero()
    )
    assert model.draft_model.candidate_selector.successor_codebook.grad.count_nonzero()
    assert all(p.grad is None for p in engine.model.parameters())
    assert result[-1]["tv_rollout_rounds"] == replay.keep.sum()
    assert result[-1]["tv_rollout_accepted_length"] <= 7
    before = conv.detach().clone()
    torch.optim.AdamW(model.draft_model.parameters(), lr=1e-3).step()
    assert not torch.equal(before, conv)


def test_rollout_does_not_use_reference_answers():
    torch.manual_seed(37)
    model, engine = make_on_policy_models(budget=5, rounds=3)
    ids, mask, loss = _batch()
    changed = torch.where(loss.bool(), (ids + 5) % 16, ids)
    results = []
    for source in (ids, changed):
        torch.manual_seed(312)
        with engine.on_policy_rollout() as session:
            results.append(collect_dflash2_rollout(model, source, mask, loss, session))
    for name in (
        "input_ids",
        "anchors",
        "tokens",
        "candidate_ids",
        "target_log_probs",
        "statistics",
    ):
        torch.testing.assert_close(
            getattr(results[0], name), getattr(results[1], name), rtol=0, atol=0
        )


@pytest.mark.parametrize("empty_kind", ["no-supervision", "seed-eos", "seed-budget"])
def test_empty_rollouts_have_finite_zero_graph_and_no_verification(empty_kind):
    model, engine = make_on_policy_models(
        budget=1 if empty_kind == "seed-budget" else 5,
        eos=tuple(range(16)) if empty_kind == "seed-eos" else (),
    )
    ids, mask, loss = _batch()
    if empty_kind == "no-supervision":
        loss.zero_()
    with (
        engine.on_policy_rollout() as session,
        patch.object(
            session, "verify", side_effect=AssertionError("empty batch verified")
        ),
    ):
        result = model(
            ids, None, loss, attention_mask=mask, tv_on_policy_session=session
        )
        assert session.closed
    assert result[0] == 0 and result[-1]["tv_valid_blocks"] == 0
    result[0].backward()
    grads = [p.grad for p in model.draft_model.parameters() if p.grad is not None]
    assert grads and all(
        torch.isfinite(g).all() and not g.count_nonzero() for g in grads
    )


def test_one_inactive_row_is_never_resurrected_by_bonus_sampling():
    model, engine = make_on_policy_models(budget=9, rounds=5)
    ids, mask, loss = _batch()
    loss[1].zero_()
    with engine.on_policy_rollout() as session:
        replay = collect_dflash2_rollout(model, ids, mask, loss, session)
    assert not replay.keep[1].any() and not replay.valid[1].any()
    assert replay.attention_mask[1].sum() == 1
    assert replay.statistics[4] == 1
    assert replay.statistics[2] == replay.keep[0].sum()
