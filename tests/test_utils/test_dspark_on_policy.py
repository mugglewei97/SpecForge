"""Real DSpark rollout, cached context and same-step beta replay invariants."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from specforge.core.dspark import OnlineDSparkModel
from specforge.core.dspark_on_policy import (
    DSparkRolloutCache,
    collect_dspark_rollout,
    forward_dspark_blocks,
    replay_dspark_blocks,
)
from specforge.core.tv_acceptance import configure_tv_acceptance
from specforge.inference.target_engine.dflash_target_model import HFDFlashTargetEngine
from specforge.legacy.dspark import DSparkConfig, DSparkDraftModel


def make_on_policy_dspark(*, head="vanilla", conv=False, budget=12, rounds=12, eos=()):
    """Use independent real target/draft networks, with nontrivial head/conv weights."""
    target_config = Qwen3Config(
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
    target_config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(target_config).eval().requires_grad_(False)
    engine = HFDFlashTargetEngine(target)
    engine.set_capture_layers([1])
    method = {"target_layer_ids": [1]}
    if conv:
        method.update(
            conv_kernel_size=2,
            conv_group_size=4,
            conv_kernel_conditioning="low-rank64",
            conv_source_rank=4,
        )
    config = DSparkConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        head_dim=4,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        block_size=7,
        num_target_layers=3,
        dflash_config=method,
        markov_rank=0 if head is None else 4,
        markov_head_type=head or "vanilla",
        enable_confidence_head=False,
    )
    config._attn_implementation = "sdpa"
    draft = DSparkDraftModel(config)
    with torch.no_grad():
        if draft.markov_head is not None:
            draft.markov_head.markov_w2.weight.normal_(std=0.25)
        if head == "carh":
            draft.markov_head.hidden_proj.weight.normal_(std=0.2)
            draft.markov_head.depth_embedding.weight.normal_(std=0.3)
            draft.markov_head.gate_proj.weight.normal_(std=0.25)
            draft.markov_head.gate_proj.bias.fill_(0.2)
        if conv:
            for layer in draft.layers:
                for convolution in (layer.attention_conv, layer.mlp_conv):
                    convolution.kernel_projection.weight.normal_(std=0.03)
                    convolution.kernel_projection.down.weight.normal_(std=0.1)
    model = OnlineDSparkModel(
        draft,
        target.lm_head,
        target.model.embed_tokens,
        mask_token_id=15,
        block_size=7,
        attention_backend="sdpa",
        num_anchors=rounds,
    )
    configure_tv_acceptance(
        model,
        temperature=0.8,
        chunk_blocks=2,
        eos_token_ids=eos,
        beta=True,
        anchor_mode="on-policy",
        rollout_max_tokens=budget,
        max_sequence_length=64,
    )
    return model, engine


def _batch():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    loss = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0]])
    return ids, attention, loss


def _policy_logits(model, hidden, previous):
    # Evaluate sequential head steps independently of the replay's block API.
    flat = hidden.flatten(0, 1)
    previous = previous.flatten(0, 1)
    base = model.lm_head(flat)
    head = model.draft_model.markov_head
    if head is not None:
        base = torch.stack(
            [
                base[:, depth]
                + head.compute_step_bias(previous[:, depth], flat[:, depth], depth)
                for depth in range(7)
            ],
            dim=1,
        )
    return base.reshape(*hidden.shape[:-1], -1)


@pytest.mark.parametrize(
    "head,conv", [(None, False), ("vanilla", False), ("carh", True)]
)
@pytest.mark.parametrize("backend", ["sdpa", "flex-mask"])
def test_incremental_cache_keeps_all_seven_dspark_slots_and_ragged_context(
    head, conv, backend, monkeypatch
):
    torch.manual_seed(137)
    model, engine = make_on_policy_dspark(head=head, conv=conv)
    if backend == "flex-mask":
        from specforge.legacy.dflash import ALL_ATTENTION_FUNCTIONS

        def cpu_flex(module, q, k, v, attention_mask, scaling, **_kwargs):
            # Evaluate the actual production BlockMask at token resolution;
            # CPU SDPA provides the independent attention kernel for this test.
            device = q.device
            b = torch.arange(q.size(0), device=device)[:, None, None, None]
            h = torch.arange(q.size(1), device=device)[None, :, None, None]
            qi = torch.arange(q.size(2), device=device)[None, None, :, None]
            ki = torch.arange(k.size(2), device=device)[None, None, None, :]
            mask = attention_mask.mask_mod(b, h, qi, ki)
            result = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=True, scale=scaling
            )
            return result.transpose(1, 2), None

        monkeypatch.setitem(
            ALL_ATTENTION_FUNCTIONS._local_mapping, "flex_attention", cpu_flex
        )
        model.attention_backend = "flex_attention"
        model.draft_model.config._attn_implementation = "flex_attention"
    cache = DSparkRolloutCache()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    with torch.no_grad():
        for lengths, flags in [
            ([4, 2], [True, True]),
            ([6, 3], [True, False]),
            ([8, 5], [True, True]),
        ]:
            attention = torch.arange(8)[None] < torch.tensor(lengths)[:, None]
            captured = engine.generate_dflash_data(
                ids, attention.long(), attention.float()
            )
            output = SimpleNamespace(
                input_ids=ids,
                attention_mask=attention,
                hidden_states=captured.hidden_states,
            )
            active = torch.tensor(flags)
            with patch(
                "torch.nn.attention.flex_attention.create_block_mask",
                side_effect=AssertionError(
                    "Rollout must not derive a dense block mask"
                ),
            ):
                anchors, incremental = cache.forward(
                    model,
                    output,
                    active,
                    lengths=torch.tensor(lengths),
                    lengths_cpu=tuple(lengths),
                )
            packed = forward_dspark_blocks(
                model,
                ids,
                captured.hidden_states,
                attention,
                anchors[:, None],
                active[:, None],
            )
            assert incremental.shape == (2, 7, 16)
            torch.testing.assert_close(
                incremental[active], packed[:, 0][active], atol=2e-6, rtol=2e-5
            )
            assert cache.kv.get_seq_length() == cache.valid_keys.size(1)
            assert cache.valid_keys.sum(-1).tolist() == [
                length - 1 for length in lengths
            ]


@pytest.mark.parametrize(
    "head,conv", [(None, False), ("vanilla", False), ("carh", True)]
)
def test_rollout_replays_actual_policy_and_original_rejected_teacher_suffix(head, conv):
    torch.manual_seed(150)
    model, engine = make_on_policy_dspark(head=head, conv=conv)
    ids, attention, loss = _batch()
    with (
        engine.on_policy_rollout() as session,
        patch.object(
            engine.model.model, "forward", wraps=engine.model.model.forward
        ) as target,
        patch.object(session, "commit", wraps=session.commit) as commits,
    ):
        replay = collect_dspark_rollout(model, ids, attention, loss, session)
        assert target.call_count == 1 + replay.anchors.size(1)
        assert replay.anchors[:, 0].tolist() == [3, 2]
        assert torch.equal(replay.valid.any(-1), replay.keep)
        assert replay.teacher_hidden.shape == (*replay.tokens.shape, 16)
        assert not replay.teacher_hidden.requires_grad
        assert not replay.behavior_log_probs.requires_grad
        assert torch.equal(replay.previous[:, :, 1:], replay.tokens[:, :, :-1])
        roots = replay.input_ids.gather(1, replay.anchors)
        torch.testing.assert_close(
            replay.previous[:, :, 0][replay.keep], roots[replay.keep]
        )
        accepted = torch.stack([call.args[0] for call in commits.call_args_list], dim=1)
        assert ((accepted < replay.valid.sum(-1)) & replay.keep).any(), (
            "fixture must contain a real rejection, with its suffix retained"
        )

        hidden = forward_dspark_blocks(
            model,
            replay.input_ids,
            replay.target_hidden,
            replay.attention_mask,
            replay.anchors,
            replay.keep,
        )
        logits = _policy_logits(model, hidden, replay.previous)
        log_q = (logits.float() / model.tv_temperature).log_softmax(-1)
        chosen = log_q.gather(-1, replay.tokens[..., None]).squeeze(-1)
        torch.testing.assert_close(
            chosen[replay.valid],
            replay.behavior_log_probs[replay.valid],
            atol=2e-6,
            rtol=2e-5,
        )

        # The final committed sequence is used only through the recorded anchor.
        # Its correction tokens must not replace the original rejected proposals.
        for row, block in replay.keep.nonzero().tolist():
            anchor = int(replay.anchors[row, block])
            count = int(replay.valid[row, block].sum())
            branch = torch.cat(
                (
                    replay.input_ids[row, : anchor + 1],
                    replay.tokens[row, block, : count - 1],
                )
            )[None]
            full = engine.generate_dflash_data(
                branch, torch.ones_like(branch), torch.ones_like(branch)
            )
            torch.testing.assert_close(
                replay.teacher_hidden[row, block, :count],
                full.last_hidden_states[0, anchor : anchor + count],
                atol=2e-6,
                rtol=2e-5,
            )
        session.close()

    with (
        patch.object(
            engine.model.model,
            "forward",
            side_effect=AssertionError("target in replay"),
        ),
        patch("torch.multinomial", side_effect=AssertionError("replay resampled")),
    ):
        result = replay_dspark_blocks(model, replay)
        result[0].backward()
    gradients = [p.grad for p in model.draft_model.parameters() if p.requires_grad]
    assert gradients and all(
        g is not None and torch.isfinite(g).all() for g in gradients
    )
    assert model.draft_model.fc.weight.grad.count_nonzero()
    if head is not None:
        for parameter in model.draft_model.markov_head.parameters():
            assert parameter.grad.count_nonzero()
    if conv:
        assert model.draft_model.layers[
            0
        ].attention_conv.kernel_projection.weight.grad.count_nonzero()
    assert all(parameter.grad is None for parameter in engine.model.parameters())
    assert result[-1]["tv_rollout_rounds"] == replay.keep.sum()
    assert result[-1]["tv_rollout_accepted_length"] <= 7
    assert result[-1]["tv_valid_blocks"] == replay.keep.sum()


def test_reference_answers_do_not_change_dspark_rollout():
    torch.manual_seed(37)
    model, engine = make_on_policy_dspark(budget=5)
    ids, attention, loss = _batch()
    changed = torch.where(loss.bool(), (ids + 5) % 16, ids)
    records = []
    for source in (ids, changed):
        torch.manual_seed(312)
        with engine.on_policy_rollout() as session:
            records.append(
                collect_dspark_rollout(model, source, attention, loss, session)
            )
    for name in (
        "input_ids",
        "anchors",
        "tokens",
        "previous",
        "valid",
        "teacher_hidden",
        "behavior_log_probs",
        "statistics",
    ):
        torch.testing.assert_close(
            getattr(records[0], name), getattr(records[1], name), rtol=0, atol=0
        )


@pytest.mark.parametrize("empty_kind", ["no-supervision", "seed-eos", "seed-budget"])
def test_empty_dspark_rollout_has_finite_zero_backward_and_closes_target(empty_kind):
    model, engine = make_on_policy_dspark(
        budget=1 if empty_kind == "seed-budget" else 5,
        eos=tuple(range(16)) if empty_kind == "seed-eos" else (),
    )
    ids, attention, loss = _batch()
    if empty_kind == "no-supervision":
        loss.zero_()
    with (
        engine.on_policy_rollout() as session,
        patch.object(
            session, "verify", side_effect=AssertionError("empty batch verified")
        ),
    ):
        result = model(
            ids, None, loss, attention_mask=attention, tv_on_policy_session=session
        )
        assert session.closed
    assert result[0] == 0 and result[-1]["tv_valid_blocks"] == 0
    result[0].backward()
    gradients = [p.grad for p in model.draft_model.parameters() if p.grad is not None]
    assert gradients and all(
        torch.isfinite(g).all() and not g.count_nonzero() for g in gradients
    )


def test_finished_dspark_rows_stay_inactive_and_round_limit_is_reported():
    model, engine = make_on_policy_dspark(budget=20, rounds=1)
    ids, attention, loss = _batch()
    loss[1].zero_()
    with engine.on_policy_rollout() as session:
        replay = collect_dspark_rollout(model, ids, attention, loss, session)
    assert not replay.keep[1].any() and not replay.valid[1].any()
    assert replay.attention_mask[1].sum() == 1
    assert replay.statistics[4] == 1
    assert replay.statistics[5] == 1
    assert replay.anchors.shape[1] == 1


@pytest.mark.parametrize("source", ["attention", "fusion", "mixer", "sliding"])
def test_dspark_on_policy_rejects_stochastic_or_windowed_context(source):
    model, _ = make_on_policy_dspark()
    if source == "attention":
        model.draft_model.config.attention_dropout = 0.1
    elif source == "fusion":
        model.draft_model.target_layer_fusion_source_dropout = 0.1
    elif source == "mixer":
        model.draft_model.config.dflash_config["cross_depth_mixer_dropout"] = 0.1
    else:
        model.draft_model.layers[0].self_attn.sliding_window = 8
    with pytest.raises(ValueError, match="zero attention/fusion/mixer dropout"):
        configure_tv_acceptance(
            model,
            temperature=0.8,
            chunk_blocks=2,
            eos_token_ids=[],
            beta=True,
            anchor_mode="on-policy",
        )


def test_dspark_on_policy_rejects_eager_attention_with_boolean_masks():
    model, _ = make_on_policy_dspark()
    model.attention_backend = "eager"
    with pytest.raises(ValueError, match="sdpa or flex_attention"):
        configure_tv_acceptance(
            model,
            temperature=0.8,
            chunk_blocks=2,
            eos_token_ids=[],
            beta=True,
            anchor_mode="on-policy",
        )
