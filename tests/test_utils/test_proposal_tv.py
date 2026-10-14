"""Proposal prefixes, rejected suffix credit and loss/gradient parity on CPU."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from specforge.core.tv_acceptance import (
    configure_tv_acceptance,
    proposal_tv_forward,
    teacher_forced_tv_forward,
)
from specforge.data.dspark_replay import ReplayCollator
from tests.test_utils.test_tv_acceptance import make_model


def proposal_trace(sample_id="a"):
    # The first correction is EOS in the final sequence. Its rejected proposal
    # suffix still has three valid tokens and remains supervised.
    return dict(
        sample_id=sample_id,
        input_ids=[1, 2, 3, 9],
        prompt_length=2,
        anchor_positions=[2],
        draft_version=0,
        target_version="t",
        trace_version=2,
        tv_prefix_mode="proposal",
        selected_round_ids=[0],
        sampling_config=dict(
            temperature=0.8, top_p=1.0, top_k=-1, max_new_tokens=8, speculative={}
        ),
        round_metadata=[
            dict(
                round_id=0,
                draft_length=3,
                verified_draft_tokens=3,
                accepted_draft_tokens=0,
                committed_tokens=1,
                remaining_budget=7,
                proposal_ids=[4, 5, 6],
                proposal_valid_mask=[True, True, True],
            )
        ],
    )


def collate(records, **kwargs):
    return ReplayCollator(
        draft_version=0,
        target_version="t",
        num_anchors=2,
        prefix_mode="proposal",
        block_size=3,
        eos_token_ids=[9],
        **kwargs,
    )(records)


def test_collator_keeps_rejected_suffix_after_final_eos_and_never_resamples():
    with patch("torch.rand", side_effect=AssertionError("must not resample")):
        data = collate([proposal_trace()])
    assert data["proposal_ids"].tolist() == [[[4, 5, 6]]]
    assert data["proposal_valid_mask"].all()
    assert data["proposal_round_ids"].tolist() == [[0]]
    assert data["anchor_positions"].tolist() == [[2]]
    assert data["block_keep_mask"].tolist() == [[True]]


@pytest.mark.parametrize("boundary", ["eos", "budget", "coverage"])
def test_proposal_mask_uses_own_eos_budget_and_verification(boundary):
    record = proposal_trace()
    meta = record["round_metadata"][0]
    if boundary == "eos":
        meta["proposal_ids"] = [4, 9, 6]
    elif boundary == "budget":
        record["sampling_config"]["max_new_tokens"] = 3
        meta["remaining_budget"] = 2
    else:
        meta["verified_draft_tokens"] = 2
    meta["proposal_valid_mask"] = [True, True, False]
    assert collate([record])["proposal_valid_mask"].tolist() == [[[True, True, False]]]
    meta["proposal_valid_mask"] = [True, True, True]
    with pytest.raises(ValueError, match="proposal"):
        collate([record])


@pytest.mark.parametrize(
    "field", ["trace_version", "selected_round_ids", "tv_prefix_mode"]
)
def test_old_or_incomplete_trace_fails_closed(field):
    record = proposal_trace()
    record.pop(field)
    with pytest.raises(ValueError, match="proposal"):
        collate([record])


def test_selection_round_identity_and_empty_samples():
    record = proposal_trace()
    record["input_ids"] = [1, 2, 3, 4, 5, 6, 7]
    record["anchor_positions"] = [2, 4, 5]
    base = record["round_metadata"][0]
    record["round_metadata"] = [
        dict(
            base, round_id=i, remaining_budget=9 - s, proposal_ids=[i + 1, i + 2, i + 3]
        )
        for i, s in enumerate(record["anchor_positions"])
    ]
    record["selected_round_ids"] = [0, 2]
    empty = proposal_trace("empty")
    empty.update(anchor_positions=[], round_metadata=[], selected_round_ids=[])
    data = collate([record, empty])
    assert data["anchor_positions"].tolist() == [[2, 5], [0, 0]]
    assert data["proposal_ids"][0].tolist() == [[1, 2, 3], [3, 4, 5]]
    assert data["proposal_round_ids"].tolist() == [[0, 2], [-1, -1]]
    assert not data["proposal_valid_mask"][1].any()


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_same_proposal_as_final_sequence_matches_loss_and_all_gradients(chunk_size):
    torch.manual_seed(91)
    model, _ = make_model()
    model.tv_objective_chunk_blocks = chunk_size
    with torch.no_grad():
        model.draft_model.markov_head.markov_w2.weight.normal_()
    hidden = torch.randn(1, 2, 3, 16, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    anchors = torch.tensor([[0, 3]])
    proposals = torch.tensor([[[2, 3, 4], [5, 6, 7]]])
    valid = torch.tensor([[[True, True, True], [True, True, False]]])
    teacher = torch.randn(1, 7, 16, requires_grad=True)
    parameters = [hidden, *model.draft_model.markov_head.parameters()]
    reference = teacher_forced_tv_forward(
        model, ids, torch.ones_like(ids), anchors, hidden, valid, teacher
    )
    reference_grads = torch.autograd.grad(reference[0], parameters)
    proposal_hidden = (
        teacher.detach()[:, [0, 1, 2, 3, 4, 5]].reshape_as(hidden).requires_grad_()
    )
    result = proposal_tv_forward(
        model, ids, anchors, hidden, proposals, valid, proposal_hidden
    )
    for left, right in zip(result[:5], reference[:5]):
        torch.testing.assert_close(left, right)
    for left, right in zip(torch.autograd.grad(result[0], parameters), reference_grads):
        torch.testing.assert_close(left, right)
    assert teacher.grad is None and proposal_hidden.grad is None
    assert model.lm_head.weight.grad is None


def test_rejection_uses_proposal_predecessors_and_full_suffix_in_real_dspark():
    torch.manual_seed(4)
    model, target = make_model()
    configure_tv_acceptance(
        model,
        temperature=0.8,
        chunk_blocks=1,
        eos_token_ids=[9],
        prefix_mode="proposal",
    )
    with torch.no_grad():
        model.draft_model.markov_head.markov_w2.weight.normal_(std=0.1)
    ids = torch.tensor([[1, 2, 3, 9]])
    context = target.generate_dflash_data(
        ids, torch.ones_like(ids), torch.ones_like(ids)
    )
    teacher = torch.randn(1, 1, 7, 16, requires_grad=True)
    proposals = torch.tensor([[[4, 5, 6, 7, 8, 10, 11]]])
    kwargs = dict(
        input_ids=ids,
        hidden_states=context.hidden_states.requires_grad_(),
        attention_mask=torch.ones_like(ids),
        loss_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        anchor_mode="replay",
        anchor_positions=torch.tensor([[2]]),
        block_keep_mask=torch.tensor([[True]]),
        proposal_ids=proposals,
        proposal_valid_mask=torch.ones_like(proposals, dtype=torch.bool),
        proposal_teacher_hidden=teacher,
    )
    head = model.draft_model.markov_head
    with (
        patch("torch.rand", side_effect=AssertionError("must not resample")),
        patch.object(head, "apply_block_logits", wraps=head.apply_block_logits) as spy,
        patch.object(
            model.draft_model, "forward", wraps=model.draft_model.forward
        ) as backbone,
    ):
        result = model(**kwargs)
        assert spy.call_args.kwargs["token_ids"].tolist() == [[3, 4, 5, 6, 7, 8, 10]]
        original_noise = backbone.call_args.kwargs["noise_embedding"].detach().clone()
        result[0].backward()
        assert result[4].tolist() == [1] * 7
        assert head.markov_w2.weight.grad.abs().sum() > 0
        # Changing proposal tokens cannot enter the bidirectional backbone.
        kwargs["proposal_ids"] = torch.tensor([[[11, 10, 8, 7, 6, 5, 4]]])
        model(**kwargs)
        torch.testing.assert_close(
            backbone.call_args.kwargs["noise_embedding"], original_noise
        )
    assert teacher.grad is None and context.hidden_states.grad is None
    assert model.lm_head.weight.grad is None


def test_parallel_replay_logits_equal_sequential_markov_sampling():
    torch.manual_seed(75)
    model, _ = make_model()
    head = model.draft_model.markov_head
    with torch.no_grad():
        head.markov_w2.weight.normal_(std=0.1)
        hidden = torch.randn(1, 2, 7, 16)
        base = model.lm_head(hidden)
        roots = torch.tensor([[2, 8]])
        previous = roots
        proposal, rollout_logits = [], []
        for position in range(7):
            logits = base[:, :, position] + head.compute_step_bias(previous)
            previous = torch.multinomial(
                torch.softmax(logits.reshape(-1, 16) / 0.8, -1), 1
            ).reshape(1, 2)
            rollout_logits.append(logits)
            proposal.append(previous)
        proposals = torch.stack(proposal, -1)
        predecessors = torch.cat([roots[..., None], proposals[..., :-1]], -1)
        replay_logits = head.apply_block_logits(
            base, token_ids=predecessors, hidden_states=hidden
        )
    torch.testing.assert_close(replay_logits, torch.stack(rollout_logits, -2))


def test_training_capture_passes_exact_selection_and_uses_verification_hidden():
    from specforge.legacy.dspark_training.trainer import _forward_dspark_data_batch

    data = collate([proposal_trace()])
    data["tv_global_block_count"] = torch.tensor(1.0)
    captured = []
    teacher = torch.randn(1, 1, 3, 16)

    def generate(ids, attention, loss, **kwargs):
        captured.append(kwargs["proposal_selection"])
        return SimpleNamespace(
            hidden_states=torch.randn(1, 4, 16),
            last_hidden_states=None,
            proposal_teacher_hidden=teacher,
        )

    args = SimpleNamespace(
        anchor_mode="replay",
        tv_prefix_mode="proposal",
        draft_version=0,
        rollout_target_version="t",
        seed=0,
        multi_teacher_oracle_export_dir=None,
        multi_teacher_oracle_cache=None,
        step_seeded_rollouts=False,
    )
    calls = []
    with patch(
        "specforge.legacy.dspark_training.trainer.dist.get_rank", return_value=0
    ):
        _forward_dspark_data_batch(
            data,
            args,
            "cpu",
            SimpleNamespace(generate_dflash_data=generate),
            lambda **kwargs: calls.append(kwargs),
            True,
            1,
        )
    assert captured == [dict(sample_ids=["a"], round_ids=[[0]], anchors=[[2]])]
    torch.testing.assert_close(calls[0]["proposal_teacher_hidden"], teacher)
    assert calls[0]["last_hidden_states"] is None
