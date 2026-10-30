"""Exact path enumeration, rollout/replay and training lifecycle coverage."""

from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from specforge.core.tv_acceptance import configure_tv_acceptance, reduce_tv_terms
from specforge.core.tv_acceptance_beta import (
    ProposalRollout,
    sample_beta_proposals,
    sampled_tv_beta_forward,
    tv_acceptance_beta_terms,
    verify_beta_proposals,
)
from specforge.inference.target_engine.proposal_verification import (
    ProposalVerificationSession,
)

from .test_tv_acceptance import RecordingTarget, make_model


def beta_model():
    model, target = make_model()
    configure_tv_acceptance(
        model, temperature=0.8, chunk_blocks=2, eos_token_ids=[], beta=True
    )
    with torch.no_grad():
        model.draft_model.markov_head.markov_w2.weight.normal_(std=0.15)
    return model, target


@pytest.mark.parametrize("eos", [None, 1])
def test_enumerated_value_and_policy_gradient_match_exact_tree(eos):
    """Conditional distributions vary with *all* previous tokens, not just depth."""
    torch.manual_seed(81)
    logits = torch.randn(7, 2, dtype=torch.double, requires_grad=True)
    target = torch.softmax(torch.randn(7, 2, dtype=torch.double), -1)
    old = torch.softmax(logits.detach(), -1)

    def exact(node=0, depth=0):
        mass = torch.minimum(target[node], torch.softmax(logits[node], -1))
        value = mass.sum()
        if depth < 2:
            value = value + sum(
                mass[d] * exact(2 * node + 1 + d, depth + 1)
                for d in range(2)
                if d != eos
            )
        return value

    estimate = logits.new_zeros(())
    for path in product(range(2), repeat=3):
        nodes, node = [], 0
        for token in path:
            nodes.append(node)
            node = 2 * node + 1 + token
        tokens = torch.tensor([path])
        old_prob = old[nodes].gather(-1, tokens.T).squeeze(-1)
        valid = torch.ones(1, 3, dtype=torch.bool)
        if eos in path:
            valid[:, path.index(eos) + 1 :] = False
        terms = tv_acceptance_beta_terms(
            logits[nodes][None],
            target[nodes].log()[None],
            valid,
            tokens,
            old_prob.log()[None],
        )
        estimate = estimate + old_prob.prod() * terms.loss_sum
    reference = 1 - exact()
    torch.testing.assert_close(estimate, reference)
    torch.testing.assert_close(
        torch.autograd.grad(estimate, logits)[0],
        torch.autograd.grad(reference, logits)[0],
    )


def test_beta_formula_no_block_rescaling_and_no_ratio_clamp():
    p = torch.tensor([[[0.8, 0.2], [0.5, 0.5], [0.4, 0.6]]], dtype=torch.double)
    q = torch.tensor([[[0.6, 0.4], [0.2, 0.8], [0.3, 0.7]]], dtype=torch.double)
    tokens = torch.tensor([[1, 0, 0]])
    old = q.gather(-1, tokens[..., None]).squeeze(-1).log()
    terms = tv_acceptance_beta_terms(q.log(), p.log(), torch.ones(1, 3), tokens, old)
    length = 0.8 + 0.5 * 0.7 + 0.5 * 1.0 * 0.9
    torch.testing.assert_close(
        terms.length_sum, torch.tensor(length, dtype=torch.double)
    )
    torch.testing.assert_close(
        terms.loss_sum, torch.tensor(1 - length, dtype=torch.double)
    )
    partial = tv_acceptance_beta_terms(
        q.log(), p.log(), torch.tensor([[1, 0, 1]]), tokens, old
    )
    torch.testing.assert_close(
        partial.loss_sum, torch.tensor(1 - 0.8, dtype=torch.double)
    )
    # Off-policy inputs are only for this derivative test: a training ratio must
    # be allowed above 1, rather than clipping the importance-weighted mass.
    changed_old = old.clone()
    changed_old[0, 0] = torch.tensor(0.1).log()
    unclamped = tv_acceptance_beta_terms(
        q.log(), p.log(), torch.ones(1, 3), tokens, changed_old
    )
    assert unclamped.length_sum > terms.length_sum


def test_beta_no_score_grad_ablation_removes_ratio_gradient():
    """sg[alpha] prefix factors: value drops sampling noise, gradient loses the
    score-function term but keeps the pathwise overlap derivative."""
    torch.manual_seed(19)
    q = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    p = torch.randn(2, 3, 5, dtype=torch.double)
    tokens = torch.tensor([[1, 2, 3], [3, 0, 1]])
    old = torch.log_softmax(q.detach(), -1).gather(-1, tokens[..., None]).squeeze(-1)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    terms = tv_acceptance_beta_terms(
        q, p, mask, tokens, old, ablation="no-score-grad"
    )
    terms.loss_sum.backward()
    ablated_grad = q.grad.clone()

    # Independent reference: prefix products of detached expected acceptances.
    q.grad = None
    valid = mask.bool().int().cumprod(-1).bool()
    log_q = torch.log_softmax(q, -1)
    log_p = torch.log_softmax(p, -1)
    overlap = torch.minimum(log_p.exp(), log_q.exp()).sum(-1).clamp(0, 1)
    overlap = torch.where(valid, overlap, 0.0)
    prefix = torch.cat(
        (torch.ones_like(overlap[..., :1]), overlap.detach()[..., :-1]), -1
    )
    ref_length = (prefix.cumprod(-1) * overlap).sum(-1)
    nonempty = valid.any(-1)
    ref_loss = torch.where(nonempty, 1 - ref_length, ref_length.new_zeros(())).sum()
    torch.testing.assert_close(terms.length_sum, ref_length.sum())
    torch.testing.assert_close(terms.loss_sum, ref_loss)
    ref_loss.backward()
    torch.testing.assert_close(ablated_grad, q.grad)

    # The removed score term must change the gradient: full formula disagrees.
    q.grad = None
    full = tv_acceptance_beta_terms(q, p, mask, tokens, old)
    full.loss_sum.backward()
    assert not torch.allclose(ablated_grad, q.grad)


def test_beta_no_prefix_weight_ablation_optimizes_plain_overlap_sum():
    """Without prefix factors every position contributes its own overlap."""
    torch.manual_seed(23)
    q = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    p = torch.randn(2, 3, 5, dtype=torch.double)
    tokens = torch.tensor([[1, 2, 3], [3, 0, 1]])
    old = torch.log_softmax(q.detach(), -1).gather(-1, tokens[..., None]).squeeze(-1)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    terms = tv_acceptance_beta_terms(
        q, p, mask, tokens, old, ablation="no-prefix-weight"
    )
    terms.loss_sum.backward()
    ablated_grad = q.grad.clone()

    q.grad = None
    valid = mask.bool().int().cumprod(-1).bool()
    log_q = torch.log_softmax(q, -1)
    log_p = torch.log_softmax(p, -1)
    overlap = torch.minimum(log_p.exp(), log_q.exp()).sum(-1).clamp(0, 1)
    overlap = torch.where(valid, overlap, 0.0)
    nonempty = valid.any(-1)
    ref_length = overlap.sum(-1)
    ref_loss = torch.where(nonempty, 1 - ref_length, ref_length.new_zeros(())).sum()
    torch.testing.assert_close(terms.length_sum, ref_length.sum())
    torch.testing.assert_close(terms.loss_sum, ref_loss)
    ref_loss.backward()
    torch.testing.assert_close(ablated_grad, q.grad)

    # Sampled tokens only matter through the prefix factors; dropping them must
    # make the objective token-independent.
    other_tokens = torch.tensor([[0, 0, 0], [1, 1, 1]])
    rerolled = tv_acceptance_beta_terms(
        q.detach(), p, mask, other_tokens, old, ablation="no-prefix-weight"
    )
    torch.testing.assert_close(rerolled.loss_sum, terms.loss_sum.detach())


def test_beta_unknown_ablation_is_rejected():
    logits = torch.zeros(1, 2, 3, dtype=torch.double)
    tokens = torch.zeros(1, 2, dtype=torch.long)
    old = torch.zeros(1, 2, dtype=torch.double)
    with pytest.raises(ValueError, match="Unknown TV beta ablation"):
        tv_acceptance_beta_terms(
            logits, logits, torch.ones(1, 2), tokens, old, ablation="no-grad"
        )


@pytest.mark.parametrize("dtype", [torch.double, torch.float16, torch.bfloat16])
def test_beta_gradients_nan_padding_and_log_space_small_probabilities(dtype):
    q = torch.tensor(
        [[[0.0, -1000.0], [0.0, -1000.0], [float("nan"), float("nan")]]],
        dtype=dtype,
        requires_grad=True,
    )
    p = q.detach().clone().requires_grad_()
    old = torch.tensor(
        [[-1000.0, -1000.0, float("nan")]], dtype=torch.double, requires_grad=True
    )
    terms = tv_acceptance_beta_terms(
        q, p, torch.tensor([[1, 1, 0]]), torch.tensor([[1, 1, -1]]), old
    )
    assert terms.length_sum.item() == 2
    terms.loss_sum.backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad[0, 2].count_nonzero() == 0
    assert p.grad is None and old.grad is None
    assert terms.loss_sum.dtype == (
        torch.double if dtype == torch.double else torch.float32
    )


def test_beta_gradcheck_and_empty_disjoint_blocks():
    torch.manual_seed(44)
    q = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    p = torch.randn_like(q)
    tokens = torch.tensor([[1, 2, 3], [3, 0, 1]])
    old = (
        torch.log_softmax(q.detach() / 0.6, -1)
        .gather(-1, tokens[..., None])
        .squeeze(-1)
    )
    mask = torch.tensor([[1, 1, 1], [0, 0, 0]])

    def objective(x):
        return tv_acceptance_beta_terms(x, p, mask, tokens, old, 0.6).loss_sum

    assert torch.autograd.gradcheck(objective, (q,))
    objective(q).backward()
    assert q.grad[1].count_nonzero() == 0
    disjoint = tv_acceptance_beta_terms(
        torch.tensor([[[0.0, -torch.inf]] * 2]),
        torch.tensor([[[-torch.inf, 0.0]] * 2]),
        torch.ones(1, 2),
        torch.zeros(1, 2).long(),
        torch.zeros(1, 2),
    )
    assert disjoint.loss_sum.item() == 1 and disjoint.length_sum.item() == 0


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_beta_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        tv_acceptance_beta_terms(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.ones(1, 1),
            torch.zeros(1, 1).long(),
            torch.zeros(1, 1),
            temperature,
        )


def test_sampled_eos_and_same_policy_replay():
    torch.manual_seed(72)
    model, _ = beta_model()
    model.tv_eos_token_ids = (3,)
    hidden = torch.randn(2, 7, 16, requires_grad=True)
    # The first proposal is EOS, but the second row survives all seven steps.
    samples = [torch.tensor([[3], [4]])] + [torch.tensor([[5], [6]])] * 6
    with patch("torch.multinomial", side_effect=samples):
        rollout = sample_beta_proposals(
            model, hidden, torch.tensor([1, 2]), torch.ones(2, 7)
        )
    assert rollout.valid.sum(-1).tolist() == [1, 7]
    assert rollout.previous[:, 0].tolist() == [1, 2]
    assert torch.equal(rollout.previous[:, 1:], rollout.tokens[:, :-1])
    logits = model.draft_model.markov_head.apply_block_logits(
        model.lm_head(hidden),
        token_ids=rollout.previous,
        hidden_states=hidden,
    )
    replay = torch.log_softmax(logits.float() / model.tv_temperature, -1)
    replay = replay.gather(-1, rollout.tokens[..., None]).squeeze(-1)
    torch.testing.assert_close(replay[rollout.valid], rollout.log_probs[rollout.valid])
    assert not rollout.log_probs.requires_grad


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_target_reuse_stops_after_first_divergence_even_if_tokens_match_again(batch_size):
    target = RecordingTarget(16)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    anchors = torch.tensor([[0, 1, 2]])
    tokens = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7, 8],  # all prefixes reused
            [3, 12, 5, 6, 7, 8, 9],  # divergence at slot 1
            [13, 5, 6, 7, 8, 9, 0],
        ]
    )
    valid = torch.tensor([[1] * 7, [1] * 7, [1] * 6 + [0]]).bool()
    rollout = ProposalRollout(tokens, torch.zeros(3, 7), tokens, valid)
    with ProposalVerificationSession(target) as session:
        original = session.capture(ids, torch.ones_like(ids), torch.ones_like(ids))
        verified = verify_beta_proposals(
            session, ids, anchors, rollout, original.last_hidden_states, batch_size
        )
    # First block adds no candidate request. Only prefixes through d_(K-1)
    # are needed, so no candidate includes its final proposal/bonus position.
    assert target.requests == [
        ids[0].tolist(),
        [1, 2, 3, 12, 5, 6, 7, 8],
        [1, 2, 3, 13, 5, 6, 7, 8],
    ]
    for block, anchor in enumerate(anchors[0]):
        n = valid[block].sum().item()
        candidate = torch.cat((ids[0, : anchor + 1], tokens[block, : n - 1]))[None]
        expected = target.generate_dflash_data(
            candidate, torch.ones_like(candidate), torch.ones_like(candidate)
        )
        torch.testing.assert_close(
            verified[block, :n], expected.last_hidden_states[0, anchor : anchor + n]
        )


@pytest.mark.parametrize("chunk_size", [1, 4])
@pytest.mark.parametrize("markov", [True, False])
def test_checkpoint_replay_value_and_gradients_match_full_candidate_reference(
    chunk_size, markov
):
    torch.manual_seed(73)
    model, target = beta_model()
    model.tv_objective_chunk_blocks = chunk_size
    if not markov:
        model.draft_model.markov_head = None
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    anchors = torch.tensor([[0, 3]])
    hidden = torch.randn(1, 2, 7, 16, requires_grad=True)
    valid = torch.tensor([[[1] * 7, [1] * 5 + [0] * 2]]).bool()
    with ProposalVerificationSession(target) as session:
        original = session.capture(ids, torch.ones_like(ids), torch.ones_like(ids))
        with patch(
            "specforge.core.tv_acceptance_beta.sample_beta_proposals",
            wraps=sample_beta_proposals,
        ) as sampler:
            # Capture rollout explicitly through a wrapper, not by resampling.
            saved = []

            def sample(*args):
                result = sample_beta_proposals(*args)
                saved.append(result)
                return result

            sampler.side_effect = sample
            result = sampled_tv_beta_forward(
                model,
                ids,
                torch.ones_like(ids),
                anchors,
                hidden,
                valid,
                original.last_hidden_states,
                session,
            )
        requests = list(target.requests)
    with patch("torch.multinomial", side_effect=AssertionError("backward resampled")):
        result[0].backward()
    assert target.requests == requests
    actual_grad = hidden.grad.clone()
    head_grad = (
        model.draft_model.markov_head.markov_w2.weight.grad.clone() if markov else None
    )
    model.zero_grad()
    reference_hidden = hidden.detach().clone().requires_grad_()
    rollout = saved[0]
    reference = hidden.new_zeros(())
    for block, anchor in enumerate(anchors[0].tolist()):
        n = rollout.valid[block].sum().item()
        candidate = torch.cat((ids[0, : anchor + 1], rollout.tokens[block, : n - 1]))[
            None
        ]
        output = target.generate_dflash_data(
            candidate, torch.ones_like(candidate), torch.ones_like(candidate)
        )
        h = reference_hidden[:, block, :n]
        logits = model.lm_head(h)
        if markov:
            logits = model.draft_model.markov_head.apply_block_logits(
                logits,
                token_ids=rollout.previous[block : block + 1, :n],
                hidden_states=h,
            )
        reference += (
            tv_acceptance_beta_terms(
                logits,
                model.lm_head(output.last_hidden_states[:, anchor : anchor + n]),
                rollout.valid[block : block + 1, :n],
                rollout.tokens[block : block + 1, :n],
                rollout.log_probs[block : block + 1, :n],
                model.tv_temperature,
            ).loss_sum
            / 2
        )
    reference.backward()
    torch.testing.assert_close(result[0], reference)
    torch.testing.assert_close(actual_grad, reference_hidden.grad)
    if markov:
        torch.testing.assert_close(
            head_grad, model.draft_model.markov_head.markov_w2.weight.grad
        )
    assert model.lm_head.weight.grad is None
    assert "tv_acceptance_length_estimate" in result[-1]


def test_real_beta_training_batch_closes_session_before_backward():
    from specforge.legacy.dspark_training.trainer import _forward_dspark_data_batch

    torch.manual_seed(34)
    model, target = beta_model()
    session = ProposalVerificationSession(target)
    target.proposal_verification = lambda: session
    ids = torch.randint(0, 15, (2, 12))
    data = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        loss_mask=torch.ones(2, 12),
        sample_id=torch.arange(2),
        source_id=torch.zeros(2),
    )
    args = SimpleNamespace(
        seed=42,
        multi_teacher_oracle_export_dir=None,
        multi_teacher_oracle_cache=None,
        step_seeded_rollouts=False,
        dspark_loss_type="tv-acceptance-beta",
    )
    with patch(
        "specforge.legacy.dspark_training.trainer.dist.get_rank", return_value=0
    ):
        result = _forward_dspark_data_batch(
            data, args, torch.device("cpu"), target, model, True, 1
        )
    assert session.closed
    count = len(target.requests)
    with patch("torch.multinomial", side_effect=AssertionError("backward resampled")):
        result[0].backward()
    assert count == len(target.requests)
    assert -6 <= result[0].item() <= 1
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.draft_model.layers.parameters()
    )
    assert model.draft_model.markov_head.markov_w2.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad is None


def test_beta_actual_block_size_and_missing_session_are_rejected():
    model, _ = beta_model()
    model.draft_model.config.block_size = 8
    with pytest.raises(ValueError, match="actual model block_size=7"):
        configure_tv_acceptance(
            model, temperature=1, chunk_blocks=1, eos_token_ids=[], beta=True
        )
    with pytest.raises(ValueError, match="verification session"):
        sampled_tv_beta_forward(
            model,
            torch.ones(1, 8).long(),
            torch.ones(1, 8),
            torch.zeros(1, 1).long(),
            torch.randn(1, 1, 7, 16),
            torch.ones(1, 1, 7),
            torch.randn(1, 8, 16),
            None,
        )


@pytest.mark.parametrize("empty_rank", [False, True])
def test_beta_global_block_normalization_including_empty_rank(empty_rank):
    torch.manual_seed(19)
    logits = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    teacher = torch.randn_like(logits)
    tokens = torch.zeros(2, 3, dtype=torch.long)
    old = torch.log_softmax(logits.detach(), -1)[..., 0]
    mask = torch.zeros(2, 3) if empty_rank else torch.ones(2, 3)
    terms = tv_acceptance_beta_terms(logits, teacher, mask, tokens, old)
    count = 1 if empty_rank else 3

    def all_reduce(tensor):
        if tensor.shape == (3,):
            tensor[2] = count

    with (
        patch("specforge.core.tv_acceptance.dist.is_initialized", return_value=True),
        patch("specforge.core.tv_acceptance.dist.get_world_size", return_value=2),
        patch("specforge.core.tv_acceptance.dist.all_reduce", side_effect=all_reduce),
    ):
        loss = reduce_tv_terms(terms)[0]
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(terms.loss_sum / count, logits)[0]
    torch.testing.assert_close(grad / 2, expected)
    if empty_rank:
        assert grad.count_nonzero() == 0


def test_beta_training_failure_closes_target_session():
    from specforge.legacy.dspark_training.trainer import _forward_dspark_data_batch

    target = RecordingTarget(16)
    session = ProposalVerificationSession(target)
    target.proposal_verification = lambda: session
    ids = torch.ones(1, 8).long()
    data = dict(
        input_ids=ids,
        attention_mask=ids,
        loss_mask=ids.float(),
        sample_id=torch.zeros(1),
        source_id=torch.zeros(1),
    )
    args = SimpleNamespace(
        seed=42,
        multi_teacher_oracle_export_dir=None,
        multi_teacher_oracle_cache=None,
        step_seeded_rollouts=False,
        dspark_loss_type="tv-acceptance-beta",
    )

    def fail(**kwargs):
        raise RuntimeError("draft failed")

    with patch(
        "specforge.legacy.dspark_training.trainer.dist.get_rank", return_value=0
    ):
        with pytest.raises(RuntimeError, match="draft failed"):
            _forward_dspark_data_batch(
                data, args, torch.device("cpu"), target, fail, True, 1
            )
    assert session.closed
