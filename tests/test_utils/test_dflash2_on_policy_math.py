"""Sparse replay must preserve the dense beta value and policy gradient."""

import pytest
import torch

from specforge.core.dflash2_on_policy import candidate_beta_terms
from specforge.core.tv_acceptance_beta import tv_acceptance_beta_terms
from specforge.modeling.draft.dflash2 import CandidateSelector


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.6, 1.3])
def test_candidate_beta_matches_dense_value_and_gradient(dtype, temperature):
    torch.manual_seed(931)
    batch, width, vocab, support = 3, 7, 11, 4
    candidate_ids = torch.rand(batch, width, vocab).argsort(-1)[..., :support]
    scores = torch.randn(batch, width, support, dtype=dtype, requires_grad=True)
    target_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    target = torch.randn(
        batch, width, vocab, dtype=target_dtype, requires_grad=True
    )
    full_log_p = (target / temperature).log_softmax(-1)
    tokens = candidate_ids.gather(
        -1, torch.randint(support, (batch, width, 1))
    ).squeeze(-1)
    # Independent behavior probabilities exercise the exact IS denominator,
    # including ratios greater than one; they must receive no gradient.
    behavior = torch.randn(
        batch, width, support, dtype=target_dtype
    ).log_softmax(-1)
    chosen = (candidate_ids == tokens[..., None]).long().argmax(-1)
    old = behavior.gather(-1, chosen[..., None]).squeeze(-1).requires_grad_()
    valid = torch.tensor(
        [[1] * 7, [1, 1, 1, 0, 0, 0, 0], [0] * 7], dtype=torch.bool
    )
    sparse = candidate_beta_terms(
        scores,
        full_log_p.gather(-1, candidate_ids),
        valid,
        candidate_ids,
        tokens,
        old,
        target.argmax(-1),
        temperature,
    )
    dense_logits = torch.full(
        (batch, width, vocab), -torch.inf, dtype=dtype
    ).scatter(-1, candidate_ids, scores)
    dense = tv_acceptance_beta_terms(
        dense_logits, target, valid, tokens, old, temperature
    )
    for sparse_value, dense_value in zip(sparse, dense):
        torch.testing.assert_close(sparse_value, dense_value)
    sparse_gradient = torch.autograd.grad(sparse.loss_sum, scores, retain_graph=True)[0]
    dense_gradient = torch.autograd.grad(dense.loss_sum, scores)[0]
    torch.testing.assert_close(sparse_gradient, dense_gradient)
    assert target.grad is None and old.grad is None
    assert sparse_gradient[1, 3:].count_nonzero() == 0
    assert sparse_gradient[2].count_nonzero() == 0


def test_candidate_beta_uses_full_target_normalizer_without_block_rescaling():
    scores = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    candidate_ids = torch.tensor([[[1, 3]]])
    # The candidate set covers only 0.3 target mass. Renormalizing it would
    # incorrectly turn this into a 5/6 overlap rather than 0.3.
    full_target = torch.tensor([[[0.6, 0.1, 0.1, 0.2]]])
    terms = candidate_beta_terms(
        scores, full_target.log().gather(-1, candidate_ids),
        torch.ones(1, 1, dtype=torch.bool), candidate_ids,
        torch.tensor([[1]]), torch.tensor([[0.5]]).log(),
        full_target.argmax(-1), 1.0,
    )
    torch.testing.assert_close(terms.length_sum, torch.tensor(0.3))
    torch.testing.assert_close(terms.loss_sum, torch.tensor(1 - 0.3))


def test_sparse_replay_backpropagates_through_actual_selector_and_hidden():
    torch.manual_seed(719)
    selector = CandidateSelector(
        hidden_size=8, vocab_size=9, state_rank=3, top_k=3,
        initializer_range=0.3,
    ).double()
    with torch.no_grad():
        selector.successor_codebook.normal_(std=0.4)
    hidden = torch.randn(2, 7, 8, dtype=torch.double, requires_grad=True)
    unary = torch.randn(2, 7, 3, dtype=torch.double, requires_grad=True)
    candidates = torch.tensor([0, 3, 7]).expand(2, 7, 3)
    previous = torch.randint(9, (2, 7))
    scores = selector.score_candidates(
        candidate_ids=candidates, unary_logits=unary,
        hidden_states=hidden, predecessor_ids=previous,
    )
    target = torch.full((2, 7, 9), 0.1 / 7, dtype=torch.double)
    target[..., 0] = 0.8
    target[..., 3] = 0.1
    tokens = torch.full((2, 7), 3)
    old = scores.detach().log_softmax(-1)[..., 1]
    terms = candidate_beta_terms(
        scores, target.log().gather(-1, candidates),
        torch.ones(2, 7, dtype=torch.bool), candidates, tokens, old,
        target.argmax(-1), 1.0,
    )
    terms.loss_sum.backward()
    for parameter in selector.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert hidden.grad.abs().sum() > 0
    assert unary.grad.abs().sum() > 0


def test_sparse_empty_or_eos_tail_padding_has_finite_zero_gradient():
    scores = torch.tensor(
        [[[0.3, -0.2], [float("nan"), float("nan")]],
         [[float("nan"), float("nan")], [float("nan"), float("nan")]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[-1.0, -2.0], [float("nan"), float("nan")]],
         [[float("nan"), float("nan")], [float("nan"), float("nan")]]],
        requires_grad=True,
    )
    old = torch.tensor([[-0.5, float("nan")], [float("nan"), float("nan")]])
    valid = torch.tensor([[True, False], [False, False]])
    candidates = torch.tensor([0, 1]).expand(2, 2, 2)
    terms = candidate_beta_terms(
        scores, teacher, valid, candidates, torch.zeros(2, 2).long(), old,
        torch.zeros(2, 2).long(), 1.0,
    )
    assert torch.isfinite(terms.loss_sum)
    assert terms.block_count == 1
    terms.loss_sum.backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0, 1].count_nonzero() == 0
    assert scores.grad[1].count_nonzero() == 0
    assert teacher.grad is None
