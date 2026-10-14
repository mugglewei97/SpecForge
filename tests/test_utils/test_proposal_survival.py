"""Enumerate complete proposal trees to check accepted-length values AND gradients."""

import itertools
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from specforge.core.tv_acceptance import (
    configure_tv_acceptance,
    proposal_tv_forward,
    tv_acceptance_terms,
)


def survival_terms(q, p, ids, mask=None):
    if mask is None:
        mask = torch.ones_like(ids, dtype=torch.bool)
    return tv_acceptance_terms(
        q, p, mask, proposal_ids=ids, length_estimator="proposal-survival"
    )


def node_index(prefix):
    index = 0
    for token in prefix:
        index = 2 * index + 1 + token
    return index


@pytest.mark.parametrize("eos", [None, 1])
@pytest.mark.parametrize("budget", [2, 3])
def test_enumerated_values_and_gradients_equal_exact_accepted_length(eos, budget):
    # Every node has a distinct conditional distribution, so losing the
    # dependence between sampled tokens and later prefixes fails this test.
    q = torch.tensor(
        [[.4, .6], [.7, .3], [.2, .8], [.6, .4], [.35, .65], [.8, .2], [.3, .7]],
        dtype=torch.double,
    ).log().requires_grad_()
    p = torch.tensor(
        [[.8, .2], [.3, .7], [.6, .4], [.2, .8], [.75, .25], [.5, .5], [.9, .1]],
        dtype=torch.double,
    ).log().requires_grad_()
    width = 3
    qp, pp = q.softmax(-1), p.detach().softmax(-1)

    def exact_tree(prefix=(), mass=None):
        mass = q.new_ones(()) if mass is None else mass
        node = node_index(prefix)
        result = q.new_zeros(())
        for token in range(2):
            accepted = mass * torch.minimum(pp[node, token], qp[node, token])
            result = result + accepted
            if token != eos and len(prefix) + 1 < budget:
                result = result + exact_tree((*prefix, token), accepted)
        return result

    exact = exact_tree()
    estimate, estimated_loss = q.new_zeros(()), q.new_zeros(())
    for path in itertools.product(range(2), repeat=width):
        nodes = [node_index(path[:i]) for i in range(width)]
        ids = torch.tensor([path])
        # Sampling is held fixed during replay. No gradients through this
        # enumeration weight; the estimator must account for the policy itself.
        behavior_mass = qp[nodes, list(path)].detach().prod()
        mask = torch.tensor([
            [i < budget and (eos is None or eos not in path[:i]) for i in range(width)]
        ])
        terms = survival_terms(q[nodes][None], p[nodes][None], ids, mask)
        estimate = estimate + behavior_mass * terms.length_sum
        estimated_loss = estimated_loss + behavior_mass * terms.loss_sum
    torch.testing.assert_close(estimate, exact)
    torch.testing.assert_close(estimated_loss, 1 - exact / width)
    expected_grad = torch.autograd.grad(1 - exact / width, q)[0]
    estimated_grad = torch.autograd.grad(estimated_loss, (q, p), allow_unused=True)
    torch.testing.assert_close(estimated_grad[0], expected_grad)
    assert estimated_grad[1] is None
    assert torch.isfinite(estimated_grad[0]).all()


def test_corrects_the_two_token_counterexample():
    q_root, p_root = [.5, .5], [.9, .1]
    new_length, old_length = 0.0, 0.0
    for token, q_next, p_next in ((0, [.5, .5], [.5, .5]), (1, [.9, .1], [.1, .9])):
        q = torch.tensor([[q_root, q_next]], dtype=torch.double).log()
        p = torch.tensor([[p_root, p_next]], dtype=torch.double).log()
        ids = torch.tensor([[token, 0]])
        new_length += .5 * survival_terms(q, p, ids).length_sum.item()
        old_length += .5 * tv_acceptance_terms(q, p, torch.ones(1, 2)).length_sum.item()
    assert new_length == pytest.approx(.6 + .52)
    assert old_length == pytest.approx(.6 + .36)


def test_single_position_is_tv_independent_of_sampled_token():
    q = torch.tensor([[[.4, .6]]], dtype=torch.double).log().requires_grad_()
    p = torch.tensor([[[.8, .2]]], dtype=torch.double).log()
    old = tv_acceptance_terms(q, p, torch.ones(1, 1))
    reference = torch.autograd.grad(old.loss_sum, q)[0]
    for token in (0, 1):
        new = survival_terms(q, p, torch.tensor([[token]]))
        torch.testing.assert_close(new.loss_sum, old.loss_sum)
        torch.testing.assert_close(torch.autograd.grad(new.loss_sum, q)[0], reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_padding_empty_blocks_and_extreme_sample_logprobs_are_finite(dtype):
    q = torch.tensor([[[0., -1000.], [1., -1.], [float("nan")] * 2]] * 2, dtype=dtype)
    q.requires_grad_()
    p = torch.zeros_like(q, requires_grad=True)
    ids = torch.tensor([[1, 0, -100], [-100, -100, -100]])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    terms = survival_terms(q, p, ids, mask)
    terms.loss_sum.backward()
    assert terms.loss_sum.dtype == torch.float32
    assert terms.block_count.item() == 1
    assert 0 <= terms.loss_sum.item() <= 1
    assert torch.isfinite(q.grad).all()
    assert not q.grad[1].any() and not q.grad[0, 2].any()
    assert p.grad is None


def test_short_blocks_use_fixed_width_and_equal_models_have_zero_gradient():
    q = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
    terms = survival_terms(
        q, q.detach().clone(), torch.tensor([[1, 2, 3]]),
        torch.tensor([[True, False, False]]),
    )
    assert terms.length_sum.item() == pytest.approx(1)
    assert terms.loss_sum.item() == pytest.approx(2 / 3)
    terms.loss_sum.backward()
    assert not q.grad.any()


@pytest.mark.parametrize(
    "ids", [None, torch.zeros(1, 2), torch.zeros(1, 1, dtype=torch.long)]
)
def test_requires_aligned_integer_proposals(ids):
    with pytest.raises(ValueError, match="proposal"):
        survival_terms(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), ids, torch.ones(1, 2)
        )


class TinyMarkov(nn.Module):
    markov_head_type = "vanilla"

    def __init__(self):
        super().__init__()
        self.bias = nn.Embedding(5, 5)

    def apply_block_logits(self, logits, *, token_ids, hidden_states):
        return logits + self.bias(token_ids)


def tiny_model():
    draft = nn.Module()
    draft.markov_head = TinyMarkov()
    draft.config = SimpleNamespace(attention_dropout=0.0)
    return SimpleNamespace(draft_model=draft, lm_head=nn.Linear(4, 5, bias=False))


def test_configuration_requires_proposals_and_deterministic_replay():
    model = tiny_model()
    options = dict(
        temperature=1,
        chunk_blocks=1,
        eos_token_ids=[],
        length_estimator="proposal-survival",
    )
    with pytest.raises(ValueError, match="proposal prefixes"):
        configure_tv_acceptance(model, **options)
    model.draft_model.config.attention_dropout = .1
    with pytest.raises(ValueError, match="dropout"):
        configure_tv_acceptance(model, **options, prefix_mode="proposal")
    model.draft_model.config.attention_dropout = 0
    model.draft_model.dropout = nn.Dropout(.1)
    with pytest.raises(ValueError, match="dropout"):
        configure_tv_acceptance(model, **options, prefix_mode="proposal")


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_proposal_forward_wires_tokens_eos_and_estimator_through_chunks(chunk_size):
    torch.manual_seed(17)
    model = tiny_model()
    configure_tv_acceptance(
        model, temperature=.8, chunk_blocks=chunk_size, eos_token_ids=[4],
        prefix_mode="proposal", length_estimator="proposal-survival",
    )
    hidden = torch.randn(1, 2, 3, 4, requires_grad=True)
    teacher = torch.randn(1, 2, 3, 4, requires_grad=True)
    ids, anchors = torch.tensor([[0, 1, 2]]), torch.tensor([[0, 2]])
    proposals = torch.tensor([[[1, 4, 3], [3, 2, 1]]])
    valid = torch.ones_like(proposals, dtype=torch.bool)
    output = proposal_tv_forward(model, ids, anchors, hidden, proposals, valid, teacher)
    previous = torch.tensor([[[0, 1, 4], [2, 3, 2]]])
    logits = model.draft_model.markov_head.apply_block_logits(
        model.lm_head(hidden), token_ids=previous, hidden_states=hidden,
    )
    valid[0, 0, 2] = False
    expected = tv_acceptance_terms(
        logits, model.lm_head(teacher), valid, .8, proposal_ids=proposals,
        length_estimator="proposal-survival",
    )
    torch.testing.assert_close(output[0], expected.loss_sum / 2)
    torch.testing.assert_close(
        output[5]["tv_acceptance_length_proxy"], expected.length_sum / 2
    )
    assert output[4].tolist() == [2, 2, 1]
    parameters = [hidden, *model.draft_model.parameters()]
    actual_grads = torch.autograd.grad(
        output[0], [*parameters, teacher], allow_unused=True
    )
    expected_grads = torch.autograd.grad(expected.loss_sum / 2, parameters)
    for actual, reference in zip(actual_grads[:-1], expected_grads):
        torch.testing.assert_close(actual, reference)
    assert actual_grads[-1] is None
    assert model.lm_head.weight.grad is None
