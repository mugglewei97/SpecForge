"""Exact finite-distribution checks for real speculative rollout transitions."""

from collections import defaultdict
from itertools import product

import pytest
import torch

from specforge.core.tv_on_policy import (
    prepare_on_policy_prompts,
    verify_on_policy_block,
)


def test_prompt_extraction_discards_answers_and_later_turns():
    ids = torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 0, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    loss = torch.tensor([[0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 0]])
    originals = [value.clone() for value in (ids, attention, loss)]
    prompts = prepare_on_policy_prompts(
        ids, attention, loss, max_seq_len=8, max_new_tokens=5
    )
    assert [value.tolist() for value in prompts.prefixes] == [[10, 11], [20]]
    assert prompts.budgets.tolist() == [5, 5]
    assert prompts.active.tolist() == [True, True]
    # None of the original answer or later user/assistant turns is a prefix.
    changed = ids.clone()
    changed[0, 2:] = 999
    changed[1, 1:] = 888
    again = prepare_on_policy_prompts(
        changed, attention, loss, max_seq_len=8, max_new_tokens=5
    )
    for before, after in zip(prompts.prefixes, again.prefixes):
        torch.testing.assert_close(before, after)
    for original, value in zip(originals, (ids, attention, loss)):
        torch.testing.assert_close(original, value)
    prompts.prefixes[0][0] = 99
    assert ids[0, 0] == 10


def test_prompt_first_token_anchor_inactive_rows_and_length_budgets():
    ids = torch.tensor([[4, 5, 6, 7], [8, 9, 0, 0], [42, 42, 42, 42], [1, 2, 3, 4]])
    attention = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
    loss = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1]])
    prompts = prepare_on_policy_prompts(
        ids, attention, loss, max_seq_len=3, max_new_tokens=8
    )
    assert [value.tolist() for value in prompts.prefixes] == [[4], [8], [0], [1, 2, 3]]
    assert prompts.budgets.tolist() == [2, 0, 0, 0]
    assert prompts.active.tolist() == [True, False, False, False]
    limited = prepare_on_policy_prompts(
        ids, attention, loss, max_seq_len=2, max_new_tokens=0
    )
    assert all(value.numel() <= 2 for value in limited.prefixes)
    assert not limited.active.any()


@pytest.mark.parametrize("attention", [[1, 0, 1], [0, 1, 1]])
def test_prompt_rejects_non_right_padding(attention):
    with pytest.raises(ValueError, match="right padding"):
        prepare_on_policy_prompts(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([attention]),
            torch.tensor([[0, 1, 1]]),
            max_seq_len=8,
            max_new_tokens=3,
        )


def verify(tokens, q, p, *, valid=None, budget=20, accept=None, sample=0.5, eos=()):
    tokens = torch.tensor([tokens], dtype=torch.long)
    q = torch.tensor([q], dtype=torch.double)
    p = torch.tensor([p], dtype=torch.double)
    return verify_on_policy_block(
        tokens,
        (
            torch.ones_like(tokens, dtype=torch.bool)
            if valid is None
            else torch.tensor([valid])
        ),
        q,
        p,
        torch.tensor([budget]),
        eos_token_ids=eos,
        accept_uniforms=torch.tensor(
            [accept or [0.0] * tokens.size(1)], dtype=torch.double
        ),
        sample_uniforms=torch.tensor([sample], dtype=torch.double),
    )


def test_exact_block_joint_distribution_with_residual_and_bonus():
    """Integrate every proposal, acceptance interval, and next-token interval."""
    p = [
        [0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.7, 0.3],
        [0.2, 0.8], [0.4, 0.6], [0.9, 0.1],
    ]
    q = [[0.25, 0.75], [0.6, 0.4], [0.45, 0.55]]
    actual, expected = defaultdict(float), defaultdict(float)
    for d0, d1 in product(range(2), repeat=2):
        target = [p[0], p[1 + d0], p[3 + 2 * d0 + d1]]
        draft = [q[0], q[1 + d0]]
        proposal_mass = draft[0][d0] * draft[1][d1]
        alpha = [min(1.0, target[j][d] / draft[j][d]) for j, d in enumerate((d0, d1))]
        for decisions in product((False, True), repeat=2):
            acceptance_mass = 1.0
            draws = []
            for accepted, probability in zip(decisions, alpha):
                acceptance_mass *= probability if accepted else 1 - probability
                draws.append(probability / 2 if accepted else (1 + probability) / 2)
            if acceptance_mass == 0:
                continue
            count = next((i for i, accepted in enumerate(decisions) if not accepted), 2)
            distribution = target[count] if count == 2 else [
                max(0.0, pi - qi) for pi, qi in zip(target[count], draft[count])
            ]
            total = sum(distribution)
            distribution = [value / total for value in distribution]
            lower = 0.0
            for token, probability in enumerate(distribution):
                if probability:
                    step = verify(
                        [d0, d1], draft, target,
                        accept=draws, sample=lower + probability / 2,
                    )
                    assert step.emit_next.item() and step.active_next.item()
                    key = (d0, d1, step.accepted_counts.item(), step.next_tokens.item())
                    actual[key] += proposal_mass * acceptance_mass * probability
                lower += probability
        # Independently compute the standard forward rejection-sampling law.
        for count in range(3):
            event_mass = 1.0
            for j in range(count):
                event_mass *= alpha[j]
            if count < 2:
                event_mass *= 1 - alpha[count]
                distribution = [
                    max(0.0, pi - qi)
                    for pi, qi in zip(target[count], draft[count])
                ]
                normalizer = sum(distribution)
                distribution = [
                    value / normalizer if normalizer else 0.0
                    for value in distribution
                ]
            else:
                distribution = target[2]
            for token, probability in enumerate(distribution):
                if probability * event_mass:
                    expected[(d0, d1, count, token)] += (
                        proposal_mass * event_mass * probability
                    )
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], abs=1e-14)
    assert sum(actual.values()) == pytest.approx(1.0)
    first_token = defaultdict(float)
    for (d0, _, count, token), probability in actual.items():
        first_token[d0 if count else token] += probability
    assert [first_token[0], first_token[1]] == pytest.approx(p[0])


def test_rejected_eos_does_not_stop_but_emitted_eos_does():
    rejected = verify(
        [1, 0], [[0.2, 0.8], [0.5, 0.5]], [[0.8, 0.2], [0.5, 0.5], [0.5, 0.5]],
        accept=[0.9, 0.0], eos=[1],
    )
    assert rejected.accepted_counts.item() == 0
    assert rejected.next_tokens.item() == 0
    assert rejected.emit_next.item() and rejected.active_next.item()
    corrected = verify(
        [0, 0], [[0.8, 0.2], [0.5, 0.5]], [[0.2, 0.8], [0.5, 0.5], [0.5, 0.5]],
        accept=[0.9, 0.0], eos=[1],
    )
    assert corrected.next_tokens.item() == 1 and corrected.emit_next.item()
    assert not corrected.active_next.item()
    bonus = verify([0], [[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], eos=[1])
    assert bonus.accepted_counts.item() == 1
    assert bonus.next_tokens.item() == 1 and bonus.emit_next.item()
    assert not bonus.active_next.item()


def test_accepted_eos_stops_without_modifying_full_proposal_block():
    tokens = torch.tensor([[0, 1, 0], [0, 0, 1]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    q = torch.full((2, 3, 2), 0.5)
    p = torch.full((2, 4, 2), 0.5)
    budget = torch.tensor([10, 10])
    values = (tokens, valid, q, p, budget)
    originals = tuple(value.clone() for value in values)
    step = verify_on_policy_block(
        *values, eos_token_ids=[1], accept_uniforms=torch.zeros(2, 3),
        sample_uniforms=torch.zeros(2),
    )
    assert step.accepted_counts.tolist() == [2, 3]
    assert step.emitted_counts.tolist() == [2, 3]
    assert not step.emit_next.any() and not step.active_next.any()
    for value, original in zip(values, originals):
        torch.testing.assert_close(value, original)


def test_budget_tail_and_inactive_rows_never_emit_extra_tokens():
    tokens = torch.zeros(5, 3, dtype=torch.long)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    valid[0] = False
    q = torch.full((5, 3, 2), 0.5)
    p = torch.full((5, 4, 2), 0.5)
    step = verify_on_policy_block(
        tokens, valid, q, p, torch.tensor([0, 1, 2, 4, 10]),
        accept_uniforms=torch.zeros(5, 3), sample_uniforms=torch.zeros(5),
    )
    assert step.accepted_counts.tolist() == [0, 1, 2, 3, 3]
    assert step.emit_next.tolist() == [False, False, False, True, True]
    assert step.emitted_counts.tolist() == [0, 1, 2, 4, 4]
    assert step.active_next.tolist() == [False, False, False, False, True]
    rejected = verify(
        [0, 0], [[0.8, 0.2], [0.8, 0.2]], [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]],
        budget=2, accept=[0.0, 0.9],
    )
    assert rejected.accepted_counts.item() == 1 and rejected.emit_next.item()
    assert rejected.emitted_counts.item() == 2 and not rejected.active_next.item()


def test_small_residual_is_sampled_instead_of_falling_back_to_target():
    step = verify(
        [0], [[0.50000001, 0.49999999]], [[0.5, 0.5], [0.5, 0.5]],
        accept=[0.9999999999], sample=0.0,
    )
    assert step.accepted_counts.item() == 0
    assert step.next_tokens.item() == 1


def test_first_rejection_keeps_all_original_proposals_for_beta_replay():
    tokens = torch.zeros(1, 3, dtype=torch.long)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    q = torch.tensor([0.8, 0.2]).expand(1, 3, 2).clone()
    p = torch.tensor([0.2, 0.8]).expand(1, 4, 2).clone()
    original_tokens, original_valid = tokens.clone(), valid.clone()
    step = verify_on_policy_block(
        tokens, valid, q, p, torch.tensor([10]),
        accept_uniforms=torch.full((1, 3), 0.9), sample_uniforms=torch.zeros(1),
    )
    assert step.accepted_counts.item() == 0
    assert step.emitted_counts.item() == 1
    torch.testing.assert_close(tokens, original_tokens)
    torch.testing.assert_close(valid, original_valid)
    assert valid.sum() == 3


def test_sparse_support_keeps_full_target_normalization():
    step = verify(
        [0], [[1.0, 0.0, 0.0]], [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
        accept=[0.4], sample=0.1,
    )
    assert step.accepted_counts.item() == 0
    assert step.next_tokens.item() == 1


def test_invalid_unused_distributions_are_ignored():
    step = verify(
        [0, -1], [[1.0, 0.0], [float("nan"), float("nan")]],
        [[1.0, 0.0], [float("nan"), float("nan")], [float("nan"), float("nan")]],
        valid=[True, False], budget=1,
    )
    assert step.accepted_counts.item() == 1 and not step.emit_next.item()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"valid": [False, True]}, "contiguous prefix"),
        ({"budget": 0}, "zero-budget"),
        ({"q": [[0.6, 0.6], [0.5, 0.5]]}, "normalized"),
        ({"p": [[0.6, 0.6], [0.5, 0.5], [0.5, 0.5]]}, "normalized"),
        ({"q": [[0.0, 1.0], [0.5, 0.5]]}, "positive draft"),
        ({"accept": [1.0, 0.0]}, r"\[0, 1\)"),
        ({"sample": -0.1}, r"\[0, 1\)"),
    ],
)
def test_rejects_inconsistent_verification_inputs(kwargs, match):
    arguments = dict(tokens=[0, 0], q=[[0.5, 0.5], [0.5, 0.5]], p=[[0.5, 0.5]] * 3)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        verify(**arguments)


def test_generator_is_reproducible():
    tokens = torch.zeros(16, 2, dtype=torch.long)
    q = torch.tensor([0.8, 0.2]).expand(16, 2, 2)
    p = torch.tensor([0.3, 0.7]).expand(16, 3, 2)
    results = [
        verify_on_policy_block(
            tokens, torch.ones_like(tokens, dtype=torch.bool), q, p,
            torch.full((16,), 10),
            generator=torch.Generator().manual_seed(42),
        )
        for _ in range(2)
    ]
    for left, right in zip(*results):
        torch.testing.assert_close(left, right)
