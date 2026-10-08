import unittest

import torch

from specforge.on_policy.objective import block_loss, candidate_mask, draft_distribution


class ObjectiveTests(unittest.TestCase):
    def test_closed_form_and_frozen_teacher(self):
        p = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        q = torch.tensor([[0.8, 0.2], [0.5, 0.5], [0.4, 0.6]], requires_grad=True)
        loss = block_loss(p, q, torch.ones(3, dtype=torch.bool))
        self.assertAlmostEqual(loss.item(), 1 - (0.8 + 0.4 + 0.16) / 3, places=6)
        loss.backward()
        self.assertIsNone(p.grad)
        # Even the final, possibly rejected candidate is supervised.
        self.assertGreater(q.grad[-1].abs().sum().item(), 0)

    def test_exact_match_disjoint_and_padding_nan(self):
        p = torch.tensor([[1.0, 0.0], [float("nan"), float("nan")]])
        q = torch.tensor([[1.0, 0.0], [float("nan"), float("nan")]], requires_grad=True)
        loss = block_loss(p, q, torch.tensor([True, False]))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertEqual(block_loss(p[:1], 1 - p[:1], torch.tensor([True])).item(), 1)

    def test_empty_and_holey_masks_rejected(self):
        probabilities = torch.full((3, 2), 0.5)
        for mask in ([False] * 3, [True, False, True]):
            with self.assertRaises(ValueError):
                block_loss(probabilities, probabilities, torch.tensor(mask))

    def test_gradient_matches_finite_difference(self):
        logits = torch.tensor([[0.3, -0.4], [0.6, -0.2]], requires_grad=True)
        p = torch.tensor([[0.9, 0.1], [0.8, 0.2]])

        def loss(z):
            return block_loss(p, draft_distribution(z, 0.8), torch.tensor([True, True]))

        loss(logits).backward()
        for row in range(2):
            for column in range(2):
                delta = torch.zeros_like(logits)
                delta[row, column] = 0.001
                numerical = (
                    loss(logits.detach() + delta) - loss(logits.detach() - delta)
                ) / 0.002
                self.assertAlmostEqual(
                    numerical.item(), logits.grad[row, column].item(), places=4
                )

    def test_masks_ignore_rejection_and_include_first_eos(self):
        self.assertEqual(
            candidate_mask([4, 5, 9, 6], 4, {9}), [True, True, True, False]
        )
        self.assertEqual(
            candidate_mask([4, 5, 9, 6], 2, {9}), [True, True, False, False]
        )
        self.assertEqual(candidate_mask([4, 5], 0, set()), [False, False])

    def test_draft_distribution_has_full_support(self):
        q = draft_distribution(torch.tensor([[2.0, 1.0, 0.0]]), 0.7)
        self.assertTrue((q > 0).all())
        with self.assertRaises(ValueError):
            draft_distribution(q, 0)
