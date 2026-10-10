"""Packed replay must preserve each proposal's q, loss, and parameter gradients."""

import copy
import unittest

import torch

from specforge.on_policy.objective import block_loss, block_losses
from specforge.on_policy.replay import packed_replay_inputs, replay_inputs
from tests.test_on_policy.test_replay import tiny_model


def replay_blocks():
    return [
        {
            "context_length": context,
            "anchor": i + 1,
            "proposal": [i + 3, i + 4, i + 5],
            # Vary EOS/budget tails independently of the original rejection.
            "valid_mask": [j < valid for j in range(3)],
            "accepted_count": i % 3,
            "sampling": {"temperature": temperature},
        }
        for i, (context, valid, temperature) in enumerate(
            [(2, 3, 0.8), (5, 2, 1.0), (9, 1, 0.5), (12, 3, 1.3), (17, 2, 0.9)]
        )
    ]


class PackedReplayTests(unittest.TestCase):
    def setUp(self):
        # Tiny model helpers seed torch; keep these new cases isolated from
        # older tests whose randomized fixtures share the process RNG.
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)

    @torch.no_grad()
    def test_64_seven_token_blocks_match_scalar_and_mask_query_edge_tiles(self):
        model = tiny_model(layer_types=["full_attention"] * 2)
        model.draft_model.block_size = model.draft_model.config.block_size = 7
        blocks = [
            {
                "context_length": 133 + 4 * i,
                "anchor": 1 + i % 14,
                "proposal": [(i + j) % 14 for j in range(7)],
                "valid_mask": [True] * 7,
                "sampling": {"temperature": 1.0},
            }
            for i in range(64)
        ]
        features = torch.randn(385, 32)
        expected = torch.stack(
            [
                model.draft_probabilities(block, features[: block["context_length"]])
                for block in blocks
            ]
        )
        actual = model.packed_draft_probabilities(blocks, features)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        _, _, masks = packed_replay_inputs(
            blocks,
            7,
            15,
            "cpu",
            ["full_attention"],
            attention_backend="flex_attention",
        )
        mask = masks["full_attention"].mask_mod
        qi = torch.arange(512)[:, None]
        ki = torch.arange(896)[None]
        allowed = mask(0, 0, qi, ki)
        self.assertFalse(allowed[448:].any())
        self.assertFalse(allowed[:, 833:].any())
        self.assertTrue(allowed[441:448, 826:833].all())

    def test_length_dependent_rope_cannot_silently_change_earlier_blocks(self):
        model = tiny_model()
        for rope_type in ("dynamic", "longrope"):
            model.draft_model.rotary_emb.rope_type = rope_type
            with self.assertRaisesRegex(ValueError, "replay_blocks_per_forward=1"):
                model.packed_draft_probabilities(replay_blocks(), torch.randn(17, 32))

    def test_packed_sdpa_matches_separate_q_loss_and_all_gradients(self):
        for head in ("vanilla", "gated", "rnn"):
            for mode in ("gqa", "mha"):
                for layout in (
                    ["full_attention", "full_attention"],
                    ["full_attention", "sliding_attention"],
                ):
                    with self.subTest(head=head, mode=mode, layout=layout):
                        self.check_equivalence(head, mode, layout)

    def check_equivalence(self, head, mode, layout):
        scalar = tiny_model(head, attention_mode=mode, layer_types=layout)
        packed = copy.deepcopy(scalar)
        blocks = replay_blocks()
        features = torch.randn(17, 32)
        p = torch.randn(len(blocks), 3, 16).softmax(-1).requires_grad_()
        with torch.no_grad():
            recorded = torch.stack(
                [
                    scalar.draft_probabilities(b, features[: b["context_length"]])
                    for b in blocks
                ]
            )
        # A nonuniform short final group must still have the same total weight.
        losses = [
            scalar(b, features[: b["context_length"]], p[i], recorded[i])[0]
            for i, b in enumerate(blocks)
        ]
        reference = torch.stack(losses).sum() / len(blocks)
        reference.backward()
        fc_lengths = []
        handle = packed.draft_model.fc.register_forward_pre_hook(
            lambda _module, args: fc_lengths.append(args[0].shape[1])
        )
        try:
            actual_q = packed.packed_draft_probabilities(blocks, features)
            torch.testing.assert_close(actual_q, recorded, atol=2e-7, rtol=2e-5)
            actual, parity = packed(blocks, features, p, recorded)
            (actual / len(blocks)).backward()
        finally:
            handle.remove()
        self.assertEqual(fc_lengths, [17, 17])  # one shared projection per forward
        torch.testing.assert_close(actual / len(blocks), reference)
        self.assertLess(parity.item(), 2e-6)
        self.assertIsNone(p.grad)
        for name, parameter in packed.named_parameters():
            expected = dict(scalar.named_parameters())[name]
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertIsNotNone(expected.grad, name)
                torch.testing.assert_close(
                    parameter.grad, expected.grad, atol=2e-6, rtol=2e-4, msg=name
                )
            else:
                self.assertIsNone(parameter.grad, name)

    def test_packed_masks_match_each_original_block_and_padded_flex_tiles(self):
        blocks = replay_blocks()
        blocks[-1]["context_length"] = 133
        kinds = ["full_attention", "sliding_attention"]
        ids, positions, dense = packed_replay_inputs(blocks, 3, 15, "cpu", kinds, 4)
        _, _, flex = packed_replay_inputs(
            blocks, 3, 15, "cpu", kinds, 4, attention_backend="flex_attention"
        )
        self.assertEqual(ids[0, ::3].tolist(), [b["anchor"] for b in blocks])
        self.assertEqual(positions[0, :133].tolist(), list(range(133)))
        for i, block in enumerate(blocks):
            context = block["context_length"]
            _, old_positions, original = replay_inputs(
                context, 3, block["anchor"], 15, "cpu", kinds, 4
            )
            torch.testing.assert_close(
                positions[:, 133 + i * 3 : 133 + (i + 1) * 3], old_positions[:, -3:]
            )
            for kind in kinds:
                rows = dense[kind][0, 0, i * 3 : (i + 1) * 3]
                visible = torch.cat(
                    (rows[:, :context], rows[:, 133 + i * 3 : 133 + (i + 1) * 3]), -1
                )
                torch.testing.assert_close(visible, original[kind][0, 0])
                self.assertFalse(rows[:, context:133].any())
                self.assertFalse(rows[:, 133 : 133 + i * 3].any())
                self.assertFalse(rows[:, 133 + (i + 1) * 3 :].any())
        qi, ki = torch.arange(128)[:, None], torch.arange(256)[None]
        for kind in kinds:
            actual = flex[kind].mask_mod(0, 0, qi, ki)
            torch.testing.assert_close(actual[:15, :148], dense[kind][0, 0])
            self.assertFalse(actual[15:].any())
            self.assertFalse(actual[:, 148:].any())

    def test_earlier_block_cannot_see_future_context_or_other_proposals(self):
        model = tiny_model(layer_types=["full_attention"] * 2)
        blocks = replay_blocks()
        features = torch.randn(17, 32)
        original = model.packed_draft_probabilities(blocks, features)
        changed = copy.deepcopy(blocks)
        changed[-1]["anchor"] = 14
        changed[-1]["proposal"] = [12, 13, 14]
        future = features.clone()
        future[2:] += torch.randn_like(future[2:]) * 10
        modified = model.packed_draft_probabilities(changed, future)
        torch.testing.assert_close(modified[0], original[0], atol=0, rtol=0)

    def test_packed_parity_matches_scalar_valid_position_check(self):
        model = tiny_model()
        blocks = replay_blocks()
        features = torch.randn(17, 32)
        with torch.no_grad():
            recorded = model.packed_draft_probabilities(blocks, features)
        p = recorded.clone()
        # Invalid suffix drift is excluded, as in scalar replay.
        recorded[2, 2, 0] += 0.2
        _, parity = model(blocks, features, p, recorded)
        torch.testing.assert_close(parity, torch.tensor(0.0))
        recorded[1, 1, 0] += 0.2  # Valid drift must still trigger the gate.
        _, parity = model(blocks, features, p, recorded)
        torch.testing.assert_close(parity, torch.tensor(0.1))
        scalar_parities = [
            model(block, features[: block["context_length"]], p[i], recorded[i])[1]
            for i, block in enumerate(blocks)
        ]
        torch.testing.assert_close(parity, torch.stack(scalar_parities).max())

    def test_batched_loss_preserves_masked_nans_and_independent_denominators(self):
        p = torch.randn(3, 3, 16).softmax(-1)
        q = torch.randn(3, 3, 16).softmax(-1)
        mask = torch.tensor([[True, False, False], [True, True, False], [True] * 3])
        p[~mask] = q[~mask] = float("nan")
        q.requires_grad_()
        expected = torch.stack([block_loss(a, b, m) for a, b, m in zip(p, q, mask)])
        actual = block_losses(p, q, mask)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertFalse(q.grad[~mask].any())
        for invalid in (
            [[False] * 3, [True] * 3, [True] * 3],
            [[True, False, True], [True] * 3, [True] * 3],
        ):
            with self.assertRaisesRegex(ValueError, "nonempty prefix"):
                block_losses(p, q, torch.tensor(invalid))


if __name__ == "__main__":
    unittest.main()
