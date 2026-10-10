import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.on_policy.replay import DSparkReplayModel, replay_inputs


def tiny_model(head="vanilla", *, attention_mode="gqa", layer_types=None):
    layer_types = layer_types or ["full_attention"]
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=4 if attention_mode == "mha" else 2,
        head_dim=4,
        layer_types=layer_types,
        use_sliding_window="sliding_attention" in layer_types,
        sliding_window=4 if "sliding_attention" in layer_types else None,
        attention_dropout=0.0,
    )
    config.block_size = 3
    config.num_target_layers = 4
    config.dflash_config = {
        "attention_mode": attention_mode,
        "target_layer_ids": [0, 1],
        "mask_token_id": 15,
        "projector_type": "dspark",
        "markov_rank": 4,
        "markov_head_type": head,
        "enable_confidence_head": False,
    }
    config._attn_implementation = "sdpa"
    torch.manual_seed(17)
    draft = DSparkDraftModel(config).eval()
    return DSparkReplayModel(
        draft,
        SimpleNamespace(
            lm_head=nn.Linear(16, 16, bias=False), embed_tokens=nn.Embedding(16, 16)
        ),
    )


class ReplayTests(unittest.TestCase):
    def test_flex_mask_matches_dense_including_padding_boundaries(self):
        for length in (3, 133):
            _, _, dense = replay_inputs(
                length, 7, 2, 15, "cpu", ["full_attention", "sliding_attention"], 3
            )
            _, _, flex = replay_inputs(
                length,
                7,
                2,
                15,
                "cpu",
                ["full_attention", "sliding_attention"],
                3,
                attention_backend="flex_attention",
            )
            qi = torch.arange(7)[:, None]
            ki = torch.arange(length + 7)[None, :]
            for kind in dense:
                actual = flex[kind].mask_mod(0, 0, qi, ki)
                torch.testing.assert_close(actual, dense[kind][0, 0])
                self.assertFalse(
                    flex[kind].mask_mod(
                        0, 0, torch.tensor(128), torch.tensor(length + 128)
                    )
                )

    def test_full_and_sliding_masks(self):
        ids, positions, masks = replay_inputs(
            3, 3, 7, 15, "cpu", ["full_attention", "sliding_attention"], 3
        )
        self.assertEqual(ids.tolist(), [[7, 15, 15]])
        self.assertEqual(positions.tolist(), [list(range(6))])
        self.assertTrue(masks["full_attention"].all())
        self.assertEqual(
            masks["sliding_attention"][0, 0].tolist(),
            [
                [False, True, True, True, False, False],
                [False, False, True, True, True, False],
                [False, False, False, True, True, True],
            ],
        )

    def test_real_dspark_all_markov_heads_replay_rejected_suffix(self):
        for head in ("vanilla", "gated", "rnn"):
            with self.subTest(head=head):
                model = tiny_model(head)
                block = {
                    "context_length": 4,
                    "anchor": 2,
                    "proposal": [3, 4, 5],
                    "accepted_count": 0,
                    "valid_mask": [True] * 3,
                    "sampling": {"temperature": 0.8},
                }
                previous = []
                original = model.draft_model.apply_logits_head

                def record(*args, **kwargs):
                    previous.append(kwargs["prev_token_ids"].tolist())
                    return original(*args, **kwargs)

                model.draft_model.apply_logits_head = record
                target = torch.softmax(torch.randn(3, 16), -1)
                features = torch.randn(4, 32)
                loss, _ = model(block, features, target, target)
                block["accepted_count"] = 3
                alternate, _ = model(block, features, target, target)
                torch.testing.assert_close(loss, alternate)
                loss.backward()
                self.assertEqual(previous[0], [[2, 3, 4]])
                self.assertGreater(model.draft_model.fc.weight.grad.abs().sum(), 0)
                self.assertGreater(
                    model.draft_model.markov_head.markov_w2.weight.grad.abs().sum(), 0
                )
                self.assertIsNone(model.lm_head.weight.grad)
                self.assertIsNone(model.embed_tokens.weight.grad)

    def test_rnn_teacher_forcing_matches_recorded_sampling_walk(self):
        model = tiny_model("rnn").draft_model.markov_head
        base, hidden = torch.randn(1, 3, 16), torch.randn(1, 3, 16)
        tokens, logits = model.sample_block_tokens(
            base,
            first_prev_token_ids=torch.tensor([2]),
            hidden_states=hidden,
            temperature=0.7,
        )
        previous = torch.cat([torch.tensor([[2]]), tokens[:, :-1]], dim=-1)
        replay = model.apply_block_logits(
            base, token_ids=previous, hidden_states=hidden
        )
        torch.testing.assert_close(logits, replay)
