"""Regression for BF16 frequency rounding before FSDP's FP32 buffer cast."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.on_policy.data import atomic_json, trace_stem
from specforge.on_policy.diagnose import main, tv_summary
from specforge.on_policy.replay import DSparkReplayModel, create_replay_draft


def config():
    # Preserve the server's real RoPE geometry without allocating an 8B model.
    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        vocab_size=16,
        layer_types=["full_attention"],
        rope_theta=1_000_000,
        architectures=["DSparkDraftModel"],
        attention_dropout=0.0,
    )
    cfg.block_size = 3
    cfg.num_target_layers = 4
    cfg.dflash_config = {
        "target_layer_ids": [0, 1],
        "mask_token_id": 15,
        "projector_type": "dspark",
        "markov_rank": 4,
        "enable_confidence_head": False,
    }
    cfg._attn_implementation = "sdpa"
    return cfg


def reference_frequencies():
    return 1.0 / (1_000_000 ** (torch.arange(0, 128, 2).float() / 128))


class RopePrecisionTests(unittest.TestCase):
    def test_legacy_cast_back_cannot_recover_frequencies(self):
        draft = AutoDraftModel.from_config(config(), torch_dtype=torch.bfloat16)
        draft.rotary_emb.float()
        phase_error = (draft.rotary_emb.inv_freq - reference_frequencies()).abs() * 4096
        self.assertGreater(phase_error.max().item(), 4.0)
        self.assertFalse(any("inv_freq" in key for key in draft.state_dict()))

    def test_replay_factory_preserves_long_position_angles(self):
        draft = create_replay_draft(config()).to(device="cpu")
        self.assertEqual(draft.fc.weight.dtype, torch.bfloat16)
        self.assertEqual(draft.rotary_emb.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(
            draft.rotary_emb.inv_freq, reference_frequencies(), atol=0, rtol=0
        )
        positions = torch.tensor([[128, 512, 2048, 4096, 8192]])
        cos, sin = draft.rotary_emb(
            torch.zeros(1, 5, 16, dtype=torch.bfloat16), positions
        )
        angles = positions[..., None].float() * reference_frequencies()
        angles = torch.cat((angles, angles), dim=-1)
        torch.testing.assert_close(cos, angles.cos().bfloat16(), atol=0, rtol=0)
        torch.testing.assert_close(sin, angles.sin().bfloat16(), atol=0, rtol=0)

    def test_diagnostic_distinguishes_invalid_suffix(self):
        rollout = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
        replay = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
        report = tv_summary(replay, rollout, [True, False])
        self.assertEqual(report["max_tv"], 1.0)
        self.assertEqual(report["max_valid_tv"], 0.0)
        self.assertEqual(report["worst_position"], 1)

    def test_saved_trajectory_audit_compares_rope_without_writing(self):
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            root = Path(directory)
            cfg = config()
            cfg.tie_word_embeddings = False
            draft = create_replay_draft(cfg)
            parts = TargetEmbeddingsAndHead(cfg).bfloat16()
            model = DSparkReplayModel(draft, parts).eval()
            context = torch.randn(512, 32, dtype=torch.bfloat16)
            block = {
                "context_length": 512,
                "context_ids": [1] * 512,
                "anchor": 2,
                "proposal": [3, 4, 5],
                "valid_mask": [True] * 3,
                "sampling": {"temperature": 1.0},
                "weight_version": 0,
                "accepted_count": 0,
                "correction_or_bonus": 6,
            }
            q = model.draft_probabilities(block, context)
            weights, target = root / "weights" / "00000000", root / "target"
            draft.config.save_pretrained(weights)
            cfg.save_pretrained(target)
            save_file(draft.state_dict(), str(weights / "model.safetensors"))
            save_file(
                {
                    "model.embed_tokens.weight": parts.embed_tokens.weight,
                    "lm_head.weight": parts.lm_head.weight,
                },
                str(target / "model.safetensors"),
            )
            atomic_json(
                root / "run.json",
                {
                    "model": {
                        "target_model_path": str(target),
                        "draft_checkpoint_path": str(weights),
                    },
                    "data": {"train_data_path": "unused"},
                    "training": {"output_dir": str(root)},
                },
            )
            atomic_json(root / "batches" / "00000000.json", ["sample"])
            stem = trace_stem(root, "sample")
            atomic_json(
                stem.with_suffix(".json"),
                {
                    "request_id": "sample",
                    "weight_version": 0,
                    "prompt_ids": [1] * 512,
                    "initial_token": 2,
                    "blocks": [block],
                    "output_ids": [2, 6],
                    "sequence_ids": [1] * 512 + [2, 6],
                },
            )
            save_file(
                {"context_hidden": context, "p_0": q.clone(), "q_0": q},
                str(stem.with_suffix(".safetensors")),
            )
            before = {
                path: path.read_bytes() for path in root.rglob("*") if path.is_file()
            }
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--run-dir", str(root), "--device", "cpu"]), 0)
            rows = [
                json.loads(line)
                for line in output.getvalue().splitlines()
                if line.startswith("{")
            ]
            self.assertEqual(rows[-1]["fp32_rope"]["max_tv"], 0.0)
            self.assertGreater(rows[-1]["legacy_rope"]["max_tv"], 0.0)
            after = {
                path: path.read_bytes() for path in root.rglob("*") if path.is_file()
            }
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
