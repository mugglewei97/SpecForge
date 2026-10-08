import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from specforge.on_policy.__main__ import validate_topology
from specforge.on_policy.config import OnPolicyConfig, TrainingConfig
from specforge.on_policy.data import trace_stem
from specforge.on_policy.schedule import planned_steps, prompt_batches
from specforge.on_policy.trainer import retire_step_artifacts


class ServerRecipeTests(unittest.TestCase):
    def test_six_complete_epochs_preserve_tail_and_sample_identity(self):
        training = TrainingConfig(output_dir="unused", num_epochs=6, batch_size=4)
        prompts = [{"sample_id": i} for i in range(9)]
        batches = list(prompt_batches(prompts, training))
        self.assertEqual(planned_steps(len(prompts), training), 18)
        self.assertEqual(len(batches), 18)
        self.assertEqual(batches, list(prompt_batches(prompts, training)))
        for epoch in range(1, 7):
            epoch_batches = [batch for e, batch in batches if e == epoch]
            self.assertEqual([len(batch) for batch in epoch_batches], [4, 4, 1])
            self.assertEqual(
                sorted(p["sample_id"] for batch in epoch_batches for p in batch),
                list(range(9)),
            )
        self.assertNotEqual(batches[0][1], batches[3][1])

    def test_step_budget_compatibility_and_ambiguous_budget_rejection(self):
        training = TrainingConfig(output_dir="unused", max_steps=3, batch_size=4)
        batches = list(prompt_batches([0, 1, 2], training))
        self.assertEqual([len(batch) for _, batch in batches], [4, 4, 4])
        self.assertEqual(planned_steps(3, training), 3)
        self.assertEqual(TrainingConfig(output_dir="unused").max_steps, 100)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            TrainingConfig(output_dir="unused", max_steps=10, num_epochs=6)

    def test_server_recipe_preserves_original_budget_and_caps_whole_sequence(self):
        cfg = OnPolicyConfig.from_file("examples/on_policy/qwen3-8b-dspark-tv.yaml")
        self.assertEqual(cfg.training.num_epochs, 6)
        self.assertIsNone(cfg.training.max_steps)
        self.assertEqual(cfg.training.batch_size, 8 * 4)
        self.assertEqual(cfg.training.learning_rate, 6e-4)
        self.assertEqual(cfg.training.warmup_ratio, 0.04)
        self.assertEqual(cfg.training.max_grad_norm, 1.0)
        self.assertEqual(cfg.training.attention_backend, "flex_attention")
        self.assertEqual(cfg.training.log_interval, 50)
        self.assertEqual(cfg.training.save_interval, 2000)
        self.assertEqual(cfg.rollout.mem_fraction_static, 0.3)
        self.assertEqual(cfg.rollout.placement, "colocated")
        self.assertEqual(cfg.rollout.cuda_devices, list(range(8)))
        validate_topology(cfg, "0,1,2,3,4,5,6,7", 8, 8)
        self.assertEqual(cfg.data.chat_template, "qwen")
        self.assertEqual(cfg.data.chat_template_kwargs, {})
        self.assertEqual(cfg.sequence_limit, 3072)
        self.assertEqual(cfg.data.prompt_limit, 3070)
        for length in (10, 1000, 3070):
            sampling = cfg.sampling_for_prompt(length)
            self.assertEqual(length + sampling["max_new_tokens"], 3072)
        with self.assertRaisesRegex(ValueError, "no budget"):
            cfg.sampling_for_prompt(3071)
        # Resolving a request must not mutate later requests' generation budget.
        self.assertEqual(cfg.sampling.max_new_tokens, 3072)
        draft = json.loads(Path(cfg.model.draft_model_config).read_text())
        self.assertEqual(draft["block_size"], 7)

    def test_topology_rejects_partial_colocation_and_preserves_dedicated_mode(self):
        cfg = OnPolicyConfig.from_file("examples/on_policy/qwen3-8b-dspark-tv.yaml")
        for visible, world, local in (
            ("1,2,3,4,5,6,7", 7, 7),
            ("0,1,2,3,4,5,6,7", 8, 7),
            ("0,0,2,3,4,5,6,7", 8, 8),
            ("7,6,5,4,3,2,1,0", 8, 8),
        ):
            with self.subTest(visible=visible, world=world, local=local):
                with self.assertRaises(ValueError):
                    validate_topology(cfg, visible, world, local)
        cfg = OnPolicyConfig.from_file("examples/on_policy/qwen3-4b-dspark-tv.yaml")
        self.assertEqual(cfg.rollout.placement, "dedicated")
        validate_topology(cfg, "1,2", 2, 2)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_topology(cfg, "0,1", 2, 2)
        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_topology(cfg, "", 1, 1)

    def test_server_script_launches_eight_ranks_on_all_eight_devices(self):
        # Capture the real shell argv/environment without starting CUDA or torchrun.
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / "torchrun"
            stub.write_text(
                '#!/bin/bash\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@"\n'
            )
            stub.chmod(0o700)
            result = subprocess.run(
                ["bash", "scripts/train_dspark_on_policy_8gpu.sh"],
                env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"]},
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        arguments = result.stdout.splitlines()
        self.assertEqual(arguments[0], "0,1,2,3,4,5,6,7")
        self.assertIn("--nproc_per_node=8", arguments)
        self.assertIn("specforge.on_policy", arguments)

    def test_post_ack_cleanup_preserves_proposals_initial_interval_and_latest(self):
        training = TrainingConfig(
            output_dir="unused", save_interval=2, retain_replay_tensors=False
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for version in range(4):
                path = root / "weights" / f"{version:08d}"
                path.mkdir(parents=True)
                for name in ("model.safetensors", "config.json"):
                    (path / name).write_text("owned artifact")
            stem = trace_stem(root, "../../request")
            stem.parent.mkdir()
            stem.with_suffix(".json").write_text('{"proposal": [1, 2, 3]}')
            stem.with_suffix(".safetensors").write_bytes(b"large replay tensors")
            retire_step_artifacts(root, 2, training, ["../../request"])
            self.assertFalse((root / "weights" / "00000001").exists())
            self.assertFalse(stem.with_suffix(".safetensors").exists())
            self.assertEqual(
                json.loads(stem.with_suffix(".json").read_text())["proposal"], [1, 2, 3]
            )
            retire_step_artifacts(root, 3, training, [])
            for version in (0, 2, 3):
                self.assertTrue(
                    (root / "weights" / f"{version:08d}" / "model.safetensors").exists()
                )


if __name__ == "__main__":
    unittest.main()
