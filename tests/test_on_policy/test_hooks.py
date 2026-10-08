import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn

from specforge.on_policy.data import load_trace
from specforge.on_policy.rollout import RolloutPool, _idle_rpc
from specforge.on_policy.sglang_hooks import (
    _expected_serving_weights,
    after_verify_accept,
    prefill,
    scheduler_rpc,
)


class ServingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        self.confidence_head = None
        self._stacked_ctx_kv_cache = "stale"
        self._fused_kv_write_cache = "stale"

    def load_weights(self, items):
        self.load_state_dict(dict(items), strict=True)


class HookTests(unittest.TestCase):
    def test_capture_keeps_rejected_tokens_and_committed_context(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"SPECFORGE_ON_POLICY_ROOT": directory}),
        ):
            worker = SimpleNamespace(_specforge_weight_version=0)
            batch = SimpleNamespace(
                reqs=[SimpleNamespace(rid="r", origin_input_ids=[1, 2])],
                prefix_lens=[0],
            )
            prefill(worker, batch, torch.ones(2, 4), torch.tensor([3]))
            block = {
                "weight_version": 0,
                "context_length": 2,
                "context_ids": [1, 2],
                "anchor": 3,
                "proposal": [4, 5, 6],
                "valid_mask": [True] * 3,
                "sampling": {"ignore_eos": False, "stop_token_ids": [15]},
            }
            after_verify_accept(
                worker,
                block,
                torch.tensor([[3, 4, 5, 6]]),
                SimpleNamespace(hidden_states=torch.full((4, 4), 2.0)),
                SimpleNamespace(
                    correct_len=torch.tensor([1]),
                    commit_lens=torch.tensor([2]),
                    bonus=torch.tensor([9]),
                ),
            )
            self.assertEqual(worker._specforge_context, [1, 2, 3, 4])
            worker._specforge_tensors = {
                "p_0": torch.ones(3, 1),
                "q_0": torch.ones(3, 1),
            }
            scheduler = SimpleNamespace(draft_worker=worker, is_fully_idle=lambda: True)
            scheduler_rpc(scheduler, "drain", 0, request_id="r", output_ids=[3, 4, 9])
            trace, tensors = load_trace(directory, "r", 0)
            self.assertEqual(trace["blocks"][0]["proposal"], [4, 5, 6])
            self.assertEqual(trace["blocks"][0]["valid_mask"], [True] * 3)
            self.assertEqual(trace["sequence_ids"], [1, 2, 3, 4, 9])
            self.assertEqual(trace["blocks"][0]["actual_accepted_count"], 1)
            self.assertEqual(tensors["context_hidden"].shape, (4, 4))

    def test_full_sync_clears_derived_weights_and_kv_and_requires_next_version(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "SPECFORGE_ON_POLICY_ROOT": directory,
                    "SPECFORGE_ON_POLICY_WORKER": "0",
                    "SPECFORGE_ON_POLICY_COLOCATED": "1",
                },
            ),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache") as release,
        ):
            model = ServingModel()
            state = {"fc.weight": torch.full((2, 2), 7.0)}
            path = Path(directory) / "weights" / "00000000"
            path.mkdir(parents=True)
            save_file(state, str(path / "model.safetensors"))
            worker = SimpleNamespace(draft_model=model)
            flushed = []
            scheduler = SimpleNamespace(
                draft_worker=worker,
                is_fully_idle=lambda: True,
                flush_cache=lambda: flushed.append(True) or True,
            )
            scheduler_rpc(scheduler, "sync", 0)
            self.assertEqual(worker._specforge_weight_version, 0)
            torch.testing.assert_close(model.fc.weight, state["fc.weight"])
            self.assertIs(model._stacked_ctx_kv_cache, False)
            self.assertIsNone(model._fused_kv_write_cache)
            self.assertEqual(flushed, [True])
            release.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "nonsequential"):
                scheduler_rpc(scheduler, "sync", 0)

    def test_overlap_block_after_eos_anchor_is_retained_but_not_supervised(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"SPECFORGE_ON_POLICY_ROOT": directory}),
        ):
            worker = SimpleNamespace(_specforge_weight_version=0)
            batch = SimpleNamespace(
                reqs=[SimpleNamespace(rid="eos", origin_input_ids=[1, 2])],
                prefix_lens=[0],
            )
            prefill(worker, batch, torch.ones(2, 4), torch.tensor([3]))
            block = {
                "weight_version": 0,
                "context_length": 2,
                "context_ids": [1, 2],
                "anchor": 3,
                "proposal": [4, 5],
                "valid_mask": [True, True],
                "sampling": {"ignore_eos": False, "stop_token_ids": [3]},
            }
            after_verify_accept(
                worker,
                block,
                torch.tensor([[3, 4, 5]]),
                SimpleNamespace(hidden_states=torch.ones(3, 4)),
                SimpleNamespace(
                    correct_len=torch.tensor([1]),
                    commit_lens=torch.tensor([2]),
                    bonus=torch.tensor([9]),
                ),
            )
            worker._specforge_tensors = {
                "p_0": torch.ones(2, 1),
                "q_0": torch.ones(2, 1),
            }
            scheduler_rpc(
                SimpleNamespace(draft_worker=worker, is_fully_idle=lambda: True),
                "drain",
                0,
                request_id="eos",
                output_ids=[3],
            )
            trace, _ = load_trace(directory, "eos", 0)
            self.assertEqual(trace["blocks"][0]["proposal"], [4, 5])
            self.assertEqual(trace["blocks"][0]["valid_mask"], [False, False])

    def test_missing_and_extra_weights_fail_before_loading(self):
        model = ServingModel()
        for state in ({}, {"fc.weight": torch.ones(2, 2), "unknown": torch.ones(1)}):
            with self.assertRaises(ValueError):
                _expected_serving_weights(model, state)

    def test_qkv_and_gate_up_packing_matches_serving_layout(self):
        model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = nn.Linear(4, 8, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = nn.Linear(4, 6, bias=False)
        model.layers = nn.ModuleList([layer])
        state = {
            "layers.0.self_attn.k_proj.weight": torch.full((2, 4), 2.0),
            "layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0),
            "layers.0.self_attn.v_proj.weight": torch.full((2, 4), 3.0),
            "layers.0.mlp.up_proj.weight": torch.full((3, 4), 5.0),
            "layers.0.mlp.gate_proj.weight": torch.full((3, 4), 4.0),
        }
        expected = _expected_serving_weights(model, state)
        self.assertEqual(
            expected["layers.0.self_attn.qkv_proj.weight"][:, 0].tolist(),
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        self.assertEqual(
            expected["layers.0.mlp.gate_up_proj.weight"][:, 0].tolist(),
            [4.0, 4.0, 4.0, 5.0, 5.0, 5.0],
        )

    def test_busy_engine_cannot_sync(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"SPECFORGE_ON_POLICY_ROOT": directory}),
        ):
            scheduler = SimpleNamespace(
                draft_worker=ServingModel(), is_fully_idle=lambda: False
            )
            with self.assertRaisesRegex(RuntimeError, "idle"):
                scheduler_rpc(scheduler, "sync", 0)

    def test_partial_worker_ack_poisons_pool(self):
        pool = RolloutPool.__new__(RolloutPool)
        pool.version, pool.poisoned = 0, False
        pool.workers = [(SimpleNamespace(send_bytes=lambda _: None), None)] * 2
        replies = iter([{"version": 1}, {"version": 0}])
        pool._receive = lambda _: next(replies)
        with self.assertRaisesRegex(RuntimeError, "wrong version"):
            pool.synchronize(1)
        self.assertTrue(pool.poisoned)
        self.assertEqual(pool.version, 0)
        with self.assertRaises(RuntimeError):
            pool.generate([])

    def test_idle_retry_does_not_retry_weight_loading_errors(self):
        calls = []

        def rpc(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise AssertionError("on-policy RPC requires an idle engine")

        with patch("time.sleep"):
            _idle_rpc(
                SimpleNamespace(collective_rpc=rpc), 1, operation="sync", version=1
            )
        self.assertEqual(len(calls), 2)

        def broken(*args, **kwargs):
            raise AssertionError("draft synchronization did not load fc.weight")

        with self.assertRaisesRegex(AssertionError, "did not load"):
            _idle_rpc(
                SimpleNamespace(collective_rpc=broken), 1, operation="sync", version=1
            )
