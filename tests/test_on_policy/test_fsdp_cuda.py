"""Two-GPU FSDP gate; run on H200 before the full SGLang smoke recipe."""

import os
import copy
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from specforge.on_policy.trainer import (
    _leader_call,
    _prepare_rollout_phase,
    train_effective_batch,
)
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from tests.test_on_policy.test_precision import check_bf16_replay_precision
from tests.test_on_policy.test_training import tensors, traces


class CudaReplay(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, device=device))

    def forward(self, block, *_args):
        blocks = block if isinstance(block, list) else [block]
        coefficient = sum(b["coefficient"] for b in blocks)
        return self.weight.square().sum() * coefficient, self.weight.new_zeros(())


class SGDStep:
    def __init__(self, module):
        self.inner = torch.optim.SGD(module.parameters(), lr=0.1)
        self.calls = 0

    def step(self):
        self.calls += 1
        self.inner.step()
        self.inner.zero_grad(set_to_none=True)


def _worker(rank, directory):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{directory}/rdzv", rank=rank, world_size=2
    )
    control_group = dist.new_group(backend="gloo")
    try:
        check_bf16_replay_precision(torch.device("cuda", rank))
        check_bf16_replay_precision(torch.device("cuda", rank), packed=True)
        model = CudaReplay(torch.device("cuda", rank))
        backend = FSDPTrainingBackend(
            ParallelConfig(
                world_size=2, sharding_strategy="FULL_SHARD", param_dtype=torch.float32
            ),
            optimizer_factory=SGDStep,
        )
        backend.prepare_model(model, optimizer_target=model)
        _prepare_rollout_phase(control_group)
        _leader_call(
            lambda: torch.ones(8, device="cuda").sum().item(), rank, control_group
        )
        train_effective_batch(backend, traces(), tensors, rank, 2, 0.02)
        state = backend._module_state_dict()
        if rank == 0:
            torch.testing.assert_close(state["weight"], torch.full((8,), 0.3))
        assert backend.optimizer.calls == 1
        _check_packed_dspark_update(rank)
    finally:
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


def _check_packed_dspark_update(rank):
    """Real wrapped decoder layers, unequal groups, and a zero-loss padding slot."""
    from tests.test_on_policy.test_packed_replay import replay_blocks
    from tests.test_on_policy.test_replay import tiny_model

    for sharding in ("FULL_SHARD", "SHARD_GRAD_OP"):
        reference = tiny_model("rnn", layer_types=["full_attention"] * 2).cuda()
        model = copy.deepcopy(reference)
        blocks = replay_blocks()
        batch = [
            {"request_id": "a", "blocks": blocks[:1]},
            {"request_id": "b", "blocks": blocks[1:]},
        ]
        torch.manual_seed(23)
        captured = {}
        expected_losses = []
        for trace in batch:
            features = torch.randn(17, 32, device="cuda")
            data = {"context_hidden": features.cpu()}
            losses = []
            for i, block in enumerate(trace["blocks"]):
                context = features[: block["context_length"]]
                p = torch.randn(3, 16, device="cuda").softmax(-1)
                with torch.no_grad():
                    q = reference.draft_probabilities(block, context)
                data[f"p_{i}"], data[f"q_{i}"] = p.cpu(), q.cpu()
                loss, _ = reference(block, context, p, q)
                losses.append(loss)
            expected_losses.append(torch.stack(losses).mean())
            captured[trace["request_id"]] = data
        expected_loss = torch.stack(expected_losses).mean()
        expected_loss.backward()
        SGDStep(reference.draft_model).step()
        backend = FSDPTrainingBackend(
            ParallelConfig(
                world_size=2, sharding_strategy=sharding, param_dtype=torch.float32
            ),
            optimizer_factory=SGDStep,
        )
        backend.prepare_model(
            model, optimizer_target=model.draft_model, cast_root_forward_inputs=False
        )
        metrics = train_effective_batch(
            backend,
            batch,
            lambda trace: captured[trace["request_id"]],
            rank,
            2,
            0.02,
            replay_blocks_per_forward=3,
        )
        state = backend._module_state_dict()
        if rank == 0:
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    state[name], expected.cpu(), atol=2e-6, rtol=2e-4, msg=name
                )
        assert abs(metrics["loss"] - expected_loss.item()) < 2e-6
        assert metrics["replay_forwards_per_rank"] == 2
        assert metrics["replay_padding_forwards"] == 1
        assert backend.optimizer.calls == 1


@unittest.skipUnless(
    os.environ.get("SPECFORGE_RUN_H200_GATE") == "1",
    "set SPECFORGE_RUN_H200_GATE=1 on the validation server",
)
class FSDPCudaTests(unittest.TestCase):
    def test_packed_flex_matches_scalar_sdpa_with_padded_query_tiles(self):
        from tests.test_on_policy.test_packed_replay import replay_blocks
        from tests.test_on_policy.test_replay import tiny_model

        for layout in (["full_attention"] * 2, ["full_attention", "sliding_attention"]):
            for head in ("vanilla", "rnn"):
                baseline = tiny_model(head, layer_types=layout).to(
                    "cuda", torch.bfloat16
                )
                packed = copy.deepcopy(baseline)
                packed.draft_model.config._attn_implementation = "flex_attention"
                # 64 blocks * 3 positions exercises multiple tiles and a partial tile.
                blocks = [copy.deepcopy(replay_blocks()[i % 5]) for i in range(64)]
                for i, block in enumerate(blocks):
                    block["context_length"] = 133 + 2 * i
                features = torch.randn(259, 32, device="cuda", dtype=torch.bfloat16)
                p = torch.randn(64, 3, 16, device="cuda").softmax(-1)
                with torch.no_grad():
                    recorded = torch.stack(
                        [
                            baseline.draft_probabilities(
                                b, features[: b["context_length"]]
                            )
                            for b in blocks
                        ]
                    )
                total = 0.0
                for i, block in enumerate(blocks):
                    loss, _ = baseline(
                        block, features[: block["context_length"]], p[i], recorded[i]
                    )
                    total += loss.detach() / len(blocks)
                    (loss / len(blocks)).backward()
                loss, parity = packed(blocks, features, p, recorded)
                (loss / len(blocks)).backward()
                self.assertLess(parity.item(), 0.02)
                torch.testing.assert_close(
                    loss / len(blocks), total, atol=0.002, rtol=0.02
                )
                for (name, a), (_, b) in zip(
                    baseline.named_parameters(), packed.named_parameters()
                ):
                    if a.requires_grad:
                        torch.testing.assert_close(
                            a.grad, b.grad, atol=0.003, rtol=0.05, msg=name
                        )

    def test_real_dspark_flex_matches_sdpa_backward(self):
        from tests.test_on_policy.test_replay import tiny_model

        self.assertTrue(torch.cuda.is_available())
        baseline = tiny_model().to(device="cuda", dtype=torch.bfloat16)
        flex = tiny_model().to(device="cuda", dtype=torch.bfloat16)
        flex.draft_model.config._attn_implementation = "flex_attention"
        features = torch.randn(133, 32, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(3, 16, device="cuda").softmax(-1)
        block = {
            "context_length": 133,
            "anchor": 2,
            "proposal": [3, 4, 5],
            "valid_mask": [True, True, True],
            "sampling": {"temperature": 1.0},
        }
        losses = []
        for model in (baseline, flex):
            loss, _ = model(block, features, target, target)
            loss.backward()
            losses.append(loss)
        torch.testing.assert_close(losses[0], losses[1], atol=0.002, rtol=0.02)
        for (name, a), (_, b) in zip(
            baseline.named_parameters(), flex.named_parameters()
        ):
            if a.requires_grad:
                self.assertIsNotNone(a.grad, name)
                self.assertIsNotNone(b.grad, name)
                torch.testing.assert_close(a.grad, b.grad, atol=0.002, rtol=0.02)

    def test_two_rank_full_shard_matches_sample_mean(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(directory,), nprocs=2, join=True)
