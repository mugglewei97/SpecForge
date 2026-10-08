import tempfile
import unittest
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from specforge.on_policy.trainer import replay_schedule, train_effective_batch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig


def traces():
    return [
        {
            "request_id": "a",
            "blocks": [{"valid_mask": [True], "context_length": 1, "coefficient": 2.0}],
        },
        {
            "request_id": "b",
            "blocks": [
                {"valid_mask": [True], "context_length": 1, "coefficient": c}
                for c in (3.0, 4.0, 8.0)
            ],
        },
        {"request_id": "empty", "blocks": []},
    ]


def tensors(_trace):
    value = {"context_hidden": torch.ones(1, 1)}
    for index in range(3):
        value[f"p_{index}"] = value[f"q_{index}"] = torch.ones(1)
    return value


class ToyReplay(nn.Module):
    def __init__(self, parity=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.parity = parity
        self.in_no_sync = False
        self.forward_sync_states = []

    @contextmanager
    def no_sync(self):
        self.in_no_sync = True
        try:
            yield
        finally:
            self.in_no_sync = False

    def forward(self, block, *_args):
        self.forward_sync_states.append(self.in_no_sync)
        return self.weight.square() * block["coefficient"], torch.tensor(self.parity)


class CountingSGD:
    def __init__(self, model):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.calls = 0

    def step(self):
        self.calls += 1
        self.optimizer.step()
        self.optimizer.zero_grad()


def _distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=3
    )
    try:
        module = DistributedDataParallel(ToyReplay())
        backend = FSDPTrainingBackend(ParallelConfig())
        backend.prepare_model(module, wrap=False)
        optimizer = CountingSGD(module)
        backend.set_optimizer(optimizer)
        metrics = train_effective_batch(backend, traces(), tensors, rank, 3, 0.02)
        # Mean of sample means: (2 + (3+4+8)/3)/2 = 3.5; grad = 7.
        torch.testing.assert_close(module.module.weight, torch.tensor(0.3))
        assert optimizer.calls == 1
        assert abs(metrics["loss"] - 3.5) < 1e-6
    finally:
        dist.destroy_process_group()


class TrainingTests(unittest.TestCase):
    def test_ranks_have_equal_slots_and_sample_weights(self):
        plans = [replay_schedule(traces(), rank, 3) for rank in range(3)]
        self.assertEqual([len(plan) for plan in plans], [3, 3, 3])
        self.assertEqual([entry[2] for entry in plans[0]], [1.5, 0, 0])
        self.assertEqual([entry[2] for entry in plans[1]], [0.5, 0.5, 0.5])
        self.assertEqual([entry[2] for entry in plans[2]], [0, 0, 0])

    def test_single_update_and_forward_inside_no_sync(self):
        model = ToyReplay()
        backend = FSDPTrainingBackend(ParallelConfig())
        backend.prepare_model(model, wrap=False)
        optimizer = CountingSGD(model)
        backend.set_optimizer(optimizer)
        metrics = train_effective_batch(backend, traces(), tensors, 0, 1, 0.02)
        self.assertEqual(model.forward_sync_states, [True, True, True, False])
        self.assertEqual(optimizer.calls, 1)
        self.assertAlmostEqual(metrics["loss"], 3.5)
        self.assertAlmostEqual(model.weight.item(), 0.3, places=6)

    def test_parity_failure_prevents_optimizer_step(self):
        model = ToyReplay(parity=0.5)
        backend = FSDPTrainingBackend(ParallelConfig())
        backend.prepare_model(model, wrap=False)
        optimizer = CountingSGD(model)
        backend.set_optimizer(optimizer)
        with self.assertRaisesRegex(RuntimeError, "optimizer was not stepped"):
            train_effective_batch(backend, traces(), tensors, 0, 1, 0.02)
        self.assertEqual(optimizer.calls, 0)
        self.assertEqual(model.weight.item(), 1)

    def test_three_rank_sample_mean_with_empty_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_worker, args=(directory,), nprocs=3, join=True)
