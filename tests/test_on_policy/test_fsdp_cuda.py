"""Two-GPU FSDP gate; run on H200 before the full SGLang smoke recipe."""

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from specforge.on_policy.trainer import train_effective_batch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from tests.test_on_policy.test_training import tensors, traces


class CudaReplay(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, device=device))

    def forward(self, block, *_args):
        return self.weight.square().sum() * block["coefficient"], self.weight.new_zeros(
            ()
        )


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
    try:
        model = CudaReplay(torch.device("cuda", rank))
        backend = FSDPTrainingBackend(
            ParallelConfig(
                world_size=2, sharding_strategy="FULL_SHARD", param_dtype=torch.float32
            ),
            optimizer_factory=SGDStep,
        )
        backend.prepare_model(model, optimizer_target=model)
        train_effective_batch(backend, traces(), tensors, rank, 2, 0.02)
        state = backend._module_state_dict()
        if rank == 0:
            torch.testing.assert_close(state["weight"], torch.full((8,), 0.3))
        assert backend.optimizer.calls == 1
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    os.environ.get("SPECFORGE_RUN_H200_GATE") == "1",
    "set SPECFORGE_RUN_H200_GATE=1 on the validation server",
)
class FSDPCudaTests(unittest.TestCase):
    def test_two_rank_full_shard_matches_sample_mean(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(directory,), nprocs=2, join=True)
