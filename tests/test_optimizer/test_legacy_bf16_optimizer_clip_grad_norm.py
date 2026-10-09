import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from specforge.legacy.optimizer import BF16Optimizer, build_bf16_optimizer


def _assert_step(optimizer, expected_norm, *, native_clip=False):
    # Inspect Adam's first moment to verify the gradients actually used for the
    # update, not just the norm reported by step().
    expected_grads = []
    for param in optimizer.model_params:
        if param.grad is None:
            expected_grads.append(None)
            continue
        grad = param.grad.detach().clone()
        coefficient = torch.clamp(
            optimizer.max_grad_norm / (expected_norm + 1e-6), max=1
        )
        if native_clip:
            grad.mul_(coefficient.to(grad.dtype))
            grad = grad.float()
        else:
            grad = grad.float().mul_(coefficient)
        expected_grads.append(grad)

    actual_norm = optimizer.step()

    torch.testing.assert_close(actual_norm.float(), expected_norm.float())
    torch.testing.assert_close(optimizer.get_grad_norm(), actual_norm)
    for master, expected in zip(optimizer.fp32_params, expected_grads):
        if expected is None:
            assert master not in optimizer.optimizer.state
        else:
            torch.testing.assert_close(
                optimizer.optimizer.state[master]["exp_avg"], expected * 0.1
            )
        assert master.grad is None
    assert all(param.grad is None for param in optimizer.model_params)


def _distributed_worker(rank, init_file, device_type):
    device = (
        torch.device(device_type, rank)
        if device_type == "cuda"
        else torch.device("cpu")
    )
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for strategy in (
                ShardingStrategy.SHARD_GRAD_OP,
                ShardingStrategy.NO_SHARD,
            ):
                model = torch.nn.Linear(2, 1, bias=False, device=device, dtype=dtype)
                # Match DSpark's nested FSDP layout: the optimizer sees only
                # the draft submodule, while clipping uses the root wrapper.
                child = FSDP(
                    model,
                    device_id=device,
                    use_orig_params=True,
                    sharding_strategy=strategy,
                )
                root = FSDP(
                    torch.nn.Sequential(child),
                    device_id=device,
                    use_orig_params=True,
                    sharding_strategy=strategy,
                )
                optimizer = build_bf16_optimizer(
                    model,
                    fsdp_model=root,
                    lr=1e-3,
                    max_grad_norm=1.0,
                    warmup_ratio=0.0,
                )
                inputs = torch.tensor([[3.0, 4.0]], device=device, dtype=dtype)
                root(inputs).sum().backward()
                # Shards [3] and [4] must both use norm 5, rather than norms
                # 3 and 4. NO_SHARD must count the replicated vector once.
                _assert_step(
                    optimizer, torch.tensor(5.0, device=device), native_clip=True
                )

        # A rank with an empty shard must still join the norm reduction.
        model = torch.nn.Linear(1, 1, bias=False, device=device)
        root = FSDP(
            model,
            device_id=device,
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        )
        optimizer = build_bf16_optimizer(
            model,
            fsdp_model=root,
            lr=1e-3,
            max_grad_norm=1.0,
            warmup_ratio=0.0,
        )
        root(torch.tensor([[3.0]], device=device)).sum().backward()
        if rank == 1:
            assert model.weight.numel() == 0
            assert model.weight.grad is None
        _assert_step(optimizer, torch.tensor(3.0, device=device), native_clip=True)

        # Distributed initialization alone must not trigger a norm reduction
        # for a regular, replicated model without FSDP.
        model = torch.nn.Linear(2, 1, bias=False, device=device)
        optimizer = build_bf16_optimizer(model, lr=1e-3, max_grad_norm=1.0)
        model.weight.grad = torch.tensor([[3.0, 4.0]], device=device)
        _assert_step(optimizer, torch.tensor(5.0, device=device))
    finally:
        dist.destroy_process_group()


class TestLegacyBF16OptimizerClipGradNorm(unittest.TestCase):
    def test_non_fsdp_keeps_fp32_master_clipping(self):
        model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
        optimizer = BF16Optimizer(model, lr=1e-3, max_grad_norm=1.0)
        model.weight.grad = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
        _assert_step(optimizer, torch.tensor(5.0))

    def _run_distributed(self, device_type):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_worker,
                args=(os.path.join(directory, "init"), device_type),
                nprocs=2,
                join=True,
            )

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_fsdp_global_clipping_cpu(self):
        self._run_distributed("cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_fsdp_global_clipping_cuda(self):
        self._run_distributed("cuda")


if __name__ == "__main__":
    unittest.main()
