"""Two-rank DSpark beta updates with empty and unequal on-policy rollouts."""

import functools
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from specforge.core.dspark_on_policy import collect_dspark_rollout
from specforge.legacy.dflash2 import Qwen3DFlash2DecoderLayer
from specforge.legacy.optimizer import BF16Optimizer
from tests.test_utils.test_dspark_on_policy import make_on_policy_dspark


def _parameters(model):
    return torch.cat(
        [
            parameter.detach().float().reshape(-1)
            for parameter in model.draft_model.parameters()
        ]
    )


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(173)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        model, engine = make_on_policy_dspark(
            head="carh", conv=True, budget=10 if rank == 0 else 2, rounds=9
        )
        model.bfloat16()
        engine.model.bfloat16()
        wrapped = FSDP(
            model,
            device_id=torch.device("cpu"),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Qwen3DFlash2DecoderLayer},
            ),
        )
        with FSDP.summon_full_params(wrapped):
            initial = _parameters(model)
        optimizer = BF16Optimizer(
            model.draft_model,
            lr=1e-3,
            total_steps=2,
            warmup_ratio=0.0,
            fsdp_model=wrapped,
        )
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        attention = torch.ones_like(ids)
        supervision = torch.tensor([[0, 0, 0, 1, 1, 1]])
        for step in range(2):
            torch.manual_seed(1500 + 10 * rank + step)
            loss_mask = (
                torch.zeros_like(supervision)
                if rank == 1 and step == 0
                else supervision
            )
            records = []

            def collect(*args, **kwargs):
                replay = collect_dspark_rollout(*args, **kwargs)
                records.append(replay)
                return replay

            with (
                engine.on_policy_rollout() as session,
                patch(
                    "specforge.core.dspark_on_policy.collect_dspark_rollout",
                    side_effect=collect,
                ),
            ):
                result = wrapped(
                    ids,
                    None,
                    loss_mask,
                    attention_mask=attention,
                    tv_on_policy_session=session,
                )
                assert session.closed and len(records) == 1
                replay = records[0]
                local_rounds = replay.keep.sum().long()
                counts = [torch.zeros_like(local_rounds) for _ in range(2)]
                dist.all_gather(counts, local_rounds)
                rounds = [int(count) for count in counts]
                assert rounds[0] >= 2
                assert rounds[1] == (0 if step == 0 else 1)
                assert result[-1]["tv_valid_blocks"] == sum(rounds)
                assert result[-1]["tv_rollout_rounds"] == sum(rounds)
                assert result[-1]["tv_rollout_generated_tokens"] == 10 + 2 * step
                assert result[-1]["tv_rollout_round_limit_count"] == 0
                assert torch.isfinite(result[0])
                if rank == 1 and step == 0:
                    assert result[0] == 0
                    assert not replay.keep.any()
                with (
                    patch.object(
                        engine.model.model,
                        "forward",
                        side_effect=AssertionError("target in backward"),
                    ),
                    patch(
                        "torch.multinomial",
                        side_effect=AssertionError("backward resampled"),
                    ),
                ):
                    result[0].backward()
                assert all(p.grad is None for p in engine.model.parameters())
            norm = optimizer.step()
            assert torch.isfinite(norm) and norm > 0
            assert all(p.grad is None for p in model.draft_model.parameters())
            with FSDP.summon_full_params(wrapped):
                current = _parameters(model)
                reference = current.clone()
                dist.broadcast(reference, src=0)
                torch.testing.assert_close(current, reference, rtol=0, atol=0)
        assert not torch.equal(current, initial)
    finally:
        dist.destroy_process_group()


def test_dspark_on_policy_fsdp_empty_rank_then_unequal_rounds(tmp_path):
    mp.spawn(
        _worker,
        args=((tmp_path / "dspark-on-policy-rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
