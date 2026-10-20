"""Two-rank FSDP with fresh rollouts, empty ranks, and unequal episode lengths."""

import functools
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from specforge.core.dflash2_on_policy import collect_dflash2_rollout
from specforge.legacy.optimizer import BF16Optimizer
from specforge.modeling.draft.dflash2 import Qwen3DFlash2DecoderLayer
from tests.test_utils.test_dflash2_on_policy import make_on_policy_models


def _flat_draft_parameters(model):
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
        # After the seed, rank 0 needs at least two rounds to emit nine tokens.
        # Rank 1 emits its single remaining token in exactly one active round.
        # Nine rounds suffice even if rank 0 rejects its first proposal always.
        model, engine = make_on_policy_models(budget=10 if rank == 0 else 2, rounds=9)
        # Use the tiny vocabulary as support so the fixture cannot land on a
        # locally flat TV region (q > p on every one of four sparse candidates).
        # Sparse-support value/gradient equivalence is tested separately.
        model.draft_model.candidate_selector.top_k = 16
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
            initial = _flat_draft_parameters(model)
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

            def record_rollout(*args, **kwargs):
                replay = collect_dflash2_rollout(*args, **kwargs)
                records.append(replay)
                return replay

            with (
                engine.on_policy_rollout() as session,
                patch(
                    "specforge.core.dflash2_on_policy.collect_dflash2_rollout",
                    side_effect=record_rollout,
                ),
                patch.object(
                    engine.model.model,
                    "forward",
                    wraps=engine.model.model.forward,
                ) as target_forward,
            ):
                result = wrapped(
                    ids,
                    None,
                    loss_mask,
                    attention_mask=attention,
                    tv_on_policy_session=session,
                )
                assert session.closed
                assert len(records) == 1
                replay = records[0]
                local_rounds = replay.keep.sum().to(torch.long)
                counts = [torch.zeros_like(local_rounds) for _ in range(2)]
                dist.all_gather(counts, local_rounds)
                rounds_by_rank = [int(count) for count in counts]
                assert rounds_by_rank[0] >= 2
                assert rounds_by_rank[1] == (0 if step == 0 else 1)
                total_rounds = sum(rounds_by_rank)
                assert result[-1]["tv_valid_blocks"] == total_rounds
                assert result[-1]["tv_rollout_rounds"] == total_rounds
                assert result[-1]["tv_rollout_episodes"] == step + 1
                assert result[-1]["tv_rollout_generated_tokens"] == 10 + 2 * step
                assert result[-1]["tv_rollout_round_limit_count"] == 0
                assert torch.isfinite(result[0])
                if rank == 1 and step == 0:
                    assert not replay.keep.any() and not replay.valid.any()
                    assert result[0] == 0
                calls = target_forward.call_count
                assert calls >= 1
                with (
                    patch.object(
                        engine.model.model,
                        "forward",
                        side_effect=AssertionError("target called during backward"),
                    ),
                    patch(
                        "torch.multinomial",
                        side_effect=AssertionError("backward resampled proposals"),
                    ),
                ):
                    result[0].backward()
                assert target_forward.call_count == calls
                assert all(
                    parameter.grad is None for parameter in engine.model.parameters()
                )
            norm = optimizer.step()
            assert torch.isfinite(norm) and norm > 0, (rank, step, norm)
            assert all(
                parameter.dtype == torch.float32 for parameter in optimizer.fp32_params
            )
            assert all(
                parameter.grad is None for parameter in model.draft_model.parameters()
            )

        assert optimizer.get_grad_stats()["grad_norm_mean"] > 0
        with FSDP.summon_full_params(wrapped):
            final = _flat_draft_parameters(model)
            assert not torch.equal(final, initial)
            reference = final.clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(final, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_dflash2_on_policy_two_rank_fsdp_with_empty_then_unequal_rollouts(tmp_path):
    mp.spawn(
        _worker,
        args=((tmp_path / "on-policy-rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
