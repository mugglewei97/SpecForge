"""Two-rank CPU FSDP coverage, including a rank with no supervised anchors."""

import functools
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from specforge.inference.target_engine.proposal_verification import (
    ProposalVerificationSession,
)
from specforge.legacy.optimizer import BF16Optimizer
from specforge.modeling.draft.dflash2 import Qwen3DFlash2DecoderLayer
from tests.test_utils.test_dflash2_tv_acceptance_beta import make_dflash2


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        model, target = make_dflash2()
        model.bfloat16()
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
        optimizer = BF16Optimizer(
            model.draft_model, lr=1e-3, total_steps=2, fsdp_model=wrapped
        )
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        for step in range(2):
            mask = (
                torch.zeros_like(ids)
                if rank == 1 and step == 0
                else torch.ones_like(ids)
            )
            with ProposalVerificationSession(target) as session:
                output = session.capture(ids, torch.ones_like(ids), mask)
                result = wrapped(
                    ids,
                    output.hidden_states.bfloat16(),
                    mask,
                    last_hidden_states=output.last_hidden_states,
                    tv_verification_session=session,
                )
            assert result[-1]["tv_valid_blocks"] == (2 if step == 0 else 4)
            requests = len(target.requests)
            result[0].backward()
            assert len(target.requests) == requests
            norm = optimizer.step()
            assert torch.isfinite(norm) and norm > 0
        # The two replicas must still agree after sharded gradients and updates.
        with FSDP.summon_full_params(wrapped):
            flat = torch.cat(
                [p.detach().float().reshape(-1) for p in model.draft_model.parameters()]
            )
            reference = flat.clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(flat, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_dflash2_beta_two_rank_fsdp_with_empty_rank(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
