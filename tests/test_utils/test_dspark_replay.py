"""Replay alignment, effective-batch gradients and strict version sequencing."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from specforge.core.tv_acceptance import (
    select_replay_anchors,
    reduce_tv_terms,
    tv_acceptance_terms,
)
from specforge.data.dspark_replay import (
    ReplayCollator,
    RolloutTraceDataset,
    assert_replay_batch,
)
from specforge.legacy.dspark_training.replay import SynchronousReplayController


def trace(sample_id="a", version=0, anchors=(2, 5), length=12):
    return dict(
        sample_id=sample_id,
        input_ids=list(range(length)),
        prompt_length=3,
        anchor_positions=list(anchors),
        draft_version=version,
        target_version="target-sha",
        round_metadata=[
            dict(draft_length=7, accepted_draft_tokens=2, committed_tokens=3)
            for _ in anchors
        ],
        sampling_config=dict(
            temperature=1.0, top_p=1.0, top_k=-1, speculative={"algorithm": "DSPARK"}
        ),
    )


def collator(**kwargs):
    return ReplayCollator(
        draft_version=0, target_version="target-sha", num_anchors=20, **kwargs
    )


def test_raw_tokens_masks_overlapping_anchors_and_no_acceptance_truncation(tmp_path):
    record = trace()
    record["loss_mask"] = [1] * 10 + [0, 0]
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(record) + "\n")
    dataset = RolloutTraceDataset(path, draft_version=0, target_version="target-sha")
    data = collator()(list(dataset))
    assert data["input_ids"].tolist() == [record["input_ids"]]
    assert data["loss_mask"].tolist() == [
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    ]
    assert data["anchor_positions"].tolist() == [[2, 5]]  # anchor 2 is in prompt
    assert data["block_keep_mask"].tolist() == [[True, True]]
    assert_replay_batch(data, 0, "target-sha")
    with pytest.raises(ValueError, match="draft_version"):
        assert_replay_batch(data, 1, "target-sha")


def test_ragged_anchors_empty_sample_and_eos():
    a, b = trace(), trace("b", anchors=(), length=6)
    data = collator(eos_token_ids=[6])([a, b])
    assert data["block_keep_mask"].tolist() == [[True, True], [False, False]]
    assert data["loss_mask"][0, 6] == 1 and not data["loss_mask"][0, 7:].any()
    assert data["attention_mask"][1].tolist() == [1] * 6 + [0] * 6


def test_budget_samples_late_anchors_and_ignores_padding():
    torch.manual_seed(412)
    anchors = torch.arange(99)[None]
    seen = torch.zeros(99)
    for _ in range(100):
        chosen, keep = select_replay_anchors(
            anchors,
            torch.ones_like(anchors).bool(),
            torch.ones(1, 100),
            torch.ones(1, 100),
            5,
        )
        seen[chosen[keep]] += 1
    assert 60 < seen[:20].sum() < 140 and 60 < seen[-20:].sum() < 140
    chosen, keep = select_replay_anchors(
        torch.tensor([[3, -200]]),
        torch.tensor([[True, False]]),
        torch.ones(1, 6),
        torch.ones(1, 6),
        3,
    )
    assert chosen.tolist() == [[3]] and keep.all()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"draft_version": 1}, "version"),
        ({"anchor_positions": [2, 2]}, "increasing"),
        ({"anchor_positions": [-1]}, "absolute"),
        ({"round_metadata": []}, "identical"),
        ({"input_ids": [1.0, 2.0, 3.0]}, "token"),
        ({"sampling_config": {}}, "sampling_config"),
    ],
)
def test_bad_trace_fails_closed(change, match):
    with pytest.raises(ValueError, match=match):
        collator()([{**trace(), **change}])


def test_microbatch_gradient_matches_single_global_block_mean():
    torch.manual_seed(123)
    q = torch.randn(5, 3, 8, dtype=torch.double, requires_grad=True)
    p = torch.randn_like(q)
    masks = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 1]])
    total = tv_acceptance_terms(q, p, masks)
    reference = torch.autograd.grad(total.loss_sum / total.block_count, q)[0]
    for start, stop in ((0, 2), (2, 3), (3, 5)):
        terms = tv_acceptance_terms(q[start:stop], p[start:stop], masks[start:stop])
        reduce_tv_terms(terms, global_block_count=total.block_count)[0].backward()
    torch.testing.assert_close(q.grad, reference)


def _distributed_replay_worker(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        # The first microbatch is empty on rank 0, the second empty on rank 1.
        masks = torch.tensor([[[0, 0], [0, 0], [1, 1]], [[1, 1], [1, 0], [0, 0]]])
        targets = torch.arange(36, dtype=torch.double).reshape(2, 3, 2, 3).sin()
        weight = torch.tensor([0.2, -0.3, 0.8], dtype=torch.double, requires_grad=True)
        total_count = (masks[rank].sum(-1) > 0).sum().to(torch.double)
        dist.all_reduce(total_count)
        for start, stop in [(0, 2), (2, 3)]:
            logits = weight.expand(stop - start, 2, 3)
            terms = tv_acceptance_terms(
                logits, targets[rank, start:stop], masks[rank, start:stop]
            )
            reduce_tv_terms(terms, global_block_count=total_count)[0].backward()
        dist.all_reduce(weight.grad)
        weight.grad /= 2  # same gradient averaging contract as FSDP
        reference_weight = weight.detach().clone().requires_grad_()
        all_terms = tv_acceptance_terms(
            reference_weight.expand(2, 3, 2, 3), targets, masks
        )
        (all_terms.loss_sum / all_terms.block_count).backward()
        torch.testing.assert_close(weight.grad, reference_weight.grad)
    finally:
        dist.destroy_process_group()


def test_two_rank_accumulation_uses_one_global_block_denominator(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_replay_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )


def test_synchronous_version_state_machine_and_failed_sync():
    events = []
    backend = SimpleNamespace(
        rollout=lambda prompts, version: events.append(("rollout", version))
        or [trace(version=version)]
    )

    def sync(version):
        events.append(("sync", version))
        return version

    optimizer = SimpleNamespace(step=lambda: events.append(("optimizer", None)))
    controller = SynchronousReplayController(backend, sync)
    with pytest.raises(RuntimeError, match="sync"):
        controller.collect([])
    controller.synchronize()
    controller.collect([])
    with pytest.raises(RuntimeError, match="sync"):
        controller.collect([])
    controller.update(optimizer)
    assert events == [("sync", 0), ("rollout", 0), ("optimizer", None), ("sync", 1)]
    with pytest.raises(RuntimeError, match="exactly one"):
        controller.update(optimizer)
    controller.collect([])
    controller.sync_weights = lambda version: -1
    with pytest.raises(RuntimeError, match="acknowledgement"):
        controller.update(optimizer)
    assert controller.phase == "failed"
    with pytest.raises(RuntimeError):
        controller.collect([])


def test_real_dspark_replay_one_capture_no_target_backward_and_full_window():
    from tests.test_utils.test_tv_acceptance import make_model
    from specforge.legacy.dspark_training.trainer import _forward_dspark_data_batch

    model, target = make_model()
    model.num_anchors = 10
    data = collator()([trace()])
    data["tv_global_block_count"] = torch.tensor(2.0)
    args = SimpleNamespace(
        anchor_mode="replay",
        draft_version=0,
        rollout_target_version="target-sha",
        seed=1,
        multi_teacher_oracle_export_dir=None,
        multi_teacher_oracle_cache=None,
        step_seeded_rollouts=False,
    )
    with (
        patch("specforge.legacy.dspark_training.trainer.dist.get_rank", return_value=0),
        patch.object(
            target, "generate_dflash_data", wraps=target.generate_dflash_data
        ) as capture,
    ):
        result = _forward_dspark_data_batch(data, args, "cpu", target, model, True, 1)
        result[0].backward()
        capture.assert_called_once()
    # 7 positions at anchor 2 + 6 at anchor 5, despite only 2 accepted drafts.
    assert result[4].sum().item() == 13
    assert model.draft_model.markov_head.markov_w2.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.lm_head.parameters())
    args.draft_version = 1
    with patch.object(
        target,
        "generate_dflash_data",
        side_effect=AssertionError("capture must not run"),
    ):
        with pytest.raises(ValueError, match="draft_version"):
            _forward_dspark_data_batch(data, args, "cpu", target, model, True, 2)


def test_parallel_overlapping_blocks_equal_independent_replays():
    from tests.test_utils.test_tv_acceptance import make_model

    model, target = make_model()
    model.num_anchors = 10
    data = collator()([trace()])
    captured = target.generate_dflash_data(
        data["input_ids"], data["attention_mask"], data["loss_mask"]
    )
    kwargs = dict(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        loss_mask=data["loss_mask"],
        hidden_states=captured.hidden_states,
        last_hidden_states=captured.last_hidden_states,
        anchor_mode="replay",
        tv_global_block_count=2.0,
    )
    result = model(
        **kwargs,
        anchor_positions=data["anchor_positions"],
        block_keep_mask=data["block_keep_mask"],
    )
    result[0].backward()
    parallel = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    separate = 0
    for s in [2, 5]:
        value = model(
            **kwargs,
            anchor_positions=torch.tensor([[s]]),
            block_keep_mask=torch.tensor([[True]]),
        )[0]
        separate += value.detach()
        value.backward()
    torch.testing.assert_close(result[0], separate)
    for name, parameter in model.named_parameters():
        if name in parallel:
            torch.testing.assert_close(
                parameter.grad, parallel[name], rtol=1e-4, atol=1e-6
            )


def test_training_loop_collects_whole_accumulation_before_backward(
    tmp_path, monkeypatch
):
    from specforge.legacy.dspark_training import replay
    from specforge.legacy.dspark_training.trainer import _forward_dspark_data_batch
    from tests.test_utils.test_tv_acceptance import make_model

    model, target = make_model()
    model.num_anchors = 10
    events = []

    class Backend:
        def sync(self, state, config, names, version):
            events.append(("sync", version))
            assert "markov_head.markov_w2.weight" in names
            return version

        def rollout(self, prompts, version):
            events.append(("rollout", version, len(prompts)))
            return [trace(sample_id=s, version=version) for s, tokens in prompts]

        def generate_dflash_data(self, *values):
            events.append(("capture", args.draft_version))
            return target.generate_dflash_data(*values)

    class Loader:
        sampler = SimpleNamespace(set_epoch=lambda epoch: None)

        def __iter__(self):
            # A partial last accumulation window still produces one update.
            return iter(
                [[dict(sample_id=str(i), input_ids=[0, 1, 2])] for i in range(3)]
            )

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = SimpleNamespace(
        step=lambda: (
            events.append(("optimizer", args.draft_version)),
            opt.step(),
            opt.zero_grad(),
        ),
        get_learning_rate=lambda: 0.01,
    )
    args = SimpleNamespace(
        anchor_mode="replay",
        output_dir=str(tmp_path),
        rollout_target_version="target-sha",
        num_anchors=10,
        num_epochs=1,
        accumulation_steps=2,
        micro_batch_size=1,
        batch_size=1,
        log_interval=100,
        save_interval=100,
        max_steps=None,
        stop_after_steps=None,
        seed=0,
        multi_teacher_oracle_export_dir=None,
        multi_teacher_oracle_cache=None,
        step_seeded_rollouts=False,
    )
    monkeypatch.setattr(replay, "full_draft_state", lambda model: {})
    monkeypatch.setattr(replay.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(replay.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(replay.dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(replay.dist, "all_reduce", lambda *a, **k: None)
    replay.run_replay_training(
        args,
        Loader(),
        Backend(),
        model,
        model.draft_model,
        optimizer,
        "cpu",
        SimpleNamespace(log=lambda *a, **k: None),
        2,
        True,
        forward_batch=_forward_dspark_data_batch,
        save_checkpoint=lambda *a: None,
    )
    assert events == [
        ("sync", 0),
        ("rollout", 0, 2),
        ("capture", 0),
        ("capture", 0),
        ("optimizer", 0),
        ("sync", 1),
        ("rollout", 1, 1),
        ("capture", 1),
        ("optimizer", 1),
        ("sync", 2),
    ]
    assert len(list(tmp_path.rglob("*.jsonl"))) == 2
