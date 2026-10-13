"""Scheduler trace hooks, complete draft synchronization and source patch gates."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from specforge.inference import dspark_replay_server as server
from specforge.inference import dspark_replay_patch as patcher


def scheduler_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECFORGE_DSPARK_REPLAY_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    draft = torch.nn.Module()
    draft.register_parameter("weight", torch.nn.Parameter(torch.zeros(2, 3)))
    draft.markov_head = torch.nn.Linear(3, 2, bias=False)
    draft._stacked_ctx_kv_cache = {"stale": True}
    draft._fused_kv_write_cache = ("stale",)
    worker = type("DSparkWorkerV2", (), {})()
    worker.draft_model = draft
    worker._simulate_acc_len = 0
    worker._verify_planner = SimpleNamespace(mode_value="static")
    worker.gamma = 3
    worker.verify_num_draft_tokens = 4
    target = torch.nn.Linear(3, 2)
    scheduler = SimpleNamespace(
        draft_worker=worker,
        is_fully_idle=lambda: True,
        flush_cache=lambda: True,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=target,
                spec_aux_config=SimpleNamespace(dflash_target_layer_ids=[1]),
            )
        ),
    )
    state = {n: torch.ones_like(p) for n, p in draft.named_parameters()}
    save_file(state, str(tmp_path / "weights.safetensors"))

    def rpc(action, **kwargs):
        (tmp_path / "request.json").write_text(
            json.dumps(dict(action=action, **kwargs))
        )
        server.dispatch(scheduler)
        return json.loads((tmp_path / "response.json").read_text())

    sync = dict(
        draft_version=0,
        target_version="t",
        trainable_names=list(state),
        capture_layer_ids=[1],
    )
    return scheduler, rpc, sync


def test_sync_full_draft_markov_readback_cache_invalidation(tmp_path, monkeypatch):
    scheduler, rpc, sync = scheduler_fixture(tmp_path, monkeypatch)
    target_before = {
        n: p.clone()
        for n, p in scheduler.tp_worker.model_runner.model.named_parameters()
    }
    ack = rpc("sync", **sync)
    assert ack["draft_version"] == 0 and ack["cache_flushed"]
    for p in scheduler.draft_worker.draft_model.parameters():
        assert torch.equal(p, torch.ones_like(p))
    for n, p in scheduler.tp_worker.model_runner.model.named_parameters():
        torch.testing.assert_close(p, target_before[n])
    assert scheduler.draft_worker.draft_model._stacked_ctx_kv_cache is False
    assert scheduler.draft_worker.draft_model._fused_kv_write_cache is None
    with pytest.raises(RuntimeError, match="follow one replay"):
        rpc("sync", **{**sync, "draft_version": 1})
    assert scheduler.draft_worker._specforge_replay["phase"] == "failed"


@pytest.mark.parametrize(
    "failure", ["missing_markov", "flush", "wrong_layers", "simulation"]
)
def test_failed_sync_never_acknowledges_or_allows_rollout(
    tmp_path, monkeypatch, failure
):
    scheduler, rpc, sync = scheduler_fixture(tmp_path, monkeypatch)
    if failure == "missing_markov":
        save_file({"weight": torch.ones(2, 3)}, str(tmp_path / "weights.safetensors"))
    elif failure == "flush":
        scheduler.flush_cache = lambda: False
    elif failure == "wrong_layers":
        sync["capture_layer_ids"] = [0]
    else:
        scheduler.draft_worker._simulate_acc_len = 1
    with pytest.raises((ValueError, RuntimeError)):
        rpc("sync", **sync)
    assert not (tmp_path / "response.json").exists()
    with pytest.raises(RuntimeError):
        rpc("begin", draft_version=0, requests=[])


def test_real_round_positions_correction_bonus_and_final_stop(tmp_path, monkeypatch):
    scheduler, rpc, sync = scheduler_fixture(tmp_path, monkeypatch)
    rpc("sync", **sync)
    prompts = [
        dict(
            sample_id="a",
            input_ids=[10, 11],
            sampling_config=dict(temperature=0.8, top_p=1, top_k=-1, speculative={}),
        )
    ]
    rpc("begin", draft_version=0, requests=prompts)
    # Prefill emitted root 12 at absolute position 2, not position 1.
    req = SimpleNamespace(
        rid="a", output_ids_through_stop=[12, 13, 14, 15, 16, 17], finished=lambda: True
    )
    batch = SimpleNamespace(reqs=[req])
    worker = scheduler.draft_worker
    server.register_requests(worker, batch)
    server.record_round(
        worker,
        batch,
        torch.tensor([2]),
        torch.tensor([[13, 99, 98]]),
        SimpleNamespace(
            correct_len=torch.tensor([1]),
            commit_lens=torch.tensor([2]),
            out_tokens=torch.tensor([[13, 14, 0, 0]]),
        ),
        None,
    )
    server.record_round(
        worker,
        batch,
        torch.tensor([4]),
        torch.tensor([[15, 16, 17]]),
        SimpleNamespace(
            correct_len=torch.tensor([3]),
            commit_lens=torch.tensor([4]),
            out_tokens=torch.tensor([[15, 16, 17, 18]]),
        ),
        None,
    )
    result = rpc("finish")["records"][0]
    assert result["input_ids"] == [10, 11, 12, 13, 14, 15, 16, 17]
    assert result["anchor_positions"] == [2, 4]
    first, second = result["round_metadata"]
    assert (
        first["root_tokens"],
        first["correction_tokens"],
        first["bonus_tokens"],
    ) == (0, 1, 0)
    assert second["accepted_draft_tokens"] == 3
    assert second["verifier_committed_tokens"] == 4 and second["committed_tokens"] == 3
    assert second["bonus_tokens"] == 0  # actual stop truncated the verifier bonus
    assert rpc("sync", **{**sync, "draft_version": 1})["draft_version"] == 1


def test_idle_guard_does_not_update_during_sample(tmp_path, monkeypatch):
    scheduler, rpc, sync = scheduler_fixture(tmp_path, monkeypatch)
    scheduler.is_fully_idle = lambda: False
    with pytest.raises(RuntimeError, match="finish"):
        rpc("sync", **sync)
    assert torch.equal(scheduler.draft_worker.draft_model.weight, torch.zeros(2, 3))


def test_fused_weight_mapping_is_complete():
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.qkv_proj = torch.nn.Linear(3, 6, bias=False)
    parts = {
        f"layer.{name}_proj.weight": torch.full((2, 3), float(i))
        for i, name in enumerate(["q", "k", "v"])
    }
    mapped = server._mapped_weights(model, parts)
    assert mapped["layer.qkv_proj.weight"][:, 0].tolist() == [0, 0, 1, 1, 2, 2]
    parts.pop("layer.k_proj.weight")
    with pytest.raises(ValueError, match="missing fused"):
        server._mapped_weights(model, parts)


def test_patch_checks_all_sources_before_edits_and_is_idempotent(tmp_path, monkeypatch):
    texts = {
        patcher.WORKER: "\n".join(old for old, new in patcher.EDITS[patcher.WORKER]),
        patcher.SCHEDULER: "def is_fully_idle\ndef handle_rpc_request\n"
        + patcher.EDITS[patcher.SCHEDULER][0][0],
        "srt/models/dspark.py": "class Qwen3DSparkModel\n_fused_kv_write_cache\n_stacked_ctx_kv_cache",
        "srt/entrypoints/engine.py": "def collective_rpc\ndef generate",
    }
    for name, text in texts.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(
        patcher,
        "AUDITED_SOURCES",
        {patcher.WORKER: hashlib.sha256(texts[patcher.WORKER].encode()).hexdigest()},
    )
    with pytest.raises(RuntimeError, match="no complete"):
        patcher.inspect_patch(tmp_path, require=True)
    patcher.inspect_patch(tmp_path, apply=True)
    before = (tmp_path / patcher.WORKER).read_text()
    patcher.inspect_patch(tmp_path, require=True, apply=True)
    assert (tmp_path / patcher.WORKER).read_text() == before
    (tmp_path / patcher.WORKER).write_text(before + "# unknown server change\n")
    with pytest.raises(RuntimeError, match="unaudited"):
        patcher.inspect_patch(tmp_path, apply=True)
