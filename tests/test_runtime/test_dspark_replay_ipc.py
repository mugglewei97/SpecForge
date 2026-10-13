"""Fixed CUDA IPC schema, owned-buffer lifetime, and optional real CUDA transfer."""

import json
from types import SimpleNamespace

import pytest
import torch

from specforge.inference import dspark_replay_ipc as ipc
from specforge.inference import dspark_replay_engine as client
from specforge.inference import dspark_replay_server as server
from tests.test_runtime.test_dspark_replay_server import scheduler_fixture


def descriptor():
    return dict(
        shape=[2, 3],
        dtype="torch.float32",
        device=0,
        handle="eA==",
        storage_bytes=24,
        storage_offset=0,
        ref_handle="eQ==",
        ref_offset=0,
        event_handle=None,
        event_sync=False,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"shape": [2, -3]},
        {"shape": [2, 4]},
        {"dtype": "torch.int64"},
        {"device": 1},
        {"storage_offset": -1},
        {"ref_offset": -1},
        {"handle": "not base64"},
        {"handle": None},
        {"event_sync": 1},
        {"function": "eval"},
    ],
)
def test_reject_invalid_handles_before_cuda_reconstruction(monkeypatch, change):
    monkeypatch.setattr(
        ipc, "rebuild_cuda_tensor", lambda *a: pytest.fail("must validate first")
    )
    with pytest.raises(ValueError):
        ipc.import_tensor({**descriptor(), **change}, device=0)


def test_fixed_reconstruction_is_detached_and_contiguous(monkeypatch):
    seen = []
    monkeypatch.setattr(ipc, "rebuild_cuda_tensor", lambda *a: seen.append(a))
    ipc.import_tensor(descriptor(), device=0)
    args = seen[0]
    assert args[:4] == (torch.Tensor, (2, 3), (3, 1), 0)
    assert args[4] is torch.storage.TypedStorage
    assert args[5:11] == (torch.float32, 0, b"x", 24, 0, False)


def test_cpu_export_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        ipc.export_tensor(torch.ones(2, 3))


def ipc_engine(tmp_path, monkeypatch):
    scheduler, rpc, sync = scheduler_fixture(tmp_path, monkeypatch)
    rpc("sync", **sync)
    scheduler.draft_worker._specforge_replay["phase"] = "replay"
    (tmp_path / "weights.safetensors").unlink()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    engine = client.SGLangReplayEngine(
        SimpleNamespace(
            replay_tensor_transport="cuda-ipc",
            rollout_target_version="t",
        )
    )
    engine.directory = tmp_path
    engine.version = 0
    engine.capture_layer_ids = [1]
    engine.engine = SimpleNamespace(
        collective_rpc=lambda method: server.dispatch(scheduler)
    )
    return engine, scheduler, rpc


def test_ipc_sync_no_tensor_files_and_target_untouched(tmp_path, monkeypatch):
    engine, scheduler, _ = ipc_engine(tmp_path, monkeypatch)
    target_before = {
        n: p.clone()
        for n, p in scheduler.tp_worker.model_runner.model.named_parameters()
    }
    weights = {
        n: torch.full_like(p, 5)
        for n, p in scheduler.draft_worker.draft_model.named_parameters()
    }
    by_id = {id(t): t for t in weights.values()}
    monkeypatch.setattr(client, "export_tensor", lambda t: dict(key=id(t)))
    monkeypatch.setattr(server, "import_tensor", lambda h, **k: by_id[h["key"]])
    assert engine.sync(weights, None, set(weights), 1) == 1
    assert not (tmp_path / "weights.safetensors").exists()
    for name, param in scheduler.draft_worker.draft_model.named_parameters():
        torch.testing.assert_close(param, weights[name])
    for name, param in scheduler.tp_worker.model_runner.model.named_parameters():
        torch.testing.assert_close(param, target_before[name])
    assert scheduler.draft_worker.draft_model._fused_kv_write_cache is None


def capture_fixture(tmp_path, monkeypatch):
    engine, scheduler, rpc = ipc_engine(tmp_path, monkeypatch)
    scheduler.draft_worker._specforge_replay["tensor_transport"] = "cuda-ipc"
    source = {
        name: torch.randn(2, 4, width)
        for name, width in (
            ("hidden_states", 12),
            ("last_hidden_states", 3),
        )
    }

    def capture(scheduler, request):
        scheduler.draft_worker._specforge_capture_tensors = source
        return {name: dict(key=name) for name in source}

    monkeypatch.setattr(server, "_capture", capture)
    monkeypatch.setattr(client, "import_tensor", lambda h, **k: source[h["key"]])
    return engine, scheduler, rpc, source


def test_capture_owned_clone_survives_release_without_tensor_file(
    tmp_path, monkeypatch
):
    engine, scheduler, _, source = capture_fixture(tmp_path, monkeypatch)
    tokens = torch.ones(2, 4, dtype=torch.long)
    result = engine.generate_dflash_data(tokens, tokens, tokens)
    assert scheduler.draft_worker._specforge_capture_tensors is None
    assert engine.capture_seconds > 0
    assert not (tmp_path / "capture.safetensors").exists()
    for name, value in source.items():
        actual = getattr(result, name)
        torch.testing.assert_close(actual, value)
        assert actual.data_ptr() != value.data_ptr()
        value.zero_()
        assert actual.count_nonzero() > 0


def test_failed_capture_clone_retains_sender_and_poison_client(tmp_path, monkeypatch):
    engine, scheduler, _, _ = capture_fixture(tmp_path, monkeypatch)

    def failed_import(*args, **kwargs):
        raise RuntimeError("simulated receiver failure")

    monkeypatch.setattr(client, "import_tensor", failed_import)
    tokens = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(RuntimeError, match="receiver failure"):
        engine.generate_dflash_data(tokens, tokens, tokens)
    assert scheduler.draft_worker._specforge_capture_tensors is not None
    assert engine.failed
    with pytest.raises(RuntimeError, match="refusing"):
        engine._rpc("begin")


def test_server_rejects_work_before_capture_release(tmp_path, monkeypatch):
    engine, scheduler, rpc, source = capture_fixture(tmp_path, monkeypatch)
    scheduler.draft_worker._specforge_capture_tensors = source
    with pytest.raises(RuntimeError, match="released"):
        rpc("begin", draft_version=engine.version, requests=[])
    assert scheduler.draft_worker._specforge_replay["phase"] == "failed"


def _cuda_receiver(connection, device):
    torch.cuda.set_device(device)
    handle = json.loads(connection.recv_bytes())
    view = ipc.import_tensor(handle, device=device)
    owned = view.clone()
    torch.cuda.synchronize()
    del view
    connection.send_bytes(b"copied")
    assert connection.recv_bytes() == b"mutated"
    torch.testing.assert_close(
        owned, torch.arange(24, device=device).reshape(4, 6).float()
    )
    connection.send_bytes(b"ok")
    connection.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA IPC on Linux")
def test_real_cuda_spawn_transfer_and_sender_lifetime():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    device = torch.cuda.current_device()
    source = torch.arange(24, device=device).reshape(4, 6).float()
    process = ctx.Process(target=_cuda_receiver, args=(child, device))
    process.start()
    try:
        parent.send_bytes(json.dumps(ipc.export_tensor(source)).encode())
        assert parent.poll(60), "receiver timed out importing IPC tensor"
        assert parent.recv_bytes() == b"copied"
        source.zero_()
        torch.cuda.synchronize()
        parent.send_bytes(b"mutated")
        assert parent.poll(60), "receiver timed out checking owned tensor"
        assert parent.recv_bytes() == b"ok"
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()
