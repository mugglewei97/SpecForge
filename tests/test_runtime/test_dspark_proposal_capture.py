"""Verification capture, bounded reservoir, identity checks and IPC ownership."""

import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch

from specforge.data.dspark_replay import ReplayCollator
from specforge.inference import dspark_replay_patch as patcher
from specforge.inference import dspark_replay_server as server
from tests.test_runtime.test_dspark_replay_ipc import capture_fixture
from tests.test_runtime.test_dspark_replay_server import scheduler_fixture


class Processor(torch.nn.Module):
    def forward(
        self, input_ids, hidden_states, lm_head, logits_metadata, aux_hidden_states=None
    ):
        return SimpleNamespace(
            next_token_logits=lm_head(hidden_states), hidden_states=aux_hidden_states
        )


def begin_fixture(tmp_path, monkeypatch, budget=2):
    scheduler, rpc, sync = scheduler_fixture(tmp_path, monkeypatch)
    sync.update(tv_prefix_mode="proposal", anchor_budget=budget, eos_token_ids=[9])
    rpc("sync", **sync)
    worker = scheduler.draft_worker
    worker.target_worker = scheduler.tp_worker
    scheduler.tp_worker.model_runner.model.logits_processor = Processor()
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.logits_processor",
        SimpleNamespace(LogitsProcessor=Processor),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.speculative.dspark_components.dspark_verify",
        SimpleNamespace(verify_logits_adjustments_are_noop=lambda x: x is None),
    )
    rpc(
        "begin",
        draft_version=0,
        requests=[
            dict(
                sample_id="a",
                input_ids=[1, 2],
                sampling_config=dict(
                    temperature=0.8,
                    top_p=1.0,
                    top_k=-1,
                    max_new_tokens=100,
                    speculative={},
                ),
            )
        ],
    )
    req = SimpleNamespace(rid="a", output_ids_through_stop=[3], finished=lambda: True)
    batch = SimpleNamespace(reqs=[req], has_grammar=False, sampling_info=None)
    server.register_requests(worker, batch)
    return scheduler, worker, rpc, sync, batch, req


def verify(worker, batch, proposal, hidden, anchor, correction):
    server.begin_verification(worker, batch, False)
    processor = worker.target_worker.model_runner.model.logits_processor
    logits = processor(
        torch.tensor([3, *proposal]),
        hidden,
        torch.nn.Identity(),
        SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: True)),
        aux_hidden_states=torch.full((4, 20), -99.0),
    )
    reference_logits = logits.next_token_logits.clone()
    hidden.fill_(-111.0)  # later reuse must not mutate the captured states
    server.record_round(
        worker,
        batch,
        torch.tensor([anchor]),
        torch.tensor([proposal]),
        SimpleNamespace(
            correct_len=torch.tensor([0]),
            commit_lens=torch.tensor([1]),
            out_tokens=torch.tensor([[correction, 0, 0, 0]]),
        ),
        None,
    )
    return reference_logits


def test_final_norm_rows_exclude_bonus_and_survive_rejection_and_reuse(
    tmp_path, monkeypatch
):
    scheduler, worker, rpc, _, batch, req = begin_fixture(tmp_path, monkeypatch)
    hidden = torch.arange(12.0).reshape(4, 3)
    reference = verify(worker, batch, [4, 9, 6], hidden, 2, 8)
    req.output_ids_through_stop.append(8)
    record = rpc("finish")["records"][0]
    meta = record["round_metadata"][0]
    assert meta["proposal_ids"] == [4, 9, 6]
    assert meta["accepted_draft_tokens"] == 0
    assert meta["proposal_valid_mask"] == [True, True, False]
    assert worker._specforge_verify_hook is None
    cache = worker._specforge_replay["proposal_pending"]["a"]["cache"]
    torch.testing.assert_close(cache[0][1], reference[:3])
    assert cache[0][1].shape == (3, 3)  # not aux width 20; not bonus row 3
    data = ReplayCollator(
        draft_version=0,
        target_version="t",
        num_anchors=2,
        prefix_mode="proposal",
        block_size=3,
        eos_token_ids=[9],
    )([record])
    request = dict(
        input_ids=data["input_ids"].tolist(),
        proposal_selection=dict(
            sample_ids=["a"],
            round_ids=data["proposal_round_ids"].tolist(),
            anchors=data["anchor_positions"].tolist(),
        ),
    )
    result, consumed = server._proposal_capture_tensors(worker, request)
    assert consumed == ["a"]
    torch.testing.assert_close(result[0, 0], reference[:3])
    request["proposal_selection"]["anchors"] = [[1]]
    with pytest.raises(ValueError, match="anchor/round"):
        server._proposal_capture_tensors(worker, request)


def test_cached_qwen_verification_logits_match_explicit_proposal_prefix(
    tmp_path, monkeypatch
):
    from transformers import Qwen3Config, Qwen3Model

    _, worker, rpc, _, batch, req = begin_fixture(tmp_path, monkeypatch)
    torch.manual_seed(31)
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
    )
    target = Qwen3Model(config).eval().requires_grad_(False)
    lm_head = torch.nn.Linear(16, 16, bias=False).requires_grad_(False)
    with torch.no_grad():
        prefix = target(torch.tensor([[1, 2]]), use_cache=True)
        verification = target(
            torch.tensor([[3, 4, 5, 6]]),
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        explicit = target(torch.tensor([[1, 2, 3, 4, 5, 6]]), use_cache=False)
    verify(worker, batch, [4, 5, 6], verification.last_hidden_state[0], 2, 7)
    req.output_ids_through_stop.append(7)
    rpc("finish")
    hidden = worker._specforge_replay["proposal_pending"]["a"]["cache"][0][1]
    torch.testing.assert_close(
        lm_head(hidden),
        lm_head(explicit.last_hidden_state[0, 2:5]),
        rtol=1e-5,
        atol=1e-6,
    )
    assert not hidden.requires_grad


def test_reservoir_bounds_hidden_and_keeps_late_rounds(tmp_path, monkeypatch):
    _, worker, rpc, _, batch, req = begin_fixture(tmp_path, monkeypatch, budget=2)
    for index in range(30):
        verify(worker, batch, [4, 5, 6], torch.full((4, 3), float(index)), index + 2, 7)
        req.output_ids_through_stop.append(7)
        entry = worker._specforge_replay["requests"]["a"]
        assert len(entry["proposal_cache"]) <= 2
    record = rpc("finish")["records"][0]
    selected = record["selected_round_ids"]
    assert len(selected) == 2 and max(selected) > 2
    cache = worker._specforge_replay["proposal_pending"]["a"]["cache"]
    for index in selected:
        anchor, teacher = cache[index]
        assert anchor == index + 2
        torch.testing.assert_close(teacher, torch.full((3, 3), float(index)))


@pytest.mark.parametrize(
    "mode", ["compact", "grammar", "processor", "softcap", "scale"]
)
def test_unsupported_verification_fails_before_forward(tmp_path, monkeypatch, mode):
    _, worker, _, _, batch, _ = begin_fixture(tmp_path, monkeypatch)
    if mode == "grammar":
        batch.has_grammar = True
    elif mode == "processor":
        batch.sampling_info = object()
    elif mode == "softcap":
        worker.target_worker.model_runner.model.logits_processor.final_logit_softcapping = 1.0
    elif mode == "scale":
        worker.target_worker.model_runner.model.logits_processor.logit_scale = 2.0
    with pytest.raises(RuntimeError, match="proposal TV"):
        server.begin_verification(worker, batch, mode == "compact")


def test_unconsumed_cache_blocks_next_sync(tmp_path, monkeypatch):
    _, worker, rpc, sync, batch, req = begin_fixture(tmp_path, monkeypatch)
    verify(worker, batch, [4, 5, 6], torch.randn(4, 3), 2, 7)
    req.output_ids_through_stop.append(7)
    rpc("finish")
    with pytest.raises(RuntimeError, match="consumed"):
        rpc("sync", **{**sync, "draft_version": 1})


def test_proposal_ipc_clone_and_release(tmp_path, monkeypatch):
    engine, scheduler, _, source = capture_fixture(tmp_path, monkeypatch)
    engine.tv_prefix_mode = "proposal"
    source.pop("last_hidden_states")
    source["proposal_teacher_hidden"] = torch.randn(2, 2, 3, 3)
    tokens = torch.ones(2, 4, dtype=torch.long)
    result = engine.generate_dflash_data(tokens, tokens, tokens, proposal_selection={})
    assert result.last_hidden_states is None
    teacher = result.proposal_teacher_hidden
    torch.testing.assert_close(teacher, source["proposal_teacher_hidden"])
    source["proposal_teacher_hidden"].zero_()
    assert teacher.count_nonzero() > 0
    assert scheduler.draft_worker._specforge_capture_tensors is None


@pytest.mark.parametrize("transport", ["files", "cuda-ipc"])
def test_capture_consumes_selected_cache_with_one_context_forward(
    tmp_path, monkeypatch, transport
):
    from safetensors.torch import load_file

    scheduler, worker, rpc, sync, batch, req = begin_fixture(tmp_path, monkeypatch)
    expected = torch.randn(4, 3)
    reference = verify(worker, batch, [4, 5, 6], expected, 2, 7)
    req.output_ids_through_stop.append(7)
    rpc("finish")
    scheduler.tp_worker.model_runner.device = "cpu"
    calls = []

    class Backend:
        def __init__(self, runner):
            pass

        def extend_dflash(self, ids, attention, loss, **kwargs):
            calls.append(ids.clone())
            return None, [torch.ones(4, 6)], [torch.full((4, 3), -99.0)]

    monkeypatch.setitem(
        sys.modules,
        "specforge.inference.target_engine.sglang_backend.capture",
        SimpleNamespace(SGLangCaptureBackend=Backend),
    )
    monkeypatch.setitem(
        sys.modules,
        "specforge.inference.target_engine.sglang_backend.utils",
        SimpleNamespace(
            LogitsProcessorForEAGLE3=lambda child, flag: torch.nn.Identity()
        ),
    )
    monkeypatch.setattr(
        server, "export_tensor", lambda value: {"shape": list(value.shape)}
    )
    worker._specforge_replay["tensor_transport"] = transport
    result = rpc(
        "capture",
        draft_version=0,
        tensor_transport=transport,
        input_ids=[[1, 2, 3, 7]],
        attention_mask=[[1] * 4],
        loss_mask=[[0, 0, 1, 1]],
        proposal_selection=dict(sample_ids=["a"], round_ids=[[0]], anchors=[[2]]),
    )
    assert len(calls) == 1
    assert worker._specforge_replay["proposal_pending"] == {}
    if transport == "files":
        tensors = load_file(str(tmp_path / "capture.safetensors"))
    else:
        tensors = worker._specforge_capture_tensors
        assert set(result["capture_handles"]) == set(tensors)
    assert set(tensors) == {"hidden_states", "proposal_teacher_hidden"}
    torch.testing.assert_close(tensors["proposal_teacher_hidden"][0, 0], reference[:3])
    assert isinstance(
        scheduler.tp_worker.model_runner.model.logits_processor, Processor
    )
    if transport == "cuda-ipc":
        rpc("release_capture", draft_version=0)
    assert rpc("sync", **{**sync, "draft_version": 1})["draft_version"] == 1


def test_protocol_one_patch_upgrades_atomically(tmp_path, monkeypatch):
    texts = {
        patcher.WORKER: "\n".join(old for old, _ in patcher.EDITS[patcher.WORKER]),
        patcher.SCHEDULER: "def is_fully_idle\ndef handle_rpc_request\n"
        + patcher.EDITS[patcher.SCHEDULER][0][0],
        "srt/models/dspark.py": "class Qwen3DSparkModel\n_fused_kv_write_cache\n_stacked_ctx_kv_cache",
        "srt/entrypoints/engine.py": "def collective_rpc\ndef generate",
    }
    for name, text in texts.items():
        for old, new in patcher.LEGACY_EDITS.get(name, []):
            text = text.replace(old, new)
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
    patcher.inspect_patch(tmp_path, require=True)
    assert "begin_verification" in (tmp_path / patcher.WORKER).read_text()
    assert "protocol 1" not in (tmp_path / patcher.WORKER).read_text()
