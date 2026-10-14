"""SGLang 0.5.18 scheduler hooks for synchronous, TP=1 DSpark replay.

Invoked only by the small source patch, never by a prefill capture observer.
The owned Engine uses local RPC and a private exchange directory, not HTTP.
"""

import json
import os
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from specforge.inference.dspark_replay_ipc import export_tensor, import_tensor

PROTOCOL = 2


def _root():
    path = os.environ.get("SPECFORGE_DSPARK_REPLAY_DIR")
    if not path:
        raise RuntimeError("replay RPC is disabled outside an owned training Engine")
    return Path(path).resolve(strict=True)


def _path(name):
    # COSEC: RPC callers can only name fixed exchange files in the private root.
    allowed = {
        "request.json",
        "response.json",
        "weights.safetensors",
        "capture.safetensors",
    }
    if name not in allowed:
        raise ValueError("invalid replay exchange filename")
    result = _root() / name
    if result.is_symlink():
        raise ValueError("replay exchange files must not be symlinks")
    return result


def register_requests(worker, batch):
    state = getattr(worker, "_specforge_replay", None)
    if state is None:
        return
    if state["phase"] != "rollout":
        raise RuntimeError("draft generation attempted outside rollout phase")
    for req in batch.reqs:
        if req.rid not in state["requests"]:
            raise RuntimeError("unregistered request entered synchronous rollout")
        entry = state["requests"][req.rid]
        if entry.get("req") is not None and entry["req"] is not req:
            raise RuntimeError("request identity changed during rollout")
        entry["req"] = req


def begin_verification(worker, batch, run_compact):
    """Arm a one-forward hook on the LM-head input, never on aux hidden output."""
    state = getattr(worker, "_specforge_replay", None)
    if state is None or state.get("tv_prefix_mode") != "proposal":
        return
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        verify_logits_adjustments_are_noop,
    )

    if (
        state["phase"] != "rollout"
        or run_compact
        or worker._verify_planner.mode_value != "static"
        or batch.has_grammar
        or not verify_logits_adjustments_are_noop(batch.sampling_info)
    ):
        raise RuntimeError(
            "proposal TV requires fixed static verification without logits processors or grammar"
        )
    processors = [
        m
        for m in worker.target_worker.model_runner.model.modules()
        if isinstance(m, LogitsProcessor)
    ]
    if len(processors) != 1:
        raise RuntimeError("proposal TV requires one audited target logits processor")
    processor = processors[0]
    if getattr(processor, "logit_scale", None) not in (None, 1.0) or getattr(
        processor, "final_logit_softcapping", None
    ):
        raise RuntimeError(
            "proposal TV cannot reconstruct scaled or softcapped target logits"
        )
    if getattr(worker, "_specforge_verify_hook", None) is not None:
        raise RuntimeError("previous proposal verification was not consumed")
    worker._specforge_verify_final = None

    def capture(module, args, kwargs):
        hidden = args[1] if len(args) > 1 else kwargs["hidden_states"]
        metadata = args[3] if len(args) > 3 else kwargs["logits_metadata"]
        if (
            not metadata.forward_mode.is_target_verify()
            or worker._specforge_verify_final is not None
        ):
            raise RuntimeError("unexpected target forward during proposal verification")
        expected = len(batch.reqs) * (worker.gamma + 1)
        if hidden.ndim != 2 or hidden.size(0) != expected:
            raise RuntimeError("proposal verification final hidden rows are misaligned")
        # The processor may release/reuse its input. Own these rows before that
        # happens; record_round retains only reservoir-selected blocks, without bonus.
        worker._specforge_verify_final = (
            hidden.detach()
            .reshape(len(batch.reqs), worker.gamma + 1, -1)[:, : worker.gamma]
            .clone()
        )

    worker._specforge_verify_hook = processor.register_forward_pre_hook(
        capture, with_kwargs=True
    )


def record_round(worker, batch, prefix_lens, draft_tokens, accept, layout):
    state = getattr(worker, "_specforge_replay", None)
    if state is None:
        return
    if state["phase"] != "rollout":
        raise RuntimeError("verify outside rollout phase")
    # Read actual verify outcomes. Do not reconstruct positions from aggregates.
    anchors = prefix_lens.detach().cpu().tolist()
    accepted = accept.correct_len.detach().cpu().tolist()
    commits = accept.commit_lens.detach().cpu().tolist()
    tokens = accept.out_tokens.detach().cpu().tolist()
    verify_lengths = (
        layout.verify_lens.detach().cpu().tolist()
        if layout is not None
        else [draft_tokens.shape[1] + 1] * len(anchors)
    )
    proposal_mode = state.get("tv_prefix_mode") == "proposal"
    teacher = None
    if proposal_mode:
        hook = getattr(worker, "_specforge_verify_hook", None)
        if hook is not None:
            hook.remove()
            worker._specforge_verify_hook = None
        teacher = getattr(worker, "_specforge_verify_final", None)
        worker._specforge_verify_final = None
        if teacher is None or teacher.shape[:2] != draft_tokens.shape:
            raise RuntimeError("proposal verification final hidden was not captured")
        proposal_ids = draft_tokens.detach().cpu().tolist()
    for i, req in enumerate(batch.reqs):
        entry = state["requests"][req.rid]
        if entry["draft_version"] != state["version"]:
            raise RuntimeError("draft changed during a sample")
        entry["rounds"].append(
            dict(
                anchor=int(anchors[i]),
                draft_length=int(draft_tokens.shape[1]),
                verified_draft_tokens=int(verify_lengths[i]) - 1,
                accepted_draft_tokens=int(accepted[i]),
                verifier_committed_tokens=int(commits[i]),
                emitted_tokens=tokens[i][: int(commits[i])],
                root_tokens=0,
            )
        )
        if proposal_mode:
            meta = entry["rounds"][-1]
            round_id = len(entry["rounds"]) - 1
            remaining = max(
                0,
                len(entry["input_ids"])
                + entry["sampling_config"]["max_new_tokens"]
                - int(anchors[i])
                - 1,
            )
            valid_length = min(
                int(verify_lengths[i]) - 1, remaining, len(proposal_ids[i])
            )
            for pos, token in enumerate(proposal_ids[i][:valid_length]):
                if token in state["eos_token_ids"]:
                    valid_length = pos + 1
                    break
            meta.update(
                round_id=round_id,
                proposal_ids=proposal_ids[i],
                proposal_valid_mask=[
                    j < valid_length for j in range(len(proposal_ids[i]))
                ],
                remaining_budget=remaining,
            )
            if valid_length:
                entry["proposal_seen"] += 1
                cache = entry["proposal_cache"]
                slot = (
                    len(cache)
                    if len(cache) < state["anchor_budget"]
                    else state["rng"].randrange(entry["proposal_seen"])
                )
                if slot < state["anchor_budget"]:
                    # Clone each selected row separately so evicting a round
                    # releases its storage, even when other requests keep theirs.
                    item = (round_id, teacher[i].clone())
                    if slot == len(cache):
                        cache.append(item)
                    else:
                        cache[slot] = item


def _proposal_capture_tensors(worker, request):
    """Gather the exact selected IDs once; consume buffers after successful transfer."""
    state = worker._specforge_replay
    selection = request.get("proposal_selection")
    if not isinstance(selection, dict) or set(selection) != {
        "sample_ids",
        "round_ids",
        "anchors",
    }:
        raise ValueError(
            "proposal capture requires sample, round and anchor identities"
        )
    samples, rounds, anchors = (
        selection[k] for k in ("sample_ids", "round_ids", "anchors")
    )
    if (
        len(samples) != len(request["input_ids"])
        or len(rounds) != len(samples)
        or len(anchors) != len(samples)
        or len(set(samples)) != len(samples)
    ):
        raise ValueError("misaligned proposal capture samples")
    template = state["proposal_template"]
    width = len(rounds[0]) if rounds else 0
    if not 1 <= width <= state["anchor_budget"]:
        raise ValueError("invalid proposal capture block count")
    result = template.new_zeros((len(samples), width, worker.gamma, template.size(-1)))
    consumed = []
    for row, sample in enumerate(samples):
        entry = state["proposal_pending"].get(sample)
        if entry is None or len(rounds[row]) != width or len(anchors[row]) != width:
            raise ValueError("unknown or misaligned proposal capture sample")
        if request["input_ids"][row][: len(entry["input_ids"])] != entry["input_ids"]:
            raise ValueError("proposal capture sequence differs from rollout")
        seen = set()
        for col, (round_id, anchor) in enumerate(zip(rounds[row], anchors[row])):
            if round_id == -1:
                continue
            if (
                type(round_id) is not int
                or round_id not in entry["cache"]
                or round_id in seen
            ):
                raise ValueError(
                    "unknown, duplicated or already consumed proposal round"
                )
            expected_anchor, hidden = entry["cache"][round_id]
            if anchor != expected_anchor:
                raise ValueError("proposal capture anchor/round mismatch")
            result[row, col].copy_(hidden)
            seen.add(round_id)
        if seen != set(entry["cache"]):
            raise ValueError(
                "proposal capture must include every selected round exactly once"
            )
        consumed.append(sample)
    return result, consumed


def _mapped_weights(model, weights):
    """Exact unquantized TP=1 mapping; reject every unsupported/missing tensor.

    Copy fused QKV/MLP parameters directly and verify readback. Generic SGLang
    loaders may silently ignore unknown names, which is unsafe for this loop.
    """
    params = dict(model.named_parameters())
    _skip_prefixes = ("embed_tokens.", "lm_head.", "confidence_head.")
    packed = {}
    for name, value in weights.items():
        if any(name.startswith(p) for p in _skip_prefixes):
            continue
        mapped, part = name, None
        for fused, source, index in (
            ("qkv_proj", "q_proj", 0),
            ("qkv_proj", "k_proj", 1),
            ("qkv_proj", "v_proj", 2),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ):
            if f".{source}." in name:
                mapped, part = name.replace(f".{source}.", f".{fused}."), index
                break
        if mapped not in params:
            raise ValueError(f"unmapped DSpark weight: {name}")
        packed.setdefault(mapped, {})[part] = value
    expected = {n for n in params if not any(n.startswith(p) for p in _skip_prefixes)}
    if set(packed) != expected:
        raise ValueError(
            f"incomplete draft state: missing={sorted(expected - packed.keys())}, extra={sorted(packed.keys() - expected)}"
        )
    result = {}
    for name, pieces in packed.items():
        if None in pieces:
            if len(pieces) != 1:
                raise ValueError(f"duplicate fused weight {name}")
            value = pieces[None]
        else:
            count = 3 if ".qkv_proj." in name else 2
            if set(pieces) != set(range(count)):
                raise ValueError(f"missing fused slices for {name}")
            value = torch.cat([pieces[i] for i in range(count)], dim=0)
        if value.shape != params[name].shape or value.dtype != params[name].dtype:
            raise ValueError(f"draft tensor shape/dtype mismatch: {name}")
        result[name] = value
    return result


def _capture(scheduler, request):
    from specforge.inference.target_engine.sglang_backend.capture import (
        SGLangCaptureBackend,
    )
    from specforge.inference.target_engine.sglang_backend.utils import (
        LogitsProcessorForEAGLE3,
    )
    from sglang.srt.layers.logits_processor import LogitsProcessor

    runner = scheduler.tp_worker.model_runner
    backend = SGLangCaptureBackend(runner)
    replacements = []
    # Temporary capture-only wrappers: speculative sampling uses the original
    # processor again immediately after the one complete-sequence forward.
    for module in list(runner.model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LogitsProcessor):
                replacements.append((module, name, child))
                setattr(module, name, LogitsProcessorForEAGLE3(child, False))
    if not replacements:
        raise RuntimeError("unsupported target logits processor for replay capture")
    try:
        ids = torch.tensor(request["input_ids"], dtype=torch.long, device=runner.device)
        attention = torch.tensor(request["attention_mask"], device=runner.device)
        loss = torch.tensor(request["loss_mask"], device=runner.device)
        _, hidden, final = backend.extend_dflash(
            ids, attention, loss, return_last_hidden_states=True
        )
        if any(h is None for h in final):
            raise RuntimeError("target capture did not return final hidden states")
        worker = scheduler.draft_worker
        state = worker._specforge_replay
        tensors = {"hidden_states": torch.stack(hidden).contiguous()}
        consumed = []
        if state.get("tv_prefix_mode") == "proposal":
            proposal_hidden, consumed = _proposal_capture_tensors(worker, request)
            tensors["proposal_teacher_hidden"] = proposal_hidden
        else:
            tensors["last_hidden_states"] = torch.stack(final).contiguous()
        if request.get("tensor_transport", "files") == "cuda-ipc":
            # Sender ownership extends through the trainer's clone + stream
            # synchronization. Never release merely because capture RPC ends.
            scheduler.draft_worker._specforge_capture_tensors = tensors
            handles = {name: export_tensor(value) for name, value in tensors.items()}
        else:
            save_file(
                {name: value.cpu() for name, value in tensors.items()},
                str(_path("capture.safetensors")),
            )
            handles = None
        for sample in consumed:
            del state["proposal_pending"][sample]
        return handles
    finally:
        for module, name, child in replacements:
            setattr(module, name, child)
        if not scheduler.flush_cache():
            raise RuntimeError("cannot flush target/draft pools after capture")


@torch.no_grad()
def dispatch(scheduler):
    """No RPC payload paths or arbitrary method dispatch; errors poison state."""
    if not scheduler.is_fully_idle():
        raise RuntimeError("replay RPC requires all generation requests to finish")
    request = json.loads(_path("request.json").read_text())
    worker = scheduler.draft_worker
    if type(worker).__name__ != "DSparkWorkerV2":
        raise RuntimeError("replay requires the audited DSparkWorkerV2")
    if worker._simulate_acc_len > 0:
        raise RuntimeError("real rollout forbids SGLANG_SIMULATE_ACC_LEN")
    state = getattr(worker, "_specforge_replay", None)
    try:
        action = request["action"]
        if state is not None and state["phase"] == "failed":
            raise RuntimeError("replay Engine is poisoned by a previous failure")
        if (
            getattr(worker, "_specforge_capture_tensors", None) is not None
            and action != "release_capture"
        ):
            raise RuntimeError(
                "previous CUDA capture must be released before more work"
            )
        if action == "sync":
            version = request["draft_version"]
            if (
                state is not None
                and state["target_version"] != request["target_version"]
            ):
                raise RuntimeError("target_version cannot change in a replay run")
            captured_layers = (
                scheduler.tp_worker.model_runner.spec_aux_config.dflash_target_layer_ids
            )
            if list(captured_layers) != request["capture_layer_ids"]:
                raise RuntimeError("SGLang and replay target capture layers differ")
            if state is not None and (
                state["phase"] != "replay" or version != state["version"] + 1
            ):
                raise RuntimeError(
                    "sync must follow one replay update, with version V+1"
                )
            if state is not None and state.get("proposal_pending"):
                raise RuntimeError(
                    "proposal caches must be consumed before the next update"
                )
            if not scheduler.flush_cache():
                raise RuntimeError("cache flush before draft sync failed")
            transport = request.get("tensor_transport", "files")
            if transport == "cuda-ipc":
                weights = {
                    name: import_tensor(handle, device=torch.cuda.current_device())
                    for name, handle in request["weight_handles"].items()
                }
            elif transport == "files":
                weights = load_file(str(_path("weights.safetensors")))
            else:
                raise ValueError("unknown replay tensor transport")
            if not set(request["trainable_names"]) <= weights.keys():
                raise RuntimeError("sync omitted trainable parameters")
            mapped = _mapped_weights(worker.draft_model, weights)
            params = dict(worker.draft_model.named_parameters())
            for name, value in mapped.items():
                value_on_device = value.to(params[name].device)
                params[name].copy_(value_on_device)
                if not torch.equal(params[name].detach(), value_on_device):
                    raise RuntimeError(f"draft weight readback failed: {name}")
            # These contain concatenated/copied projections, not live views.
            worker.draft_model._fused_kv_write_cache = None
            worker.draft_model._stacked_ctx_kv_cache = False
            if not scheduler.flush_cache():
                raise RuntimeError("cache flush after draft sync failed")
            torch.cuda.synchronize()
            state = dict(
                version=version,
                phase="ready",
                requests={},
                target_version=request["target_version"],
                tensor_transport=transport,
                tv_prefix_mode=request.get("tv_prefix_mode", "teacher-forced"),
                anchor_budget=request.get("anchor_budget", 1),
                eos_token_ids=tuple(request.get("eos_token_ids", ())),
                rng=random.Random(request.get("replay_seed", 0)),
            )
            if (
                state["tv_prefix_mode"] not in ("teacher-forced", "proposal")
                or state["anchor_budget"] < 1
            ):
                raise ValueError("invalid proposal replay configuration")
            worker._specforge_replay = state
            result = dict(
                protocol=PROTOCOL,
                draft_version=version,
                cache_flushed=True,
                loaded_names=sorted(weights),
                trainable_names=request["trainable_names"],
            )
        elif action == "begin":
            if (
                state is None
                or state["phase"] != "ready"
                or state["version"] != request["draft_version"]
            ):
                raise RuntimeError("rollout cannot start before successful weight sync")
            if state.get("tv_prefix_mode") == "proposal":
                for entry in request["requests"]:
                    sampling = entry["sampling_config"]
                    if (
                        not 1e-5 <= sampling["temperature"] < float("inf")
                        or sampling["top_p"] != 1
                        or sampling["top_k"] != -1
                    ):
                        raise ValueError(
                            "proposal rollout requires temperature >= 1e-5, top_p=1, top_k=-1"
                        )
            state["requests"] = {
                r["sample_id"]: dict(
                    r,
                    draft_version=state["version"],
                    rounds=[],
                    proposal_cache=[],
                    proposal_seen=0,
                )
                for r in request["requests"]
            }
            if len(state["requests"]) != len(request["requests"]):
                raise ValueError("duplicate rollout sample IDs")
            state["phase"] = "rollout"
            result = dict(draft_version=state["version"])
        elif action == "finish":
            if state is None or state["phase"] != "rollout":
                raise RuntimeError("finish requires a live rollout batch")
            records = []
            proposal_mode = state.get("tv_prefix_mode") == "proposal"
            state["proposal_pending"] = {}
            for entry in state["requests"].values():
                req = entry.get("req")
                if req is None or not req.finished():
                    raise RuntimeError("rollout request did not finish")
                ids = entry["input_ids"] + list(req.output_ids_through_stop)
                rounds, anchors = [], []
                for raw in entry["rounds"]:
                    meta = dict(raw)
                    anchor = meta.pop("anchor")
                    emitted = meta.pop("emitted_tokens")
                    committed = min(len(emitted), max(0, len(ids) - anchor - 1))
                    if (
                        anchor >= len(ids)
                        or ids[anchor + 1 : anchor + 1 + committed]
                        != emitted[:committed]
                    ):
                        raise RuntimeError(
                            "verify trace does not align with final raw tokens"
                        )
                    meta["committed_tokens"] = committed
                    # Root was already emitted by prefill/previous round. The
                    # final verifier token is correction, or bonus at full accept.
                    last_emitted = committed > meta["accepted_draft_tokens"]
                    full_accept = (
                        meta["accepted_draft_tokens"] == meta["verified_draft_tokens"]
                    )
                    meta["bonus_tokens"] = int(last_emitted and full_accept)
                    meta["correction_tokens"] = int(last_emitted and not full_accept)
                    anchors.append(anchor)
                    rounds.append(meta)
                sampling = dict(entry["sampling_config"])
                sampling["speculative"] = {
                    **sampling["speculative"],
                    "verify_mode": worker._verify_planner.mode_value,
                    "gamma": worker.gamma,
                    "num_draft_tokens": worker.verify_num_draft_tokens,
                }
                records.append(
                    dict(
                        sample_id=entry["sample_id"],
                        input_ids=ids,
                        prompt_length=len(entry["input_ids"]),
                        anchor_positions=anchors,
                        round_metadata=rounds,
                        draft_version=state["version"],
                        target_version=state["target_version"],
                        sampling_config=sampling,
                    )
                )
                if proposal_mode:
                    cache = {
                        index: (entry["rounds"][index]["anchor"], hidden)
                        for index, hidden in entry["proposal_cache"]
                    }
                    records[-1].update(
                        trace_version=2,
                        tv_prefix_mode="proposal",
                        selected_round_ids=sorted(cache),
                    )
                    state["proposal_pending"][entry["sample_id"]] = dict(
                        input_ids=ids, cache=cache
                    )
                    if cache:
                        # Zero rows hold dtype/device/width without retaining a block allocation.
                        value = next(iter(cache.values()))[1]
                        state["proposal_template"] = value.new_empty(
                            (0, value.size(-1))
                        )
            if proposal_mode and "proposal_template" not in state:
                # Empty local ranks must still participate in global normalization.
                param = next(scheduler.tp_worker.model_runner.model.parameters())
                hidden_size = scheduler.tp_worker.model_runner.model.config.hidden_size
                state["proposal_template"] = param.new_empty((0, hidden_size))
            state["requests"] = {}
            state["phase"] = "replay"
            result = dict(records=records, draft_version=state["version"])
        elif action == "capture":
            if (
                state is None
                or state["phase"] != "replay"
                or state["version"] != request["draft_version"]
            ):
                raise RuntimeError("capture requires current-version replay")
            if request.get("tensor_transport", "files") != state["tensor_transport"]:
                raise RuntimeError("capture transport differs from synchronized Engine")
            if not scheduler.flush_cache():
                raise RuntimeError("cache flush before target capture failed")
            handles = _capture(scheduler, request)
            result = dict(draft_version=state["version"])
            if handles is not None:
                result["capture_handles"] = handles
        elif action == "release_capture":
            if (
                state is None
                or state["phase"] != "replay"
                or state["version"] != request["draft_version"]
                or getattr(worker, "_specforge_capture_tensors", None) is None
            ):
                raise RuntimeError("release requires a pending current-version capture")
            worker._specforge_capture_tensors = None
            result = dict(draft_version=state["version"])
        else:
            raise ValueError("unknown replay RPC action")
        _path("response.json").write_text(json.dumps(result))
    except BaseException:
        worker._specforge_replay = dict(phase="failed")
        raise
