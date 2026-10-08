"""Opt-in hooks for the checked-in SGLang v0.5.18 patch.

The native engine still owns proposing, verifying, rejecting and correcting.
Only the eager, static, one-request/TP=1 path is enabled for this first gate.
The frozen teacher probabilities are computed in the engine on every complete
candidate prefix and retained for replay, avoiding a second target allocation.
"""

import os
from pathlib import Path

import torch

from .data import atomic_json, trace_stem
from .objective import candidate_mask, draft_distribution


def _root():
    value = os.environ.get("SPECFORGE_ON_POLICY_ROOT")
    return Path(value).resolve() if value else None


def prefill(worker, batch, hidden_states, next_token_ids):
    if _root() is None:
        return
    if len(batch.reqs) != 1 or batch.prefix_lens != [0]:
        raise RuntimeError("on-policy capture requires one complete uncached prefill")
    req = batch.reqs[0]
    prompt = list(req.origin_input_ids)
    if hidden_states.shape[0] != len(prompt):
        raise RuntimeError("chunked/projected target capture cannot be replayed")
    if getattr(worker, "_specforge_trace", None) is not None:
        raise RuntimeError("previous on-policy trajectory has not been drained")
    worker._specforge_trace = {
        "request_id": req.rid,
        "weight_version": worker._specforge_weight_version,
        "worker_id": os.environ.get("SPECFORGE_ON_POLICY_WORKER"),
        "engine_seed": getattr(
            getattr(worker, "server_args", None), "random_seed", None
        ),
        "prompt_ids": prompt,
        "initial_token": int(next_token_ids[0]),
        "blocks": [],
    }
    worker._specforge_context = list(prompt)
    worker._specforge_hidden = [hidden_states.detach().cpu().clone()]
    worker._specforge_tensors = {}


def before_verify_accept(worker, batch, proposal, logits_output, layout, run_compact):
    if _root() is None:
        return None
    from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs

    trace = worker._specforge_trace
    if len(batch.reqs) != 1 or run_compact or layout is not None:
        raise RuntimeError("on-policy requires static full-width verification")
    req = batch.reqs[0]
    if req.rid != trace["request_id"]:
        raise RuntimeError("request changed during trajectory capture")
    params = req.sampling_params
    if params.temperature < 1e-5 or params.top_k in (0, 1):
        raise RuntimeError("greedy on-policy TV is unsupported")
    tokens = proposal.draft_block.draft_tokens[0].tolist()
    anchor = int(proposal.draft_block_ids[0, 0])
    context_length = int(batch.seq_lens[0])
    if context_length != len(worker._specforge_context):
        raise RuntimeError("recorded committed context differs from draft KV length")
    # Includes EOS at a valid candidate; rejection does not shorten this mask.
    remaining = params.max_new_tokens - (context_length - len(trace["prompt_ids"]) + 1)
    stop_ids = set()
    if not params.ignore_eos:
        stop_ids.update(params.stop_token_ids or ())
        stop_ids.update(getattr(req, "eos_token_ids", None) or ())
        tokenizer = getattr(req, "tokenizer", None)
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            stop_ids.add(eos)
        stop_ids.update(getattr(tokenizer, "additional_stop_token_ids", None) or ())
    block = {
        "context_length": context_length,
        "context_ids": list(worker._specforge_context),
        "anchor": anchor,
        "proposal": tokens,
        "valid_mask": candidate_mask(tokens, max(0, remaining), stop_ids),
        "weight_version": trace["weight_version"],
        "sampling": {
            "temperature": float(params.temperature),
            "top_k": int(params.top_k),
            "top_p": float(params.top_p),
            "ignore_eos": bool(params.ignore_eos),
            "stop_token_ids": sorted(stop_ids),
            "max_new_tokens": int(params.max_new_tokens),
            "draft_filter": "temperature_only",
            "rejection_rule": "sglang_dspark_v0.5.18",
        },
    }
    # Use SGLang's own normalization, including its top-k-first/top-p behavior.
    # The no-filter native path uses FP32 SoftmaxTemp; the filtered path below
    # is exactly the helper used by AcceptSampling.
    info = batch.sampling_info
    width = worker.verify_num_draft_tokens
    if info.need_top_k_sampling or info.need_top_p_sampling:
        p = build_dflash_verify_target_probs(
            next_token_logits=logits_output.next_token_logits,
            sampling_info=info,
            draft_token_num=width,
            bs=1,
        )[0, : len(tokens)]
    else:
        p = torch.softmax(
            logits_output.next_token_logits.float() / params.temperature, -1
        )[: len(tokens)]
    q = draft_distribution(proposal.draft_block.corrected_logits[0], params.temperature)
    index = len(trace["blocks"])
    worker._specforge_tensors[f"p_{index}"] = p.detach().float().cpu().clone()
    worker._specforge_tensors[f"q_{index}"] = q.detach().cpu().clone()
    return block


def after_verify_accept(worker, block, verify_ids, logits_output, accept):
    if block is None:
        return
    accepted = int(accept.correct_len[0])
    committed = int(accept.commit_lens[0])
    if committed != accepted + 1:
        raise RuntimeError("unexpected static DSpark commit length")
    block.update(
        accepted_count=accepted,
        correction_or_bonus=int(accept.bonus[0]),
        token_kind="bonus" if accepted == len(block["proposal"]) else "correction",
    )
    worker._specforge_trace["blocks"].append(block)
    worker._specforge_context.extend(verify_ids[0, :committed].tolist())
    worker._specforge_hidden.append(
        logits_output.hidden_states[:committed].detach().cpu().clone()
    )


def _expected_serving_weights(model, state):
    """Pack trainer projections exactly as the TP=1 SGLang loader does."""
    expected = {}
    consumed = set()
    for name, value in state.items():
        if name in consumed:
            continue
        for marker, parts, replacement in (
            (".q_proj.", ("q_proj", "k_proj", "v_proj"), "qkv_proj"),
            (".gate_proj.", ("gate_proj", "up_proj"), "gate_up_proj"),
        ):
            if marker in name:
                names = [name.replace(marker, f".{part}.") for part in parts]
                if all(key in state for key in names):
                    expected[name.replace(marker, f".{replacement}.")] = torch.cat(
                        [state[key] for key in names]
                    )
                    consumed.update(names)
                    break
        else:
            if any(part in name for part in (".k_proj.", ".v_proj.", ".up_proj.")):
                continue
            expected[name] = value
            consumed.add(name)
    if consumed != set(state):
        raise ValueError("incomplete draft projection weights")
    parameters = dict(model.named_parameters())
    draft_names = {
        name
        for name in parameters
        if not name.startswith(("embed_tokens.", "lm_head."))
    }
    if set(expected) != draft_names:
        raise ValueError(f"draft serving key mismatch: {set(expected) ^ draft_names}")
    for name, value in expected.items():
        if value.shape != parameters[name].shape:
            raise ValueError(f"draft serving shape mismatch: {name}")
    return expected


def scheduler_rpc(scheduler, operation, version, request_id=None, output_ids=None):
    root = _root()
    if root is None:
        raise RuntimeError("on-policy RPC is disabled")
    worker = scheduler.draft_worker
    if not scheduler.is_fully_idle():
        raise RuntimeError("on-policy RPC requires an idle engine")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError("invalid weight version")
    if operation == "sync":
        from safetensors.torch import load_file

        previous = getattr(worker, "_specforge_weight_version", -1)
        if (
            version != previous + 1
            or getattr(worker, "_specforge_trace", None) is not None
        ):
            raise RuntimeError("nonsequential weight update or undrained trajectory")
        # COSEC: RPC chooses only a numeric version under the configured run root.
        path = (root / "weights" / f"{version:08d}").resolve()
        if not path.is_relative_to(root):
            raise ValueError("weight path escapes run root")
        state = load_file(str(path / "model.safetensors"))
        # Static SGLang normally omits this frozen head. Retain and synchronize
        # it too so the exported checkpoint remains complete and compatible.
        if (
            worker.draft_model.confidence_head is None
            and "confidence_head.proj.weight" in state
        ):
            from sglang.srt.models.dspark import DSparkConfidenceHead

            hidden_size = worker.draft_model.config.hidden_size
            worker.draft_model.confidence_head = (
                DSparkConfidenceHead(
                    hidden_size=hidden_size,
                    markov_rank=worker.draft_model.markov_head.markov_rank,
                    with_markov=state["confidence_head.proj.weight"].shape[1]
                    != hidden_size,
                )
                .to(device=worker.device, dtype=worker.draft_model.fc.weight.dtype)
                .eval()
            )
        expected = _expected_serving_weights(worker.draft_model, state)
        worker.draft_model.load_weights(state.items())
        for name, parameter in worker.draft_model.named_parameters():
            if name in expected and not torch.equal(
                parameter.detach().cpu(), expected[name].to(parameter.dtype)
            ):
                raise RuntimeError(f"draft synchronization did not load {name}")
        worker.draft_model._stacked_ctx_kv_cache = False
        worker.draft_model._fused_kv_write_cache = None
        if not scheduler.flush_cache():
            raise RuntimeError("failed to invalidate draft KV cache")
        worker._specforge_weight_version = version
        torch.cuda.synchronize()
        atomic_json(
            root / "acks" / f"worker-{os.environ['SPECFORGE_ON_POLICY_WORKER']}.json",
            {"weight_version": version, "keys": len(state), "cache_invalidated": True},
        )
    elif operation == "drain":
        from safetensors.torch import save_file

        trace = worker._specforge_trace
        if trace["request_id"] != request_id or trace["weight_version"] != version:
            raise RuntimeError("wrong trajectory/version requested")
        trace["output_ids"] = output_ids
        trace["sequence_ids"] = trace["prompt_ids"] + output_ids
        sequence = trace["sequence_ids"]
        for block in trace["blocks"]:
            anchor_position = block["context_length"]
            terminal_anchor = (
                not block["sampling"]["ignore_eos"]
                and block["anchor"] in block["sampling"]["stop_token_ids"]
            )
            # The overlap scheduler may launch a block before learning that
            # the previous output contained EOS. Preserve it, but exclude it.
            if (
                terminal_anchor
                or anchor_position >= len(sequence)
                or sequence[:anchor_position] != block["context_ids"]
                or sequence[anchor_position] != block["anchor"]
            ):
                block["valid_mask"] = [False] * len(block["proposal"])
            block["actual_accepted_count"] = min(
                block["accepted_count"],
                sum(block["valid_mask"]),
                max(0, len(sequence) - anchor_position - 1),
            )
            bonus_position = anchor_position + block["accepted_count"] + 1
            block["correction_or_bonus_emitted"] = (
                any(block["valid_mask"])
                and bonus_position < len(sequence)
                and sequence[bonus_position] == block["correction_or_bonus"]
            )
        tensors = worker._specforge_tensors
        tensors["context_hidden"] = torch.cat(worker._specforge_hidden)
        stem = trace_stem(root, request_id)
        stem.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(stem.with_suffix(".safetensors")))
        atomic_json(stem.with_suffix(".json"), trace)
        worker._specforge_trace = None
        worker._specforge_hidden = []
        worker._specforge_tensors = {}
    else:
        raise ValueError(f"unknown on-policy operation: {operation}")
