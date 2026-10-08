"""Conversation prompts and JSON/safetensors trajectory IO."""

import hashlib
import json
from pathlib import Path


def prompt_messages(record: dict) -> list[dict]:
    source = record.get("messages", record.get("conversations"))
    if not isinstance(source, list) or not source:
        raise ValueError("each sample must contain messages or conversations")
    messages = []
    for message in source:
        role = message.get("role", message.get("from"))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role!r}")
        value = {k: v for k, v in message.items() if k not in {"from", "value"}}
        value.update(
            role=role, content=message.get("content", message.get("value", ""))
        )
        messages.append(value)
    # Reject ambiguous records instead of leaking turns after the held-out answer.
    if messages[-1]["role"] != "assistant" or len(messages) < 2:
        raise ValueError(
            "training conversations must end with a held-out assistant message"
        )
    return messages[:-1]


def load_prompts(path, tokenizer, max_length, template_kwargs=None):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        records = (
            [json.loads(line) for line in stream if line.strip()]
            if path.suffix == ".jsonl"
            else json.load(stream)
        )
    if not isinstance(records, list):
        raise ValueError("dataset must be a JSON array or JSONL")
    kwargs = dict(template_kwargs or {})
    owned = {
        "tokenize",
        "add_generation_prompt",
        "return_dict",
        "return_tensors",
        "padding",
        "truncation",
        "max_length",
        "continue_final_message",
        "return_assistant_tokens_mask",
    }
    if owned & kwargs.keys():
        raise ValueError(
            "tokenization and generation-prompt options are owned by on-policy training"
        )
    prompts = []
    for index, record in enumerate(records):
        ids = tokenizer.apply_chat_template(
            prompt_messages(record),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **kwargs,
        )
        if not isinstance(ids, list) or any(
            not isinstance(token, int) for token in ids
        ):
            raise ValueError("chat template must return a single list of token IDs")
        if 0 < len(ids) <= max_length:
            prompts.append({"sample_id": str(index), "input_ids": list(ids)})
    if not prompts:
        raise ValueError("no valid prompts within data.max_prompt_length")
    return prompts


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def trace_stem(root, request_id):
    # COSEC: request IDs never become path components, including IDs from SGLang.
    return Path(root) / "trajectories" / hashlib.sha256(request_id.encode()).hexdigest()


def load_trace(root, request_id, version, *, metadata_only=False):
    from safetensors.torch import load_file

    stem = trace_stem(root, request_id)
    metadata = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["weight_version"] != version or metadata["request_id"] != request_id:
        raise ValueError("stale or mismatched on-policy trajectory")
    if metadata_only:
        return metadata, None
    tensors = load_file(str(stem.with_suffix(".safetensors")))
    validate_trace(metadata, tensors)
    return metadata, tensors


def validate_trace(trace, tensors):
    """Validate provenance and the rejected-branch chain before any gradient."""
    if "blocks" not in trace:
        raise ValueError("trajectory has no blocks")
    context = list(trace["prompt_ids"])
    anchor = trace["initial_token"]
    for index, block in enumerate(trace["blocks"]):
        proposal = block["proposal"]
        mask = block["valid_mask"]
        if block["weight_version"] != trace["weight_version"]:
            raise ValueError("mixed policy versions in one trajectory")
        if (
            block["context_ids"] != context
            or block["context_length"] != len(context)
            or block["anchor"] != anchor
        ):
            raise ValueError(
                "trajectory context/anchor does not follow real acceptance"
            )
        if len(mask) != len(proposal) or any(
            mask[i] and not mask[i - 1] for i in range(1, len(mask))
        ):
            raise ValueError("invalid candidate mask")
        accepted = block["accepted_count"]
        if not 0 <= accepted <= len(proposal):
            raise ValueError("invalid acceptance count")
        p, q = tensors[f"p_{index}"], tensors[f"q_{index}"]
        if p.shape != q.shape or p.ndim != 2 or p.shape[0] != len(proposal):
            raise ValueError("trace probabilities do not cover the complete proposal")
        if tensors["context_hidden"].shape[0] < len(context):
            raise ValueError("target context features were truncated")
        context.extend([anchor, *proposal[:accepted]])
        anchor = block["correction_or_bonus"]
    if trace["sequence_ids"] != trace["prompt_ids"] + trace["output_ids"]:
        raise ValueError("incorrect final sequence")
    if trace["sequence_ids"] != (context + [anchor])[: len(trace["sequence_ids"])]:
        raise ValueError("final sequence differs from accepted/corrected trajectory")
