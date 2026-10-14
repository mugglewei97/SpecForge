"""Raw-token DSpark rollout records; no tokenizer or chat template on replay."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from specforge.core.tv_acceptance import select_replay_anchors


def validate_trace(
    record,
    *,
    draft_version,
    target_version,
    vocab_size=None,
    prefix_mode="teacher-forced",
    block_size=None,
):
    required = {
        "sample_id",
        "input_ids",
        "prompt_length",
        "anchor_positions",
        "round_metadata",
        "draft_version",
        "target_version",
        "sampling_config",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"replay trace missing fields: {sorted(missing)}")
    if (
        record["draft_version"] != draft_version
        or record["target_version"] != target_version
    ):
        raise ValueError("replay draft_version/target_version mismatch")
    ids, prompt, anchors = (
        record["input_ids"],
        record["prompt_length"],
        record["anchor_positions"],
    )
    if (
        not isinstance(ids, list)
        or not ids
        or type(prompt) is not int
        or not 0 < prompt <= len(ids)
        or any(
            type(t) is not int or t < 0 or (vocab_size is not None and t >= vocab_size)
            for t in ids
        )
    ):
        raise ValueError("invalid raw token IDs or prompt_length")
    if not isinstance(anchors, list) or any(
        type(s) is not int or s < 0 or s >= len(ids) for s in anchors
    ):
        raise ValueError("invalid absolute anchor positions")
    if anchors != sorted(set(anchors)):
        raise ValueError("round anchors must be strictly increasing")
    rounds = record["round_metadata"]
    if not isinstance(rounds, list) or len(rounds) != len(anchors):
        raise ValueError("anchors and round_metadata must have identical lengths")
    for s, meta in zip(anchors, rounds):
        for key in ("draft_length", "accepted_draft_tokens", "committed_tokens"):
            if type(meta.get(key)) is not int or meta[key] < 0:
                raise ValueError(f"invalid round {key}")
        if (
            meta["accepted_draft_tokens"] > meta["draft_length"]
            or meta["committed_tokens"] > meta["accepted_draft_tokens"] + 1
            or s + meta["committed_tokens"] >= len(ids)
        ):
            raise ValueError("round counts are inconsistent with the final sequence")
    sampling = record["sampling_config"]
    if (
        not isinstance(sampling, dict)
        or not {"temperature", "top_p", "top_k", "speculative"} <= sampling.keys()
    ):
        raise ValueError("trace requires complete sampling_config")
    if prefix_mode == "proposal":
        _validate_proposal_trace(record, vocab_size=vocab_size, block_size=block_size)
    return record


def _validate_proposal_trace(record, *, vocab_size, block_size):
    if record.get("trace_version") != 2 or record.get("tv_prefix_mode") != "proposal":
        raise ValueError("proposal TV requires a version 2 proposal trace")
    selected = record.get("selected_round_ids")
    rounds = record["round_metadata"]
    if (
        not isinstance(selected, list)
        or any(type(i) is not int or not 0 <= i < len(rounds) for i in selected)
        or selected != sorted(set(selected))
    ):
        raise ValueError("invalid selected proposal round IDs")
    sampling = record["sampling_config"]
    if (
        not isinstance(sampling.get("temperature"), (int, float))
        or not 1e-5 <= sampling["temperature"] < float("inf")
        or sampling["top_p"] != 1
        or sampling["top_k"] != -1
        or type(sampling.get("max_new_tokens")) is not int
    ):
        raise ValueError("proposal trace has unsupported sampling settings")
    for index, (anchor, meta) in enumerate(zip(record["anchor_positions"], rounds)):
        ids, valid = meta.get("proposal_ids"), meta.get("proposal_valid_mask")
        width = meta["draft_length"]
        verified, remaining = (
            meta.get("verified_draft_tokens"),
            meta.get("remaining_budget"),
        )
        if (
            width < 1
            or (block_size is not None and width != block_size)
            or type(meta.get("round_id")) is not int
            or meta["round_id"] != index
            or not isinstance(ids, list)
            or len(ids) != width
            or any(
                type(t) is not int
                or t < 0
                or (vocab_size is not None and t >= vocab_size)
                for t in ids
            )
            or not isinstance(valid, list)
            or len(valid) != width
            or any(type(v) is not bool for v in valid)
            or valid != sorted(valid, reverse=True)
            or type(verified) is not int
            or not 0 <= verified <= width
            or meta["accepted_draft_tokens"] > verified
            or type(remaining) is not int
            or remaining < 0
            or remaining
            != max(0, record["prompt_length"] + sampling["max_new_tokens"] - anchor - 1)
            or sum(valid) > min(verified, remaining)
        ):
            raise ValueError("misaligned proposal IDs, validity, round ID or budget")


def validate_proposal_anchors(anchors, keep, attention_mask, budget):
    """Validate already selected blocks without consuming RNG or reordering rows."""
    if (
        anchors is None
        or keep is None
        or anchors.ndim != 2
        or anchors.dtype != torch.long
        or keep.dtype != torch.bool
        or anchors.shape != keep.shape
        or anchors.size(0) != attention_mask.size(0)
        or not 1 <= anchors.size(1) <= budget
        or ((anchors < 0) | (anchors >= attention_mask.size(1))).any()
        or (keep & ~attention_mask.gather(1, anchors).bool()).any()
    ):
        raise ValueError("invalid selected proposal anchors")


class RolloutTraceDataset(Dataset):
    """Read a step's saved JSONL traces for inspection or a single replay."""

    def __init__(
        self,
        path,
        *,
        draft_version,
        target_version,
        prefix_mode="teacher-forced",
        block_size=None,
    ):
        with Path(path).expanduser().resolve().open() as stream:
            self.records = [
                validate_trace(
                    json.loads(line),
                    draft_version=draft_version,
                    target_version=target_version,
                    prefix_mode=prefix_mode,
                    block_size=block_size,
                )
                for line in stream
                if line.strip()
            ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class TokenPromptDataset(Dataset):
    def __init__(self, path):
        with Path(path).expanduser().resolve().open() as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if not self.records:
            raise ValueError("empty rollout prompt dataset")
        for index, row in enumerate(self.records):
            ids = row.get("input_ids")
            prompt = row.get("prompt_length", len(ids) if isinstance(ids, list) else 0)
            if (
                not isinstance(ids, list)
                or not ids
                or type(prompt) is not int
                or not 0 < prompt <= len(ids)
                or any(type(t) is not int or t < 0 for t in ids)
            ):
                raise ValueError(f"invalid token prompt at row {index}")
            self.records[index] = dict(
                sample_id=str(row.get("sample_id", index)), input_ids=ids[:prompt]
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_token_prompts(records):
    return records


class ReplayCollator:
    def __init__(
        self,
        *,
        draft_version,
        target_version,
        num_anchors,
        eos_token_ids=(),
        pad_token_id=0,
        prefix_mode="teacher-forced",
        block_size=None,
    ):
        self.draft_version = draft_version
        self.target_version = target_version
        self.num_anchors = num_anchors
        self.eos_token_ids = tuple(eos_token_ids)
        self.pad_token_id = pad_token_id
        self.prefix_mode = prefix_mode
        self.block_size = block_size

    def __call__(self, records):
        if not records:
            raise ValueError("cannot collate an empty rollout batch")
        for record in records:
            validate_trace(
                record,
                draft_version=self.draft_version,
                target_version=self.target_version,
                prefix_mode=self.prefix_mode,
                block_size=self.block_size,
            )
        batch = len(records)
        length = max(len(r["input_ids"]) for r in records)
        width = max(1, max(len(r["anchor_positions"]) for r in records))
        ids = torch.full((batch, length), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        loss = torch.zeros_like(ids, dtype=torch.float32)
        anchors = torch.zeros((batch, width), dtype=torch.long)
        keep = torch.zeros_like(anchors, dtype=torch.bool)
        for row, record in enumerate(records):
            tokens, prompt = record["input_ids"], record["prompt_length"]
            ids[row, : len(tokens)] = torch.tensor(tokens)
            attention[row, : len(tokens)] = 1
            supplied_attention = record.get("attention_mask")
            if supplied_attention is not None:
                if (
                    len(supplied_attention) != len(tokens)
                    or any(v not in (0, 1) for v in supplied_attention)
                    or supplied_attention != sorted(supplied_attention, reverse=True)
                    or not all(supplied_attention[:prompt])
                ):
                    raise ValueError(
                        "trace attention_mask must be binary right padding with an unpadded prompt"
                    )
                attention[row, : len(tokens)] = torch.tensor(supplied_attention)
            loss[row, prompt : len(tokens)] = 1
            loss[row] *= attention[row]
            supplied = record.get("loss_mask")
            if supplied is not None:
                if len(supplied) != len(tokens) or any(
                    v not in (0, 1) for v in supplied
                ):
                    raise ValueError(
                        "trace loss_mask must be binary and aligned with tokens"
                    )
                loss[row, : len(tokens)] *= torch.tensor(supplied)
            for pos in range(prompt, len(tokens)):
                if tokens[pos] in self.eos_token_ids:
                    loss[row, pos + 1 :] = 0
                    break
            count = len(record["anchor_positions"])
            anchors[row, :count] = torch.tensor(
                record["anchor_positions"], dtype=torch.long
            )
            keep[row, :count] = True
        proposal_data = {}
        if self.prefix_mode == "proposal":
            proposal_data = self._collate_proposals(records, attention)
            anchors = proposal_data.pop("anchor_positions")
            keep = proposal_data.pop("block_keep_mask")
        else:
            anchors, keep = select_replay_anchors(
                anchors, keep, loss, attention, self.num_anchors
            )
        return dict(
            input_ids=ids,
            attention_mask=attention,
            loss_mask=loss,
            anchor_positions=anchors,
            block_keep_mask=keep,
            sample_id=torch.arange(batch),
            source_id=torch.zeros(batch, dtype=torch.long),
            trace_sample_ids=[r["sample_id"] for r in records],
            draft_version=[r["draft_version"] for r in records],
            target_version=[r["target_version"] for r in records],
            round_metadata=[r["round_metadata"] for r in records],
            **proposal_data,
        )

    def _collate_proposals(self, records, attention):
        # Reservoir sampling already ran in verification. Every field follows
        # the same round IDs, including teacher rows fetched by the capture RPC.
        width = max(1, max(len(r["selected_round_ids"]) for r in records))
        if width > self.num_anchors or self.block_size is None or self.block_size < 1:
            raise ValueError(
                "proposal replay requires a fixed K and bounded selected rounds"
            )
        shape = (len(records), width)
        anchors = torch.zeros(shape, dtype=torch.long)
        round_ids = torch.full(shape, -1, dtype=torch.long)
        proposals = torch.zeros((*shape, self.block_size), dtype=torch.long)
        valid = torch.zeros_like(proposals, dtype=torch.bool)
        for row, record in enumerate(records):
            for col, index in enumerate(record["selected_round_ids"]):
                meta = record["round_metadata"][index]
                anchors[row, col] = record["anchor_positions"][index]
                round_ids[row, col] = index
                proposals[row, col] = torch.tensor(meta["proposal_ids"])
                valid[row, col] = torch.tensor(meta["proposal_valid_mask"])
                expected = min(meta["verified_draft_tokens"], meta["remaining_budget"])
                for pos, token in enumerate(meta["proposal_ids"][:expected]):
                    if token in self.eos_token_ids:
                        expected = pos + 1
                        break
                if meta["proposal_valid_mask"] != [
                    i < expected for i in range(self.block_size)
                ]:
                    raise ValueError(
                        "proposal mask differs from verification coverage, budget or EOS"
                    )
        keep = valid.any(-1)
        validate_proposal_anchors(anchors, keep, attention, self.num_anchors)
        return dict(
            anchor_positions=anchors,
            block_keep_mask=keep,
            proposal_ids=proposals,
            proposal_valid_mask=valid,
            proposal_round_ids=round_ids,
        )


def assert_replay_batch(data, draft_version, target_version):
    for key in (
        "anchor_positions",
        "block_keep_mask",
        "draft_version",
        "target_version",
        "round_metadata",
    ):
        if key not in data:
            raise ValueError(f"replay batch missing {key}")
    size = data["input_ids"].size(0)
    for key, expected in (
        ("draft_version", draft_version),
        ("target_version", target_version),
    ):
        if len(data[key]) != size or any(v != expected for v in data[key]):
            raise ValueError(f"replay {key} mismatch before target capture")
