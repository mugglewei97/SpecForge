"""Memory-safe offline trajectory exchange for Multi-Teacher Oracle training.

Teacher checkpoints are never resident together.  Each teacher exports compact
Top-k proposal distributions and verifier outcomes into sharded ``.pt`` files.
The merge utility selects the highest continuation-value teacher for every
``(sample, anchor, temperature)`` state, and student training performs a cheap
CPU lookup followed by one serial verifier pass for the selected trajectory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch


def temperature_key(temperature: float) -> int:
    """Stable cache key for serving temperatures (millidegrees)."""
    return int(round(float(temperature) * 1000.0))


class MultiTeacherTrajectoryWriter:
    """Buffer and shard compact teacher trajectories on CPU."""

    def __init__(
        self,
        output_dir: str,
        teacher_name: str,
        rank: int,
        top_k: int = 32,
        flush_records: int = 4096,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.teacher_name = str(teacher_name)
        self.rank = int(rank)
        self.top_k = int(top_k)
        self.flush_records = int(flush_records)
        self.part = 0
        self.count = 0
        self._buffer = []

    @torch.no_grad()
    def record(
        self,
        sample_ids: torch.Tensor,
        anchors: torch.Tensor,
        temperatures: torch.Tensor,
        proposals: torch.Tensor,
        logits: torch.Tensor,
        verifier_ids: torch.Tensor,
        valid: torch.Tensor,
        active_rows: torch.Tensor,
        acceptance_probability: Optional[torch.Tensor] = None,
        continuation_value: Optional[torch.Tensor] = None,
    ) -> None:
        if sample_ids is None:
            raise RuntimeError("multi-teacher export requires stable sample_ids")
        keep = active_rows.bool()
        if not keep.any():
            return
        logits = logits.float()
        k = min(self.top_k, logits.size(-1))
        top_values, top_ids = torch.topk(logits, k=k, dim=-1)
        top_probs = torch.exp(top_values - torch.logsumexp(logits, dim=-1, keepdim=True))
        accepted = proposals.eq(verifier_ids) & valid.bool()
        if acceptance_probability is None:
            acceptance_probability = accepted.float()
        acceptance_probability = torch.where(
            valid.bool(), acceptance_probability.float(), torch.ones_like(valid)
        )
        if continuation_value is None:
            continuation_value = (
                torch.cumprod(acceptance_probability, dim=-1) * valid
            ).sum(dim=-1)
        item = {
            "sample_id": sample_ids[keep].detach().cpu().to(torch.int64),
            "anchor": anchors[keep].detach().cpu().to(torch.int32),
            "temperature_key": torch.tensor(
                [temperature_key(v) for v in temperatures[keep].detach().cpu().tolist()],
                dtype=torch.int32,
            ),
            "proposals": proposals[keep].detach().cpu().to(torch.int32),
            "topk_ids": top_ids[keep].detach().cpu().to(torch.int32),
            "topk_probs": top_probs[keep].detach().cpu().to(torch.float16),
            "accepted": accepted[keep].detach().cpu(),
            "acceptance_probability": (
                acceptance_probability[keep].detach().cpu().to(torch.float16)
            ),
            "value": continuation_value[keep].detach().cpu().to(torch.float16),
        }
        self._buffer.append(item)
        self.count += int(keep.sum().item())
        if sum(int(x["sample_id"].numel()) for x in self._buffer) >= self.flush_records:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        payload = {
            key: torch.cat([item[key] for item in self._buffer], dim=0)
            for key in self._buffer[0]
        }
        payload.update(
            {
                "format_version": 1,
                "teacher_name": self.teacher_name,
                "top_k": self.top_k,
            }
        )
        path = self.output_dir / (
            f"{self.teacher_name}.rank{self.rank:05d}.part{self.part:05d}.pt"
        )
        torch.save(payload, path)
        self.part += 1
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        manifest = {
            "format_version": 1,
            "teacher_name": self.teacher_name,
            "rank": self.rank,
            "records": self.count,
            "parts": self.part,
            "top_k": self.top_k,
        }
        path = self.output_dir / f"{self.teacher_name}.rank{self.rank:05d}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")


class MultiTeacherOracleCache:
    """CPU-resident lookup table for the merged per-state oracle cache."""

    def __init__(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError(f"unsupported multi-teacher cache format: {path}")
        self.payload = payload
        self.teacher_names = tuple(payload["teacher_names"])
        self._index: Dict[Tuple[int, int, int], int] = {}
        for index, values in enumerate(
            zip(
                payload["sample_id"].tolist(),
                payload["anchor"].tolist(),
                payload["temperature_key"].tolist(),
            )
        ):
            self._index[(int(values[0]), int(values[1]), int(values[2]))] = index

    def lookup(
        self,
        sample_ids: torch.Tensor,
        anchors: torch.Tensor,
        temperatures: torch.Tensor,
        device: torch.device,
    ) -> dict:
        keys = zip(
            sample_ids.detach().cpu().tolist(),
            anchors.detach().cpu().tolist(),
            temperatures.detach().cpu().tolist(),
        )
        indices = [
            self._index.get((int(sample), int(anchor), temperature_key(temp)), -1)
            for sample, anchor, temp in keys
        ]
        hit = torch.tensor([index >= 0 for index in indices], device=device)
        safe = torch.tensor([max(index, 0) for index in indices], dtype=torch.long)

        def take(name: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
            value = self.payload[name].index_select(0, safe)
            if dtype is not None:
                value = value.to(dtype)
            return value.to(device, non_blocking=True)

        return {
            "hit": hit,
            "proposals": take("proposals", torch.long),
            "topk_ids": take("topk_ids", torch.long),
            "topk_probs": take("topk_probs", torch.float32),
            "cached_value": take("value", torch.float32),
            "baseline_value": take("baseline_value", torch.float32),
            "teacher_index": take("teacher_index", torch.long),
            "acceptance_probability": take(
                "acceptance_probability", torch.float32
            ),
        }
