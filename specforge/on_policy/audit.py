"""Audit a completed H200 smoke run without loading SGLang or any model."""

import argparse
import json
from pathlib import Path

from .data import load_trace


def audit(root, min_steps=2):
    root = Path(root)
    metrics_files = sorted((root / "metrics").glob("*.json"))
    if len(metrics_files) < min_steps:
        raise ValueError(f"need at least {min_steps} completed, synchronized steps")
    rejections, bonuses, blocks_count = 0, 0, 0
    for version, path in enumerate(metrics_files):
        metrics = json.loads(path.read_text())
        if metrics["step"] != version + 1 or metrics["weight_version"] != version + 1:
            raise ValueError("nonconsecutive completed weight versions")
        ids = json.loads((root / "batches" / f"{version:08d}.json").read_text())
        if len(ids) != metrics["samples"]:
            raise ValueError("sample count differs from the effective batch")
        for request_id in ids:
            trace, _ = load_trace(root, request_id, version)
            for block in trace["blocks"]:
                if not any(block["valid_mask"]):
                    continue
                blocks_count += 1
                if block["accepted_count"] < len(block["proposal"]):
                    rejections += 1
                else:
                    bonuses += 1
        if not (
            root / "weights" / f"{version + 1:08d}" / "model.safetensors"
        ).is_file():
            raise ValueError("missing full draft checkpoint")
    cfg = json.loads((root / "run.json").read_text())
    for worker in range(len(cfg["rollout"]["cuda_devices"])):
        ack = json.loads((root / "acks" / f"worker-{worker}.json").read_text())
        if ack["weight_version"] != len(metrics_files) or not ack["cache_invalidated"]:
            raise ValueError("not every rollout worker acknowledged the final version")
    if not rejections:
        raise ValueError(
            "smoke run did not exercise rejection/correction; use more prompts"
        )
    return {
        "steps": len(metrics_files),
        "valid_blocks": blocks_count,
        "rejection_blocks": rejections,
        "all_accepted_bonus_blocks": bonuses,
        "final_weight_version": len(metrics_files),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--min-steps", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.run_dir, args.min_steps), indent=2))


if __name__ == "__main__":
    main()
