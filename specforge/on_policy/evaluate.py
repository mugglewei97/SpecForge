"""Paired held-out acceptance measurement for two published draft versions."""

import argparse
import json
import shutil
import uuid
from pathlib import Path

from .config import OnPolicyConfig
from .data import atomic_json, load_prompts, load_trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Recipe with held-out data.train_data_path"
    )
    parser.add_argument(
        "--before", required=True, help="Published weights/00000000 directory"
    )
    parser.add_argument(
        "--after", required=True, help="Published trained weights/version directory"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args(argv)
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")
    from transformers import AutoConfig, AutoTokenizer

    from .rollout import RolloutPool

    cfg = OnPolicyConfig.from_file(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.target_model_path, trust_remote_code=cfg.model.trust_remote_code
    )
    target_config = AutoConfig.from_pretrained(
        cfg.model.target_model_path, trust_remote_code=cfg.model.trust_remote_code
    )
    if not cfg.sampling.ignore_eos:
        eos = getattr(target_config, "eos_token_id", None)
        eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
        cfg.sampling.stop_token_ids = sorted(
            set(
                cfg.sampling.stop_token_ids
                + eos
                + ([] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id])
            )
        )
    prompts = load_prompts(
        cfg.data.train_data_path,
        tokenizer,
        cfg.data.max_prompt_length,
        cfg.data.chat_template_kwargs,
    )[: args.max_samples]
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    configs = [
        json.loads((Path(source) / "config.json").read_text())
        for source in (args.before, args.after)
    ]
    if configs[0] != configs[1]:
        raise ValueError("paired evaluation requires identical draft configurations")
    results = {}
    for label, source in (("before", args.before), ("after", args.after)):
        root = destination / label
        weights = root / "weights" / "00000000"
        weights.mkdir(parents=True)
        for filename in ("config.json", "model.safetensors"):
            shutil.copyfile(Path(source) / filename, weights / filename)
        pool = RolloutPool(cfg, root, int(configs[0]["block_size"]))
        counts = []
        try:
            for offset in range(0, len(prompts), len(pool.workers)):
                requests = [
                    {"request_id": uuid.uuid4().hex, "input_ids": prompt["input_ids"]}
                    for prompt in prompts[offset : offset + len(pool.workers)]
                ]
                for result in pool.generate(requests):
                    trace, _ = load_trace(root, result["request_id"], 0)
                    blocks = [
                        block for block in trace["blocks"] if any(block["valid_mask"])
                    ]
                    counts.append(
                        {
                            "accepted": sum(
                                block["actual_accepted_count"] for block in blocks
                            ),
                            "blocks": len(blocks),
                            "output_tokens": len(trace["output_ids"]),
                        }
                    )
        finally:
            pool.close()
        total_blocks = sum(row["blocks"] for row in counts)
        if not total_blocks:
            raise RuntimeError(f"{label}: no speculative candidates to evaluate")
        results[label] = {
            "samples": counts,
            "mean_actual_accepted_length": sum(row["accepted"] for row in counts)
            / total_blocks,
        }
    results["delta_actual_accepted_length"] = (
        results["after"]["mean_actual_accepted_length"]
        - results["before"]["mean_actual_accepted_length"]
    )
    results["sampling"] = cfg.sampling.model_dump()
    results["note"] = (
        "Accepted draft tokens per valid block; excludes anchor/bonus. Capture overhead makes these runs unsuitable for throughput measurement."
    )
    atomic_json(destination / "comparison.json", results)
    print(
        json.dumps(
            {
                name: value
                for name, value in results.items()
                if name not in {"before", "after"}
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
