"""Read-only comparison of fixed and legacy RoPE against saved rollout q.

python -m specforge.on_policy.diagnose --run-dir /path/to/run --max-blocks 8
Only the draft and frozen target embedding/head are loaded, on one device.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from .config import OnPolicyConfig
from .data import load_trace, trace_stem
from .replay import DSparkReplayModel, create_replay_draft


def tv_summary(q, rollout_q, valid_mask):
    tv = (q.float() - rollout_q.to(q.device).float()).abs().sum(-1) * 0.5
    valid = torch.tensor(valid_mask, device=q.device, dtype=torch.bool)
    return {
        "max_tv": tv.max().item(),
        "max_valid_tv": tv[valid].max().item(),
        "worst_position": int(tv.argmax()),
        "tv_by_position": tv.tolist(),
    }


def _run_path(root, *parts):
    path = root.joinpath(*parts).resolve()
    # COSEC: generated artifacts must remain inside the selected run, including
    # symlink targets; request IDs themselves are hashed by trace_stem.
    if not path.is_relative_to(root):
        raise ValueError("diagnostic artifact escapes the run directory")
    return path


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention-backend", choices=("sdpa", "eager", "flex_attention")
    )
    args = parser.parse_args(argv)
    if args.version < 0 or args.max_blocks <= 0:
        parser.error("version must be nonnegative and max-blocks must be positive")
    root = args.run_dir.expanduser().resolve()
    cfg = OnPolicyConfig.model_validate(
        json.loads(_run_path(root, "run.json").read_text())
    )
    weights = _run_path(root, "weights", f"{args.version:08d}")
    for name in ("config.json", "model.safetensors"):
        _run_path(root, "weights", f"{args.version:08d}", name)
    request_ids = json.loads(
        _run_path(root, "batches", f"{args.version:08d}.json").read_text()
    )

    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    from specforge.training.model_loading import (
        load_draft_config_source,
        warm_start_draft_model,
    )

    config = load_draft_config_source(str(weights))
    config._attn_implementation = (
        args.attention_backend or cfg.training.attention_backend
    )
    draft = create_replay_draft(config)
    warm_start_draft_model(draft, str(weights), draft_config=config, strategy="dspark")
    parts = TargetEmbeddingsAndHead.from_pretrained(
        cfg.model.target_model_path,
        embed_key=cfg.model.embedding_key,
        lm_head_key=cfg.model.lm_head_key,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
        device=args.device,
        dtype=torch.bfloat16,
    )
    model = DSparkReplayModel(draft, parts).to(device=args.device).eval()
    fixed_rope = draft.rotary_emb
    # Reproduce the old BF16 model cast followed by FSDP's FP32 buffer cast.
    legacy_rope = copy.deepcopy(fixed_rope).to(dtype=torch.bfloat16).float()
    print(
        json.dumps(
            {
                "version": args.version,
                "attention_backend": config._attn_implementation,
                "max_inv_freq_error": (fixed_rope.inv_freq - legacy_rope.inv_freq)
                .abs()
                .max()
                .item(),
                "replay_max_tv": cfg.training.replay_max_tv,
                "note": "No FSDP, optimizer update, or new rollout; other GPU kernel differences may remain.",
            }
        ),
        flush=True,
    )

    # Round-robin across requests, matching the first replay slots across ranks.
    candidates = []
    for request_id in request_ids:
        stem = trace_stem(root, request_id)
        for suffix in (".json", ".safetensors"):
            _run_path(root, "trajectories", stem.with_suffix(suffix).name)
        trace, _ = load_trace(root, request_id, args.version, metadata_only=True)
        candidates.append(
            (
                trace,
                [
                    i
                    for i, block in enumerate(trace["blocks"])
                    if any(block["valid_mask"])
                ],
            )
        )
    measured, slot = 0, 0
    while measured < args.max_blocks:
        found = False
        for trace, indices in candidates:
            if slot >= len(indices):
                continue
            found = True
            index = indices[slot]
            _, tensors = load_trace(root, trace["request_id"], args.version)
            block = trace["blocks"][index]
            context = tensors["context_hidden"][: block["context_length"]]
            result = {
                "request_id": trace["request_id"],
                "block": index,
                "context_length": block["context_length"],
                "valid_mask": block["valid_mask"],
            }
            for name, rotary in (
                ("legacy_rope", legacy_rope),
                ("fp32_rope", fixed_rope),
            ):
                draft.rotary_emb = rotary
                q = model.draft_probabilities(block, context)
                result[name] = tv_summary(q, tensors[f"q_{index}"], block["valid_mask"])
                del q
            print(json.dumps(result), flush=True)
            del tensors, context
            measured += 1
            if measured >= args.max_blocks:
                break
        if not found:
            break
        slot += 1
    if not measured:
        raise ValueError("batch has no valid blocks to diagnose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
