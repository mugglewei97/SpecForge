"""``torchrun ... -m specforge.on_policy --config recipe.yaml``."""

import argparse
import json
import logging
import os

from .config import OnPolicyConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="DSpark on-policy TV post-training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    cfg = OnPolicyConfig.from_file(args.config)
    if args.plan:
        print(json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False))
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not visible or any(not item.isdigit() for item in visible):
        raise ValueError(
            "set CUDA_VISIBLE_DEVICES to explicit trainer GPU ordinals, separate from rollout.cuda_devices"
        )
    if set(map(int, visible)) & set(cfg.rollout.cuda_devices):
        raise ValueError("trainer and rollout GPUs must be disjoint")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (
        int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != world_size
        or len(visible) != world_size
    ):
        raise ValueError(
            "initial on-policy launcher supports one node, one process per visible trainer GPU"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "on-policy training requires CUDA; use --plan on a personal computer"
        )
    from specforge.cli import _bootstrap_single_process_env
    from specforge.distributed import destroy_distributed, init_distributed

    from .trainer import run

    logging.basicConfig(level=logging.INFO)
    os.environ["FSDP_SHARDING"] = cfg.training.fsdp_sharding
    _bootstrap_single_process_env()
    init_distributed(timeout=cfg.training.dist_timeout)
    failed = True
    try:
        run(cfg)
        failed = False
        return 0
    finally:
        destroy_distributed(abort=failed)


if __name__ == "__main__":
    raise SystemExit(main())
