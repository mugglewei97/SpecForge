"""``torchrun ... -m specforge.on_policy --config recipe.yaml``."""

import argparse
import json
import logging
import os

from .config import OnPolicyConfig


def validate_topology(cfg, visible, world_size, local_world_size):
    devices = visible.split(",")
    if not devices or any(not item.isdigit() for item in devices):
        raise ValueError("set CUDA_VISIBLE_DEVICES to explicit trainer GPU ordinals")
    devices = list(map(int, devices))
    if (
        local_world_size != world_size
        or len(devices) != world_size
        or len(set(devices)) != len(devices)
    ):
        raise ValueError(
            "on-policy supports one node, one process per distinct visible trainer GPU"
        )
    if cfg.rollout.placement == "colocated":
        if devices != cfg.rollout.cuda_devices:
            raise ValueError(
                "colocated rollout requires one worker per trainer GPU, in CUDA_VISIBLE_DEVICES order"
            )
    elif set(devices) & set(cfg.rollout.cuda_devices):
        raise ValueError("dedicated trainer and rollout GPUs must be disjoint")


def main(argv=None):
    parser = argparse.ArgumentParser(description="DSpark on-policy TV post-training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    cfg = OnPolicyConfig.from_file(args.config)
    if args.plan:
        print(json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False))
        return 0
    validate_topology(
        cfg,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
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
