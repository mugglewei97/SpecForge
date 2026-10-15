"""Export DSpark or native DFlash2 model and optimizer checkpoints."""

import json
import os
import shutil

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from specforge.legacy.optimizer import save_fsdp_optimizer_state
from specforge.utils import print_on_rank0

from .config import is_dflash2_config


def save_checkpoint(args, epoch, step, dspark_model, draft_model, optimizer):
    """Save checkpoint."""
    is_dflash2 = is_dflash2_config(getattr(draft_model, "config", {}))
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    runtime = getattr(dspark_model, "netprefix_runtime", None)
    if runtime is not None:
        runtime.save(save_dir, step)
    reference_runtime = getattr(dspark_model, "fixed_prefix_reference_runtime", None)
    if reference_runtime is not None:
        reference_runtime.save(save_dir, step)

    with FSDP.state_dict_type(dspark_model, StateDictType.FULL_STATE_DICT):
        state_dict = dspark_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        training_state_path = os.path.join(save_dir, "training_state.pt")
        # Save optimizer state in rank-agnostic (full-param) format so that
        # all ranks can load the same file and re-shard to their local
        # FSDP param shapes.  Without this, each rank stores different
        # param shapes (FSDP-sharded), and loading rank 0's state on
        # other ranks causes size-mismatch crashes in Adam.
        save_fsdp_optimizer_state(
            optimizer,
            dspark_model,
            training_state_path,
        )
        if dist.get_rank() == 0:
            # Append non-optimizer fields (epoch, step, args) to the
            # file that save_fsdp_optimizer_state already created.
            saved = torch.load(
                training_state_path, map_location="cpu", weights_only=is_dflash2
            )
            saved["epoch"] = epoch
            saved["global_step"] = step
            saved["args"] = vars(args) if is_dflash2 else args
            if is_dflash2:
                from .dflash2 import dflash2_beta_resume_contract

                saved["dflash2_tv_beta"] = dflash2_beta_resume_contract(draft_model)
            torch.save(saved, training_state_path)

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            # Patch config.json: HuggingFace save_pretrained writes
            # architectures=["DSparkDraftModel"] (the SpecForge class name),
            # but sglang registers the draft model as "Qwen3DSparkModel".
            # Rewrite the field so sglang's ModelRegistry can find it.
            config_path = os.path.join(save_dir, "config.json")
            with open(config_path) as f:
                saved_cfg = json.load(f)
            if saved_cfg.get("architectures") == ["DSparkDraftModel"]:
                saved_cfg["architectures"] = ["Qwen3DSparkModel"]
                with open(config_path, "w") as f:
                    json.dump(saved_cfg, f, indent=2)
                    f.write("\n")

            # Copy the modeling files next to the checkpoint so auto_map can
            # resolve DSparkDraftModel (which subclasses DFlashDraftModel) on
            # reload with trust_remote_code.
            modeling_dir = os.path.join(os.path.dirname(__file__), "..")
            modeling_files = (
                ()
                if is_dflash2
                else (
                    "dspark.py",
                    "dflash.py",
                    "dflash2.py",
                    "neighbor_residual.py",
                    "light_conv_kernel.py",
                )
            )
            for fname in modeling_files:
                src = os.path.join(modeling_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(save_dir, fname))

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()
