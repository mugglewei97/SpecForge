"""Experiment E: fixed pre-repair greedy reference, without update selection.

Replay only the one-sided reference loss, never another ordinary training loss.
Reference rows are training resources, not held-out evaluation data.
"""
import math
from pathlib import Path

import torch
import torch.distributed as dist

from specforge.netprefix import GreedyPrefixHook, cpu_copy, file_digest, rng_state, restore_rng
from specforge.netprefix_stream import TrainingStreamSource

from specforge.legacy.dspark_training.options import add_reference_args as add_reference_args




def validate_reference_args(args):
    if not math.isfinite(args.fixed_prefix_reference_alpha) or args.fixed_prefix_reference_alpha < 0:
        raise ValueError("Fixed reference alpha must be finite and non-negative")
    if min(args.on_policy_repair_value_horizon, args.fixed_prefix_reference_batches,
           args.fixed_prefix_reference_interval) < 1:
        raise ValueError("Repair horizon and reference batch count/interval must be positive")
    if not (args.on_policy_repair_value or args.fixed_prefix_reference_alpha):
        return
    if args.netprefix_mode != "off" or args.accumulation_steps != 1 or args.micro_batch_size or args.tp_size != 1:
        raise ValueError("D/E v1 requires NetPrefix off, TP=1, accumulation=1, no microbatch splitting")
    if args.init_draft_model_path or args.additional_train_data_path:
        raise ValueError("D/E requires scratch or same-arm resume and one fixed training source")
    if args.on_policy_start_step is None or not args.step_seeded_rollouts:
        raise ValueError("D/E requires absolute on-policy start and step-seeded rollouts")
    if args.fixed_prefix_reference_alpha and not args.on_policy_repair_value:
        raise ValueError("E requires D repair-value weighting")
    if args.multi_teacher_oracle_export_dir or args.multi_teacher_oracle_cache:
        raise ValueError("D/E does not support oracle export/cache")


class FixedPrefixReference:
    def __init__(self, args, model, loader, forward):
        self.args, self.model, self.forward = args, model, forward
        # Attribute writes on FSDP do not forward to its wrapped module.
        self.core = getattr(model, "module", model)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.source = TrainingStreamSource(loader.dataset, loader.collate_fn, args.batch_size,
                                          self.rank, self.world,
                                          args.fixed_prefix_reference_batches, args.seed + 2700000)
        self.batches = self.source.control_batches()
        self.references = []
        self.cycles = self.conflicts = 0
        self.hook = GreedyPrefixHook(self.core)
        self.core.netprefix_hook = self.hook
        keys = ("seed", "batch_size", "max_steps", "num_epochs", "on_policy_start_step",
                "on_policy_ramp_steps", "on_policy_interval", "fixed_prefix_reference_alpha",
                "fixed_prefix_reference_batches", "fixed_prefix_reference_interval")
        self.signature = {key: getattr(args, key) for key in keys}
        paths = args.train_data_path
        paths = paths if isinstance(paths, (list, tuple)) else [paths]
        hashes = [file_digest(path) for path in paths] if self.rank == 0 else None
        payload = [hashes]
        dist.broadcast_object_list(payload, src=0)
        self.signature.update(hashes=payload[0], rows=len(loader.dataset), world=self.world)

    def replay(self, index, *, capture=False):
        state = rng_state()
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        self.core._reference_replay_active = True
        self.hook.mode = "measure" if capture else "protect"
        self.hook.reference = None if capture else self.references[index]
        try:
            # The ordinary forward helper seeds block selection identically on replay.
            result = self.forward(self.batches[index], self.args.seed + 870000 + index)
            if capture:
                self.references.append(cpu_copy(self.hook.record))
            else:
                result[0].backward()
            return result[0].detach()
        finally:
            self.hook.mode = "off"
            self.hook.reference = None
            self.core._reference_replay_active = False
            for module, training in modes:
                module.training = training
            restore_rng(state)

    @torch.no_grad()
    def before_step(self, step):
        if step < self.args.on_policy_start_step or self.references:
            return
        if step != self.args.on_policy_start_step:
            raise RuntimeError("Missing pre-repair references: cannot silently recapture after training")
        for index in range(len(self.batches)):
            self.replay(index, capture=True)

    def backward(self, data, step):
        if step < self.args.on_policy_start_step or step % self.args.fixed_prefix_reference_interval:
            return {}
        ids = data["sample_id"].detach().cpu().reshape(-1).tolist()
        all_ids = [None] * self.world
        dist.all_gather_object(all_ids, ids)
        if self.source.plan.control_conflict([i for part in all_ids for i in part]):
            self.conflicts += 1
            return {"fixed_reference_conflicts": torch.tensor(float(self.conflicts), device=next(self.model.parameters()).device)}
        index = self.cycles % len(self.batches)
        ramp = min(1.0, (step - self.args.on_policy_start_step + 1)
                   / max(1, self.args.on_policy_ramp_steps))
        self.hook.weight = self.args.fixed_prefix_reference_alpha * ramp
        loss = self.replay(index)
        self.cycles += 1
        return {"fixed_reference_loss": loss, "fixed_reference_cycles": loss.new_tensor(self.cycles),
                **{"fixed_" + key: value for key, value in self.hook.protection_diagnostics.items()}}

    def save(self, directory, step):
        torch.save(dict(version=1, step=step, signature=self.signature,
                        references=self.references, batches=cpu_copy(self.batches),
                        cycles=self.cycles, conflicts=self.conflicts),
                   Path(directory) / f"fixed_prefix_reference_rank_{self.rank}.pt")

    def load(self, directory, step):
        state = torch.load(Path(directory) / f"fixed_prefix_reference_rank_{self.rank}.pt",
                           map_location="cpu", weights_only=False)
        if state["version"] != 1 or state["step"] != step or state["signature"] != self.signature:
            raise ValueError("Fixed reference checkpoint/configuration mismatch")
        refs = state["references"]
        expected = len(self.batches) if step >= self.args.on_policy_start_step else 0
        if len(refs) != expected or len(state["batches"]) != len(self.batches):
            raise ValueError("Incomplete fixed reference bank")
        self.references, self.batches = refs, state["batches"]
        self.cycles, self.conflicts = state["cycles"], state["conflicts"]
