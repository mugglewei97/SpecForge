"""Experimental H=1, greedy NetPrefix controller. No serving-path changes.

Control values are measured update effects, not oracle token replacements.
The first version enumerates a small fixed candidate set (no learned predictor).
"""

import contextlib
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from specforge.legacy.dspark_training.options import add_netprefix_args as add_netprefix_args




def validate_netprefix_args(a):
    if a.stop_after_steps is not None and (a.stop_after_steps < 1 or a.accumulation_steps != 1):
        raise ValueError("--stop-after-steps requires positive steps and accumulation_steps=1")
    if a.netprefix_mode == "off":
        return
    if a.accumulation_steps != 1 or a.micro_batch_size or a.optimizer != "adamw" or a.tp_size != 1:
        raise ValueError("NetPrefix v1 requires AdamW, TP=1, accumulation=1, no microbatch splitting")
    if a.init_draft_model_path:
        raise ValueError("NetPrefix requires scratch or full --resume, not weight-only initialization")
    if getattr(a, "additional_train_data_path", []):
        raise ValueError("NetPrefix v1 supports one fixed training source, not a changing mixture")
    if a.netprefix_start_step < 1 or a.netprefix_interval < 1 or a.netprefix_control_batches < 1 or a.netprefix_ramp_steps < 0:
        raise ValueError("NetPrefix start/interval/control-batches must be positive")
    if (not 0 < a.netprefix_extra_time_ratio <= 1 or
            not 1 <= len(a.netprefix_repair_weights) <= 2 or
            any(not np.isfinite(w) or w <= 0 for w in a.netprefix_repair_weights) or
            not np.isfinite(a.netprefix_protect_weight) or a.netprefix_protect_weight < 0 or
            not np.isfinite(a.netprefix_min_gain) or a.netprefix_min_gain < 0 or
            not np.isfinite(a.netprefix_audit_tolerance) or a.netprefix_audit_tolerance < 0):
        raise ValueError("Invalid NetPrefix weights, gain, tolerance or time ratio")
    # Stateful/alternative objectives are not part of the first factorial pilot.
    # Oracle alpha defaults to 0.005 even when mode="none". Reject the
    # activation switches, not dormant tuning parameters (also at alpha=0).
    if getattr(a, "multi_teacher_oracle_mode", "none") != "none":
        raise ValueError("NetPrefix v1 cannot combine with multi_teacher_oracle_mode")
    for key in ("multi_teacher_oracle_cache", "multi_teacher_oracle_export_dir"):
        if getattr(a, key, None):
            raise ValueError(f"NetPrefix v1 cannot combine with {key}")
    for key, value in vars(a).items():
        if key.startswith("on_policy_") and key.endswith("_alpha") and value:
            raise ValueError(f"NetPrefix v1 cannot combine with {key}")
    # Prefix-Full/T2CM/DSG are stateless auxiliary losses. Keep them in EVERY
    # ordinary/candidate update; protection replay returns only its own loss.
    for key in ("shallow_frc_alpha",):
        if getattr(a, key, 0):
            raise ValueError(f"NetPrefix v1 cannot combine with {key}")
    if a.netprefix_mode in ("fixed-greedy", "h1-greedy"):
        paths = [a.netprefix_control_data_path, a.netprefix_audit_data_path]
        if a.netprefix_data_source == "train-stream":
            if any(paths):
                raise ValueError("train-stream mode does not use control/audit file paths; unset them")
            return
        if not all(paths) or Path(paths[0]).resolve() == Path(paths[1]).resolve():
            raise ValueError("Distinct held-out training control/audit JSON files are required")
        train_paths = a.train_data_path if isinstance(a.train_data_path, list) else [a.train_data_path]
        if any(Path(p).resolve() == Path(t).resolve() for p in paths for t in train_paths):
            raise ValueError("Control/audit files must be separate from training files")


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["cpu"])
    if s["cuda"]:
        torch.cuda.set_rng_state_all(s["cuda"])


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def full_parameters(model, writeback=False):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model, recurse=True, writeback=writeback,
                                    offload_to_cpu=True, rank0_only=False):
            yield
    else:
        yield


class UpdateSnapshot:
    """CPU transaction; FSDP full trainable weights + rank-local Adam masters.

    Must be captured at a clean optimizer boundary. All ranks participate.
    Frozen parameters are never modified by this controller.
    """

    def __init__(self, model, optimizer):
        if any(p.grad is not None for p in optimizer.model_params):
            raise RuntimeError("NetPrefix snapshot requires cleared gradients")
        with full_parameters(model):
            self.weights = {n: cpu_copy(p) for n, p in model.named_parameters() if p.requires_grad}
            self.buffers = {n: cpu_copy(b) for n, b in model.named_buffers()}
        self.masters = cpu_copy(optimizer.fp32_params)
        self.adam = cpu_copy(optimizer.optimizer.state_dict())
        self.scheduler = cpu_copy(optimizer.scheduler.state_dict())
        self.grad_norm = cpu_copy(optimizer.last_grad_norm)
        self.rng = rng_state()

    @torch.no_grad()
    def restore(self, model, optimizer):
        with full_parameters(model, writeback=True):
            for n, p in model.named_parameters():
                if n in self.weights:
                    p.copy_(self.weights[n].to(p.device))
            for n, b in model.named_buffers():
                if n in self.buffers:
                    b.copy_(self.buffers[n].to(b.device))
        for p, value in zip(optimizer.fp32_params, self.masters):
            if p.shape != value.shape:
                raise RuntimeError("FSDP master shard layout changed during transaction")
            p.copy_(value.to(p.device))
            p.grad = None
        for p in optimizer.model_params:
            p.grad = None
        optimizer.optimizer.load_state_dict(copy.deepcopy(self.adam))
        optimizer.scheduler.load_state_dict(copy.deepcopy(self.scheduler))
        optimizer.last_grad_norm = cpu_copy(self.grad_norm)
        restore_rng(self.rng)


def prefix_masks(proposals, labels, valid):
    survived = (proposals.eq(labels) & valid).long().cumprod(-1).bool()
    alive = torch.cat([torch.ones_like(survived[:, :1]), survived[:, :-1]], -1)
    return survived, alive & ~proposals.eq(labels) & valid


def one_sided_prefix_loss(log_probs, reference, mask):
    # Compare each cumulative prefix, retaining all lengths, not just full chains.
    regression = (reference.cumsum(-1) - log_probs.cumsum(-1)).clamp_min(0)
    return (regression * mask).sum() / mask.sum().clamp_min(1)


class GreedyPrefixHook:
    def __init__(self, model):
        self.model = model
        self.mode = "off"
        self.record = None
        self.reference = None
        self.weight = 0.0
        self.target_scored_tokens = 0

    def __call__(self, base, hidden, anchors, valid, input_ids, attention_mask):
        if self.mode == "off":
            return base.new_zeros(())
        # One uniformly selected valid block per sequence. Seeded by the controller.
        eligible = valid.bool().any(-1)
        scores = torch.rand_like(eligible.float()).masked_fill(~eligible, -1)
        block = scores.argmax(-1)
        row = torch.arange(base.size(0), device=base.device)
        anchor = anchors[row, block]
        mask = valid[row, block].bool()
        active = mask.any(-1)
        selected_base, selected_hidden = base[row, block], hidden[row, block]
        previous = input_ids[row, anchor]
        steps, proposals = [], []
        ref = self.reference
        if self.mode == "protect":
            if not torch.equal(anchor.cpu(), ref["anchors"]):
                raise RuntimeError("Reference anchor changed; refusing stale-prefix protection")
        for depth in range(base.size(2)):
            bias = self.model.draft_model.markov_head.compute_step_bias(
                previous, selected_hidden[:, depth], depth_idx=depth)
            logits = selected_base[:, depth] + bias
            token = (ref["proposals"][:, depth].to(base.device) if self.mode == "protect"
                     else logits.detach().argmax(-1))
            steps.append(logits)
            proposals.append(token)
            previous = token
        logits = torch.stack(steps, 1).float()
        proposals = torch.stack(proposals, 1)
        logp = logits.log_softmax(-1).gather(-1, proposals.unsqueeze(-1)).squeeze(-1)
        if self.mode == "protect":
            with torch.no_grad():
                old = ref["survived"].to(base.device).bool()
                retained = logits.argmax(-1).eq(proposals).int().cumprod(-1).bool() & old
                self.protection_diagnostics = {
                    "reference_accepted_tokens": old.sum().float(),
                    "reference_lost_tokens": (old & ~retained).sum().float(),
                }
            return self.weight * one_sided_prefix_loss(
                logp, ref["logp"].to(base.device), ref["survived"].to(base.device))
        # All ranks call the target, even when one rank has no valid row.
        labels = self.model.on_policy_scorer(input_ids, attention_mask, anchor, proposals, active)
        if isinstance(labels, tuple):
            # Distribution scorers also return exact target argmax IDs first.
            labels = labels[0]
        if not torch.is_tensor(labels):
            raise RuntimeError("NetPrefix greedy scorer must return exact target argmax IDs")
        self.target_scored_tokens += int(attention_mask.sum().item())
        survived, frontier = prefix_masks(proposals, labels, mask)
        self.record = cpu_copy(dict(anchors=anchor, proposals=proposals, logp=logp,
                                   survived=survived, lengths=survived.sum(-1), active=active))
        if self.mode == "repair":
            ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(mask)
            return self.weight * (ce * frontier).sum() / frontier.sum().clamp_min(1)
        return logits.new_zeros(())


def distributed_mean(values, mask):
    stat = torch.stack([(values * mask).sum(), mask.sum()]).double()
    if dist.is_initialized():
        # TP=1 is enforced, so world == data parallel group.
        dist.all_reduce(stat)
    return float(stat[0] / stat[1]) if stat[1] else None


def eligible_update(before, after, tolerance):
    """Audit pre-update strata. Empty groups do not certify protection."""
    lengths, active = before
    delta = after[0] - lengths
    masks = [active, active & (lengths > 0) & (lengths <= 2),
             active & (lengths > 2) & (lengths <= 4), active & (lengths > 4)]
    means = [distributed_mean(delta, m) for m in masks]
    return means[0] is not None and all(x is None or x >= -tolerance for x in means)


class NetPrefixRuntime:
    def __init__(self, args, model, optimizer, hook, forward, control_batches=(), audit_loader=None):
        self.args, self.model, self.optimizer = args, model, optimizer
        self.hook, self.forward = hook, forward
        self.control = list(control_batches)
        self.audit_loader = audit_loader
        self.audit_iter = iter(audit_loader) if audit_loader is not None else None
        self.stream_source = audit_loader if getattr(args, "netprefix_data_source", "files") == "train-stream" else None
        self.stream_control_conflicts = 0
        self.references = []
        self.audit_used = 0
        self.base_seconds = 0.0
        self.extra_seconds = 0.0
        self.last_cycle_seconds = 0.0
        self.cycles = 0
        self.repairs_committed = 0
        self.updates_skipped = 0
        self.last_choice = 0
        self.last_gain = 0.0
        self.manifest = {}
        for key in ("train_data_path", "netprefix_control_data_path", "netprefix_audit_data_path"):
            paths = getattr(args, key, None)
            if paths:
                paths = paths if isinstance(paths, (list, tuple)) else [paths]
                hashes = [file_digest(p) for p in paths] if not dist.is_initialized() or dist.get_rank() == 0 else None
                if dist.is_initialized():
                    payload = [hashes]
                    dist.broadcast_object_list(payload, src=0)
                    hashes = payload[0]
                self.manifest[key] = hashes

    def clock(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def elapsed(self, start):
        elapsed = self.clock() - start
        value = torch.tensor(elapsed, device=next(self.model.parameters()).device)
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return float(value)

    def auxiliary_forward(self, batch, mode, seed, reference=None):
        state = rng_state()
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()  # Consistent reference/control scoring, no dropout.
        self.hook.mode, self.hook.reference = mode, reference
        torch.manual_seed(seed)
        try:
            return self.forward(batch, seed)
        finally:
            for m, training in modes:
                m.training = training
            self.hook.mode = "off"
            restore_rng(state)

    @torch.no_grad()
    def measure(self, batches, seed, capture=False):
        lengths, active = [], []
        for i, batch in enumerate(batches):
            self.auxiliary_forward(batch, "measure", seed + i)
            r = self.hook.record
            if capture:
                self.references.append(cpu_copy(r))
            lengths.append(r["lengths"])
            active.append(r["active"])
        device = next(self.model.parameters()).device
        return torch.cat(lengths).to(device), torch.cat(active).to(device)

    def update(self, batch, step, weight=0.0):
        self.hook.mode, self.hook.weight = ("repair" if weight else "off"), weight
        try:
            result = self.forward(batch, step)
            result[0].backward()
            if weight and self.args.netprefix_protect_weight:
                i = self.cycles % len(self.control)
                ramp = min(1.0, (step - self.args.netprefix_start_step + 1) / max(1, self.args.netprefix_ramp_steps))
                self.hook.weight = self.args.netprefix_protect_weight * ramp
                # Hook-only scalar avoids adding the base loss on reference data.
                self.auxiliary_forward(self.control[i], "protect", self.args.seed + 700000 + i,
                                       self.references[i])[0].backward()
            self.optimizer.step()
            return (result[0].detach(), result[1].detach(), *result[2:])
        finally:
            self.hook.mode = "off"

    def step(self, batch, step):
        a = self.args
        start = self.clock()
        potential_cycle = (a.netprefix_mode in ("fixed-greedy", "h1-greedy") and
                           step >= a.netprefix_start_step and
                           (step - a.netprefix_start_step) % a.netprefix_interval == 0)
        if potential_cycle and self.stream_source is not None:
            ids = batch["sample_id"].detach().cpu().tolist()
            if dist.is_initialized():
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, ids)
                ids = [i for group in gathered for i in group]
            # Same decision on all ranks. A control/reference row may appear
            # in ordinary training, but never in this cycle's repair batch.
            if self.stream_source.set_repair_ids(ids):
                self.stream_control_conflicts += 1
                result = self.update(batch, step)
                self.base_seconds += self.elapsed(start)
                return result
        ramp = min(1.0, max(0.0, (step - a.netprefix_start_step + 1) / max(1, a.netprefix_ramp_steps))) if a.netprefix_mode != "baseline" else 1.0
        if a.netprefix_mode == "fixed-greedy":
            due = step >= a.netprefix_start_step and (step - a.netprefix_start_step) % a.netprefix_interval == 0
            if due and not self.references:
                self.measure(self.control, a.seed + 700000, capture=True)
                count = torch.tensor(sum(int(r["survived"].sum()) for r in self.references), device=next(self.model.parameters()).device)
                if dist.is_initialized():
                    dist.all_reduce(count)
                if count == 0:
                    self.references = []
                    due = False
            result = self.update(batch, step, a.netprefix_repair_weights[0] * ramp if due else 0.0)
            # Fixed mode does not run a plain branch, so its full update cost
            # cannot be decomposed into a measured ordinary/candidate delta.
            self.base_seconds += self.elapsed(start)
            self.cycles += int(due)
            self.repairs_committed += int(due)
            self.last_choice = int(due)
            return result
        credit = a.netprefix_extra_time_ratio * self.base_seconds - self.extra_seconds
        due = (a.netprefix_mode == "h1-greedy" and step >= a.netprefix_start_step and
               (step - a.netprefix_start_step) % a.netprefix_interval == 0 and
               credit > max(self.last_cycle_seconds * 1.25, 0))
        if not due:
            result = self.update(batch, step)
            self.base_seconds += self.elapsed(start)
            return result
        # Audit is a finite, never-recycled stream. Stop probing on exhaustion.
        audit = next(self.audit_iter, None)
        available = torch.tensor(int(audit is not None), device=next(self.model.parameters()).device)
        if dist.is_initialized():
            dist.all_reduce(available, op=dist.ReduceOp.MIN)
        if not available:
            result = self.update(batch, step)
            self.base_seconds += self.elapsed(start)
            return result
        self.audit_used += 1
        initial = UpdateSnapshot(self.model, self.optimizer)
        try:
            control_seed = a.seed + 700000
            if not self.references:
                self.measure(self.control, control_seed, capture=True)
                device = next(self.model.parameters()).device
                reference_tokens = torch.tensor(sum(int(r["survived"].sum()) for r in self.references), device=device)
                if dist.is_initialized():
                    dist.all_reduce(reference_tokens)
                if reference_tokens == 0:
                    self.references = []
                    initial.restore(self.model, self.optimizer)
                    base_start = self.clock()
                    result = self.update(batch, step)
                    base_time = self.elapsed(base_start)
                    self.base_seconds += base_time
                    self.last_cycle_seconds = max(0, self.elapsed(start) - base_time)
                    self.extra_seconds += self.last_cycle_seconds
                    self.cycles += 1
                    result[-1]["netprefix_empty_reference"] = result[0].new_tensor(1)
                    return result
            before = self.measure([audit], a.seed + 900000 + self.audit_used)
            initial.restore(self.model, self.optimizer)
            base_start = self.clock()
            result = self.update(batch, step)
            base_time = self.elapsed(base_start)
            base_state = UpdateSnapshot(self.model, self.optimizer)
            base_control = self.measure(self.control, control_seed)
            base_audit = self.measure([audit], a.seed + 900000 + self.audit_used)
            base_ok = eligible_update(before, base_audit, a.netprefix_audit_tolerance)
            best_state = base_state
            best_gain = a.netprefix_min_gain
            chosen = 0
            base_result = result
            candidate_gains = []
            for index, weight in enumerate(a.netprefix_repair_weights):
                # Non-preemptible operations may overshoot. Never start another
                # candidate once the observed cycle credit has been consumed.
                if self.elapsed(start) - base_time >= credit:
                    break
                initial.restore(self.model, self.optimizer)
                candidate_result = self.update(batch, step, weight * ramp)
                candidate_state = UpdateSnapshot(self.model, self.optimizer)
                control = self.measure(self.control, control_seed)
                gain = distributed_mean(control[0] - base_control[0], control[1] & base_control[1])
                candidate_gains.append({"weight": weight * ramp, "gain_vs_base": gain})
                if gain is not None and gain > best_gain:
                    best_state, best_gain, chosen = candidate_state, gain, index + 1
                    result = candidate_result
            # Audit exactly the control-selected winner, not each candidate.
            # A failed audit never triggers another candidate search on that batch.
            if chosen > 0:
                best_state.restore(self.model, self.optimizer)
                audit_result = self.measure([audit], a.seed + 900000 + self.audit_used)
                if not eligible_update(before, audit_result, a.netprefix_audit_tolerance):
                    chosen, result = 0, base_result
            if chosen == 0:
                best_state = base_state if base_ok else initial
                chosen = 0 if base_ok else -1
            best_state.restore(self.model, self.optimizer)
            # Preserve the ordinary branch's data/RNG stream, independent of extra forwards.
            restore_rng(base_state.rng)
            if chosen == -1:
                # No parameter update, but consume this scheduled training opportunity.
                self.optimizer.scheduler.step()
            result[-1].update({"netprefix_chosen": result[0].new_tensor(chosen),
                               "netprefix_control_gain": result[0].new_tensor(best_gain if chosen > 0 else 0)})
        except BaseException:
            initial.restore(self.model, self.optimizer)
            raise
        self.cycles += 1
        duration = self.elapsed(start)
        self.base_seconds += base_time
        self.last_cycle_seconds = max(0, duration - base_time)
        self.extra_seconds += self.last_cycle_seconds
        self.repairs_committed += int(chosen > 0)
        self.updates_skipped += int(chosen < 0)
        self.last_choice, self.last_gain = chosen, best_gain if chosen > 0 else 0.0
        if not dist.is_initialized() or dist.get_rank() == 0:
            Path(a.output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(a.output_dir) / "netprefix_cycles.jsonl", "a") as stream:
                stream.write(json.dumps(dict(step=step, chosen=chosen, candidates=candidate_gains,
                                             base_audit_passed=base_ok, base_seconds=self.base_seconds,
                                             extra_seconds=self.extra_seconds, credit_before=credit,
                                             target_scored_tokens_rank0=self.hook.target_scored_tokens)) + "\n")
        result[-1].update({"netprefix_extra_seconds": result[0].new_tensor(self.extra_seconds),
                           "netprefix_base_seconds": result[0].new_tensor(self.base_seconds)})
        return result

    def metrics(self, device):
        values = dict(cycles=self.cycles, repairs_committed=self.repairs_committed,
                      stream_control_conflicts=self.stream_control_conflicts,
                      updates_skipped=self.updates_skipped, last_choice=self.last_choice,
                      last_control_gain=self.last_gain, base_seconds=self.base_seconds,
                      extra_seconds=self.extra_seconds, target_scored_tokens=self.hook.target_scored_tokens)
        return {"netprefix_" + k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in values.items()}

    def save(self, directory, step):
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.save(dict(version=1, step=step, world=dist.get_world_size() if dist.is_initialized() else 1,
                        stream_state=self.stream_source.state_dict() if self.stream_source is not None else None,
                        stream_control_conflicts=self.stream_control_conflicts,
                        references=self.references, audit_used=self.audit_used, cycles=self.cycles,
                        repairs_committed=self.repairs_committed, updates_skipped=self.updates_skipped,
                        last_choice=self.last_choice, last_gain=self.last_gain,
                        base_seconds=self.base_seconds, extra_seconds=self.extra_seconds,
                        last_cycle_seconds=self.last_cycle_seconds, rng=rng_state(),
                        masters=cpu_copy(self.optimizer.fp32_params), args=vars(self.args), manifest=self.manifest,
                        target_scored_tokens=self.hook.target_scored_tokens),
                   Path(directory) / f"netprefix_rank_{rank}.pt")

    def load(self, directory, step):
        rank = dist.get_rank() if dist.is_initialized() else 0
        s = torch.load(Path(directory) / f"netprefix_rank_{rank}.pt", map_location="cpu", weights_only=False)
        if s["version"] != 1 or s["step"] != step or s["world"] != (dist.get_world_size() if dist.is_initialized() else 1):
            raise ValueError("NetPrefix resume requires matching checkpoint and world size")
        if s["args"].get("netprefix_data_source", "files") != getattr(self.args, "netprefix_data_source", "files"):
            raise ValueError("Cannot switch NetPrefix data source during resume")
        keys = ["batch_size", "seed", "max_steps", "num_epochs", "learning_rate", "warmup_ratio", "train_data_path",
                "step_seeded_rollouts", "max_length", "num_anchors", "block_size", "trainable_parameter_scope",
                "target_model_path", "target_model_backend", "weight_decay", "max_grad_norm"]
        keys += [k for k in vars(self.args) if k.startswith(("prefix_credit_", "transition2_margin_", "deep_survival_guard_", "pace_"))]
        if s["references"]:
            keys += [k for k in vars(self.args) if k.startswith("netprefix_")]
        for key in keys:
            if s["args"].get(key) != getattr(self.args, key):
                raise ValueError(f"NetPrefix resume setting changed: {key}")
        for key, digest in s["manifest"].items():
            if key == "train_data_path" or s["references"]:
                if self.manifest.get(key) != digest:
                    raise ValueError(f"NetPrefix data content changed: {key}")
        for key in ["references", "audit_used", "cycles", "base_seconds", "extra_seconds", "last_cycle_seconds",
                    "repairs_committed", "updates_skipped", "last_choice", "last_gain"]:
            setattr(self, key, s[key])
        self.stream_control_conflicts = s.get("stream_control_conflicts", 0)
        if self.stream_source is not None:
            self.stream_source.load_state_dict(s["stream_state"])
        else:
            for _ in range(self.audit_used):
                if next(self.audit_iter, None) is None:
                    raise ValueError("Audit stream changed or truncated on resume")
        if len(s["masters"]) != len(self.optimizer.fp32_params):
            raise ValueError("NetPrefix master parameter count changed")
        with torch.no_grad():
            for p, saved in zip(self.optimizer.fp32_params, s["masters"]):
                if p.shape != saved.shape:
                    raise ValueError("NetPrefix master parameter shard changed")
                p.copy_(saved.to(p.device))
        self.hook.target_scored_tokens = s["target_scored_tokens"]
        restore_rng(s["rng"])
