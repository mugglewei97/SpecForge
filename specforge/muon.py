"""Muon + AdamW with FP32 masters and an explicit FSDP1 full-matrix path.

Parameter routing follows speculators/train/optimizers.py. Unlike its FSDP2
DTensor path, FSDP1 shards must be gathered before matrix orthogonalization.
Optimizer/master states here are replicated, not sharded.
"""
from contextlib import nullcontext
import logging
import math

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

from specforge.lr_scheduler import CosineAnnealingWarmupLR

from specforge.legacy.dspark_training.options import add_muon_optimizer_args as add_muon_optimizer_args

logger = logging.getLogger(__name__)


def configure_fsdp_optimizer_precision(model, optimizer_name):
    """Configure BEFORE FSDP wrapping; never change FSDP parameter storage later.

    summon_full_params(with_grads=True) attaches gathered gradients to full
    parameters. Muon needs matching FP32 storage/gradients there, even though
    forward/backward runs with BF16 parameter views. Convert the entire wrapper
    (including frozen target embedding/head) to avoid mixed storage dtypes in
    one FSDP flat parameter. This promotes existing values, not initialization.
    """
    if optimizer_name == "adamw":
        # Exactly the historical DSpark mixed-precision configuration.
        return MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    if optimizer_name != "muon":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    model.float()
    logger.info("Muon FSDP precision: FP32 parameter storage and gradient reduction; "
                "BF16 forward/backward parameter views and buffers.")
    return MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,
                          reduce_dtype=torch.float32, keep_low_precision_grads=False)




def validate_muon_options(*, lr, muon_lr, momentum, weight_decay,
                          muon_weight_decay, ns_steps, adjust_lr_fn):
    for name, value in (("lr", lr), ("muon_lr", muon_lr),
                        ("weight_decay", weight_decay), ("muon_weight_decay", muon_weight_decay)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("muon momentum must be in [0, 1)")
    if not isinstance(ns_steps, int) or ns_steps < 1:
        raise ValueError("muon ns_steps must be a positive integer")
    if adjust_lr_fn not in ("original", "match_rms_adamw"):
        raise ValueError("unknown muon adjust_lr_fn")


def canonical_name(name):
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


def split_named_params_for_muon(model):
    """Route genuine nondegenerate hidden matrices to Muon; all else to AdamW.

Call only while original/full shapes are visible. Detect Embedding modules as
well as name hints: CARH depth/iteration embeddings have nonstandard names.
CARH markov_w2 is a vocabulary output projection, not a hidden matrix.
"""
    embedding_ids = {
        id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding)
        for p in module.parameters(recurse=False)
    }
    muon, adamw = [], []
    hints = ("embed_tokens", "lm_head", "codebook", "markov_w2")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name = canonical_name(name)
        destination = muon if (
            param.ndim == 2 and min(param.shape) > 1
            and id(param) not in embedding_ids
            and not any(hint in name for hint in hints)
        ) else adamw
        destination.append((name, param))
    return muon, adamw


class BF16MuonOptimizer:
    """Drive native torch Muon and AdamW, each with its own LR schedule.

    For FSDP1, pass the root wrapper as fsdp_model and its draft submodule as
    model. All ranks must call construction/step/load, including empty local
    shards. Each step gathers full synchronized gradients, updates full FP32
    masters on every rank, and writes model shards back before clearing grads.
    """
    def __init__(self, model, lr, *, fsdp_model=None, weight_decay=0.0,
                 max_grad_norm=0.5, total_steps=800_000, warmup_ratio=0.015,
                 muon_lr=None, muon_momentum=0.95, muon_weight_decay=0.1,
                 muon_ns_steps=5, muon_adjust_lr_fn="match_rms_adamw"):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("--optimizer muon requires native torch.optim.Muon (use repository PyTorch 2.11).")
        muon_lr = 10 * lr if muon_lr is None else muon_lr
        validate_muon_options(lr=lr, muon_lr=muon_lr, momentum=muon_momentum,
            weight_decay=weight_decay, muon_weight_decay=muon_weight_decay,
            ns_steps=muon_ns_steps, adjust_lr_fn=muon_adjust_lr_fn)
        if total_steps <= 0 or not 0 <= warmup_ratio < 1:
            raise ValueError("total_steps must be positive and warmup_ratio in [0, 1)")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        self.model, self.fsdp_model = model, fsdp_model
        self.max_grad_norm, self.last_grad_norm = max_grad_norm, None
        if fsdp_model is not None:
            units = FSDP.fsdp_modules(fsdp_model)
            if not units or any(not unit._use_orig_params for unit in units):
                raise ValueError("Muon FSDP1 requires the root FSDP wrapper and use_orig_params=True")
            if any(unit.cpu_offload.offload_params for unit in units):
                raise ValueError("Muon FSDP1 full-gradient path does not support CPU parameter offload")
            logger.warning("Muon FSDP1: full parameters/gradients are gathered at each optimizer step; "
                           "FP32 masters and both optimizer states are replicated on every rank.")
        elif any(getattr(p, "_fsdp_flattened", False) for p in model.parameters()):
            raise ValueError("Pass fsdp_model=root for FSDP-managed Muon parameters")

        with self._full_params(writeback=False):
            if fsdp_model is not None and any(
                p.is_floating_point() and p.dtype != torch.float32
                for p in fsdp_model.parameters()
            ):
                raise ValueError(
                    "Muon FSDP1 requires FP32 parameter storage for full-gradient gathering. "
                    "Call configure_fsdp_optimizer_precision(model, 'muon') BEFORE FSDP wrapping; "
                    "BF16 forward/backward is retained via MixedPrecision."
                )
            if fsdp_model is not None and any(
                unit.mixed_precision.keep_low_precision_grads for unit in units
            ):
                raise ValueError("Muon FSDP1 requires keep_low_precision_grads=False")
            muon, adamw = split_named_params_for_muon(model)
            routes = {id(p): "muon" for _, p in muon}
            named = [(canonical_name(n), p) for n, p in model.named_parameters() if p.requires_grad]
            if not named:
                raise ValueError("No trainable parameters found to optimize")
            if len({n for n, _ in named}) != len(named):
                raise ValueError("Duplicate canonical optimizer parameter names")
            if any(hasattr(p, "placements") or getattr(p, "_is_flat_param", False) for _, p in named):
                raise ValueError("This Muon adapter requires original full tensors, not DTensors/flat parameters")
            self.model_params = [p for _, p in named]
            self.fp32_params = [p.detach().clone().float().requires_grad_() for p in self.model_params]
            self._full_param_shapes = [p.shape for p in self.model_params]
            self.manifest = [dict(name=n, shape=list(p.shape), optimizer=routes.get(id(p), "adamw"))
                             for n, p in named]
            self.optimizers = {}
            for kind, selected in (("muon", muon), ("adamw", adamw)):
                ids = {id(p) for _, p in selected}
                masters = [mp for p, mp in zip(self.model_params, self.fp32_params) if id(p) in ids]
                if not masters:
                    continue
                if kind == "muon":
                    self.optimizers[kind] = torch.optim.Muon(masters, lr=muon_lr,
                        momentum=muon_momentum, weight_decay=muon_weight_decay,
                        ns_steps=muon_ns_steps, adjust_lr_fn=muon_adjust_lr_fn)
                else:
                    self.optimizers[kind] = torch.optim.AdamW(masters, lr=lr, weight_decay=weight_decay)
        self.schedulers = {
            name: CosineAnnealingWarmupLR(opt, total_steps=total_steps,
                                         warmup_steps=int(warmup_ratio * total_steps))
            for name, opt in self.optimizers.items()
        }
        # Legacy consumers use these aliases only for LR logging. Checkpoints
        # below explicitly serialize BOTH optimizers/schedulers.
        self.primary = "adamw" if "adamw" in self.optimizers else "muon"
        self.optimizer = self.optimizers[self.primary]
        self.scheduler = self.schedulers[self.primary]
        logger.info("Muon parameter routing: %d matrices via Muon, %d tensors via AdamW; "
                    "peak lr(muon)=%g lr(adamw)=%g", len(muon), len(adamw), muon_lr, lr)

    def _full_params(self, *, writeback=True, with_grads=False):
        if self.fsdp_model is None:
            return nullcontext()
        return FSDP.summon_full_params(self.fsdp_model, recurse=True,
            writeback=writeback, rank0_only=False, offload_to_cpu=False, with_grads=with_grads)

    def step(self):
        with self._full_params(with_grads=True), torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                if p.shape != mp.shape or (p.grad is not None and p.grad.shape != mp.shape):
                    raise RuntimeError("Muon requires full unsharded parameters and gradients; shape mismatch")
                if p.grad is not None and p.grad.is_sparse:
                    raise RuntimeError("Muon adapter does not support sparse gradients")
                mp.grad = p.grad.detach().float() if p.grad is not None else None
            # One norm over BOTH disjoint groups, with no world-size multiplier:
            # full FSDP gradients are already reduced and identical on all ranks.
            self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.fp32_params, self.max_grad_norm, error_if_nonfinite=True).detach()
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            for scheduler in self.schedulers.values():
                scheduler.step()
            for p, mp in zip(self.model_params, self.fp32_params):
                p.copy_(mp.to(p.dtype))
        # Do not clear grads while summon_full_params is writing gradients back.
        for p in self.model_params:
            p.grad = None
        return self.last_grad_norm

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_learning_rates(self):
        return {name: opt.param_groups[0]["lr"] for name, opt in self.optimizers.items()}

    def get_grad_norm(self):
        return self.last_grad_norm

    def set_full_param_shapes(self, shapes):
        if list(shapes) != self._full_param_shapes:
            raise ValueError("Captured pre-FSDP shapes do not match Muon full parameter shapes")

    @property
    def full_param_shapes(self):
        return self._full_param_shapes

    def state_dict(self):
        return dict(optimizer_type="muon", muon_state_version=1,
            parameter_manifest=self.manifest,
            fp32_master_params=[p.detach() for p in self.fp32_params],
            optimizer_state_dict={name: opt.state_dict() for name, opt in self.optimizers.items()},
            scheduler_state_dict={name: sch.state_dict() for name, sch in self.schedulers.items()})

    def load_state_dict(self, state):
        if state.get("optimizer_type", "adamw") != "muon" or state.get("muon_state_version") != 1:
            raise ValueError("Cannot resume Muon from AdamW/unknown optimizer state; use weight-only initialization")
        if state.get("parameter_manifest") != self.manifest:
            raise ValueError("Muon checkpoint parameter names/shapes/routing differ from the current model")
        masters = state.get("fp32_master_params", [])
        if len(masters) != len(self.fp32_params) or any(
            not isinstance(a, torch.Tensor) or a.shape != b.shape or a.dtype != torch.float32
            for a, b in zip(masters, self.fp32_params)
        ):
            raise ValueError("Muon checkpoint FP32 master parameters are missing or incompatible")
        for key in ("optimizer_state_dict", "scheduler_state_dict"):
            if set(state.get(key, {})) != set(self.optimizers):
                raise ValueError(f"Muon checkpoint {key} group mismatch")
        # Fail before mutation on mismatched per-parameter momentum tensors.
        for name, opt in self.optimizers.items():
            saved = state["optimizer_state_dict"][name]
            expected = [p for g in opt.param_groups for p in g["params"]]
            saved_ids = [i for g in saved["param_groups"] for i in g["params"]]
            if len(saved_ids) != len(expected):
                raise ValueError(f"Muon checkpoint {name} parameter count mismatch")
            for index, p in zip(saved_ids, expected):
                for key, value in saved["state"].get(index, {}).items():
                    if isinstance(value, torch.Tensor) and key != "step" and value.shape != p.shape:
                        raise ValueError(f"Muon checkpoint {name}/{key} shape mismatch")
        for name, opt in self.optimizers.items():
            opt.load_state_dict(state["optimizer_state_dict"][name])
            self.schedulers[name].load_state_dict(state["scheduler_state_dict"][name])
        with self._full_params(), torch.no_grad():
            for p, mp, saved in zip(self.model_params, self.fp32_params, masters):
                mp.copy_(saved)
                p.copy_(mp.to(p.dtype))
                mp.grad = None
        for p in self.model_params:
            p.grad = None
