import os

import torch
import torch.distributed as dist

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        markov_lr_scale=1.0,
        fsdp_model=None,
    ):
        # TODO: For now, we only support cosine annealing warmup lr scheduler and AdamW optimizer
        # TODO: We should make these parameters configurable
        #   These magic numbers: weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k are copied from
        #   https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json
        self.model = model
        self.fsdp_model = fsdp_model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True

        # Build parameter groups: Markov head params get a scaled LR to
        # compensate for the weaker gradient signal through L1-softmax.
        # When markov_lr_scale == 1.0 (default), all params share the same LR
        # (backward compatible).
        if markov_lr_scale != 1.0:
            markov_names = set()
            for name, param in model.named_parameters():
                if param.requires_grad and ("markov_head" in name or "markov_scaffold" in name):
                    markov_names.add(id(param))
            if markov_names:
                base_group = []
                markov_group = []
                for p, mp in zip(self.model_params, self.fp32_params):
                    if id(p) in markov_names:
                        markov_group.append(mp)
                    else:
                        base_group.append(mp)
                self.optimizer = torch.optim.AdamW(
                    [
                        {"params": base_group, "lr": lr, "weight_decay": weight_decay},
                        {
                            "params": markov_group,
                            "lr": lr * markov_lr_scale,
                            "weight_decay": weight_decay,
                        },
                    ],
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    self.fp32_params, lr=lr, weight_decay=weight_decay
                )
            self._has_markov_group = len(markov_names) > 0
        else:
            self.optimizer = torch.optim.AdamW(
                self.fp32_params, lr=lr, weight_decay=weight_decay
            )
            self._has_markov_group = False

        self.last_grad_norm = None
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def step(self):
        # FSDP must clip the original gradients before they are copied to the
        # FP32 masters: only the root wrapper knows which gradients are sharded
        # and which process group to reduce over. Every rank must participate,
        # including ranks with no local gradients.
        if self.fsdp_model is not None:
            grad_norm = self.fsdp_model.clip_grad_norm_(self.max_grad_norm)
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )
        if self.fsdp_model is None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.fp32_params, self.max_grad_norm
            )
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
        return self.last_grad_norm

    def load_state_dict(self, state_dict):
        validate_optimizer_resume(state_dict, "adamw")
        osd = state_dict["optimizer_state_dict"]
        # A checkpoint can carry Adam buffers whose sizes do not match the
        # current model (e.g. it was produced by a differently-shaped build,
        # an architecture change mid-training, or an FSDP-shard save).  Passing
        # such tensors straight through crashes Adam's first step in
        # torch._foreach_lerp_ ("The size of tensor a (...) must match ...").
        # Drop the incompatible buffers and let Adam re-initialize them on the
        # first step so the model weights can still resume.
        osd = _sanitize_optimizer_state(osd, self.optimizer)
        self.optimizer.load_state_dict(osd)
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def get_learning_rate(self):
        # Return base LR (group 0); Markov group may have a different LR.
        return self.optimizer.param_groups[0]["lr"]

    def get_grad_norm(self):
        """Return the most recent pre-clipping gradient norm, if available."""
        return self.last_grad_norm

    def set_full_param_shapes(self, shapes):
        """Store the full (unsharded) parameter shapes captured *before*
        FSDP wrapping.  This is needed by :func:`save_fsdp_optimizer_state`
        and :func:`load_fsdp_optimizer_state` to convert between the
        rank-agnostic (full-param) format and the per-rank FSDP-sharded
        format.

        ``shapes`` must be a list of ``torch.Size``, one per
        ``requires_grad`` parameter, in the same order as
        ``self.model_params`` / ``self.fp32_params``.
        """
        self._full_param_shapes = list(shapes)

    @property
    def full_param_shapes(self):
        return getattr(self, "_full_param_shapes", None)


def validate_optimizer_resume(state, optimizer_name):
    saved_name = state.get("optimizer_type", "adamw")
    if saved_name != optimizer_name:
        raise ValueError(
            f"Cannot resume {optimizer_name} from {saved_name} optimizer state. "
            "Use weight-only --init-draft-model-path to change optimizers."
        )
    if optimizer_name == "muon" and not state.get("_fsdp_full_param_format"):
        raise ValueError("Muon resume requires the full-parameter checkpoint format")


def build_bf16_optimizer(model, *, optimizer_name="adamw", fsdp_model=None,
                         muon_lr=None, muon_momentum=0.95, muon_weight_decay=0.1,
                         muon_ns_steps=5, muon_adjust_lr_fn="match_rms_adamw", **kwargs):
    """Build an optimizer with the root FSDP wrapper for distributed updates."""
    if optimizer_name == "adamw":
        return BF16Optimizer(model, fsdp_model=fsdp_model, **kwargs)
    if optimizer_name != "muon":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    from specforge.muon import BF16MuonOptimizer

    return BF16MuonOptimizer(model, fsdp_model=fsdp_model, muon_lr=muon_lr,
        muon_momentum=muon_momentum, muon_weight_decay=muon_weight_decay,
        muon_ns_steps=muon_ns_steps, muon_adjust_lr_fn=muon_adjust_lr_fn, **kwargs)


def _optimizer_param_numels(optimizer):
    """Global per-param element counts in the same order as the optimizer's
    state-dict integer keys (concatenation of all param_groups)."""
    numels = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            numels.append(p.numel())
    return numels


def _sanitize_optimizer_state(osd, optimizer):
    """Drop per-param Adam buffers whose size does not match the optimizer's
    current parameters.

    Returns a *copy* of ``osd`` (the input dict is left untouched).  Param
    state entries with mismatched tensor sizes are removed wholesale so the
    optimizer lazily re-initializes them on the first step (equivalent to
    starting Adam fresh for just those parameters).  Scheduler state and
    param-group metadata are never modified.  """
    if not isinstance(osd, dict):
        return osd

    expected = _optimizer_param_numels(optimizer)
    state = osd.get("state")
    if not isinstance(state, dict) or not expected:
        return osd

    # Only these buffers are matched elementwise against gradients during
    # Adam's foreach step; the scalar "step" state must not be size-checked.
    _MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

    new_state = {}
    dropped_idx = []
    for key, entry in state.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            new_state[key] = entry
            continue
        # Global param missing from the current model (e.g. a removed head).
        if idx < 0 or idx >= len(expected):
            dropped_idx.append(idx)
            continue
        target = expected[idx]
        if any(
            isinstance(entry.get(mk), torch.Tensor)
            and entry.get(mk).numel() != target
            for mk in _MOMENT_KEYS
        ):
            dropped_idx.append(idx)
            continue
        new_state[key] = entry

    if dropped_idx:
        print_on_rank0(
            "Optimizer state: dropped %d param-state entr%s whose tensor shape(s) "
            "did not match the current model (global param indices %s). These "
            "parameters will re-initialize Adam momentum from zero on the first "
            "step. If you were not expecting a checkpoint/architecture mismatch, "
            "verify the model config: the checkpoint state is out of sync with "
            "the model."
            % (
                len(dropped_idx),
                "y" if len(dropped_idx) == 1 else "ies",
                sorted(dropped_idx)[:50],
            )
        )
        # Keep bookkeeping keys, replace state.
        osd = dict(osd)
        osd["state"] = new_state

    return osd


# ---------------------------------------------------------------------------
# FSDP-aware optimizer state save / load
# ---------------------------------------------------------------------------
#
# BF16Optimizer stores Adam state on fp32 *clones* of the model parameters.
# Because these clones are taken *before* FSDP wrapping, they always have
# the full (unsharded) shape — even after FSDP shards the original model
# parameters.  This means the optimizer state dict is naturally
# rank-agnostic: every rank sees the same parameter shapes and therefore
# produces identical Adam state tensors.
#
# However, the *legacy* checkpoint format (used before these helpers were
# introduced) was saved inside ``with FSDP.state_dict_type(FULL_STATE_DICT)``
# which temporarily all-gathers model params.  If the BF16Optimizer was
# constructed *after* FSDP wrapping (which the old codePath did at some
# point), the fp32 clones would have FSDP-sharded shapes, making the
# saved state rank-dependent.
#
# The new helpers always save in rank-agnostic (full-param) format and
# mark the checkpoint with ``"_fsdp_full_param_format": True`` so that
# :func:`load_fsdp_optimizer_state` knows how to decode it.


def save_fsdp_optimizer_state(
    optimizer: BF16Optimizer,
    fsdp_model: torch.nn.Module,
    path: str,
):
    """Save optimizer state in rank-agnostic (full-param) format.

    Must be called **inside** ``with FSDP.state_dict_type(model,
    StateDictType.FULL_STATE_DICT):`` so that FSDP-managed parameters
    are temporarily unsharded and the optimizer's fp32 clones have
    consistent shapes across ranks.

    Only rank 0 writes the file to avoid race conditions and wasted I/O.
    """
    state = optimizer.state_dict()
    state["_fsdp_full_param_format"] = True
    # Stash full param shapes so that a future load can verify / convert
    # even if the world size has changed.
    if optimizer.full_param_shapes is not None:
        state["_full_param_shapes"] = [
            list(s) for s in optimizer.full_param_shapes
        ]

    should_save = (not dist.is_initialized()) or dist.get_rank() == 0
    if should_save:
        # Ensure the directory exists (the caller may not have created it
        # yet when delegating checkpoint responsibility to us).
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(state, path)
        print_on_rank0(f"Saved rank-agnostic optimizer state to {path}")


def load_fsdp_optimizer_state(
    optimizer: BF16Optimizer,
    fsdp_model: torch.nn.Module,
    path: str,
    *,
    restore_scheduler: bool = True,
):
    """Load optimizer state from the rank-agnostic (full-param) format
    written by :func:`save_fsdp_optimizer_state`.

    Works regardless of the current world size or FSDP sharding, because
    the saved Adam state tensors are already in full-param shape — the
    same shape as the optimizer's fp32 clones.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not state.get("_fsdp_full_param_format"):
        raise RuntimeError(
            f"Checkpoint at {path} does not have the _fsdp_full_param_format "
            f"marker.  Use the legacy load path instead."
        )

    # Remove our bookkeeping keys before passing to optimizer.
    state.pop("_fsdp_full_param_format", None)
    state.pop("_full_param_shapes", None)

    if not restore_scheduler:
        state["scheduler_state_dict"] = optimizer.scheduler.state_dict()
    optimizer.load_state_dict(state)
    print_on_rank0(
        f"Loaded rank-agnostic optimizer state from {path}"
    )
