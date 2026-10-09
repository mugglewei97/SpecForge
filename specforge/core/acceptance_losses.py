# coding=utf-8
"""Greedy Survival Loss: a verification-native acceptance objective.

At serving time, the greedy verifier accepts draft token *y* at depth *d*
if and only if::

    argmax(z_d) == y*

where *z_d* are the draft logits and *y** is the target's greedy token.
A greedy speculative chain of length *L* survives up to depth *d* when
**every** earlier token at depths 0 … d-1 is accepted.

The Survival Loss optimises the expected number of accepted tokens in such
a chain.  For each depth *d*, the differentiable surrogate is:

-  Soft accept:  ``â_d = σ(m_d / τ)``  where
   ``m_d = z_d(y*_d) - max_{y ≠ y*_d} z_d(y)`` is the **top-2 margin** and
   ``τ > 0`` is a temperature.
-  Soft survival:  ``ŝ_d = Π_{k=0}^{d} â_k``
-  Hard alive mask (default):  after the first hard rejection (``argmax ≠ y*``),
   the chain is dead — eligible positions receive zero contribution.

The loss is::

    L_survival = 1 - (Σ_d contribution_d) / capacity

where ``capacity`` counts all valid positions (including rejected suffixes)
so that early rejection penalises the **entire lost suffix**, not just the
rejected token.

Two modes are supported via the ``hard_alive`` parameter:

-  **hard_alive=True** (default):  The hard alive mask is used.
   After the first hard rejection, ``eligible = alive & valid_mask`` becomes
   ``False`` and contribution is zero.  Gradient flows only through
   soft survival (**not** through discrete rollout decisions).

   When ``leaky_eta > 0`` is used with ``hard_alive=True``, rejected
   suffixes retain a small weight η instead of zero::

       H̃_{d-1} = H_{d-1} + η(1 - H_{d-1})

   Only **valid** positions that are dead due to hard rejection receive
   the leaky weight — padding / OOV positions always remain zero so
   numerator and denominator stay aligned.  The alive propagation
   remains hard — η only affects the contribution weight, not the
   alive mask update.

-  **hard_alive=False** (soft survival):  No alive mask is applied.
   Contribution at every depth uses all valid positions, so gradient
   flows through the **entire chain**, including rejected suffix positions.
   This allows the loss to back-propagate surviving-signal corrections
   to early-step logits even when a later step was hard-rejected.

Key implementation constraints
------------------------------
1. Survival Loss operates on **base_logits** only — at deployment the
   Markov head is removed, so improving corrected_logits would optimise
   something that doesn't exist at serving time.
2. The hard alive mask is **detached** (when used) — gradient flows only
   through the soft survival (logsigmoid), not through discrete rollout
   decisions.
3. Positions whose target top-1 is outside the draft vocabulary are masked
   out via ``valid_mask`` (provided by the caller from ``position_mask``).
4. After the first hard rejection (hard_alive mode), ``eligible = alive &
   valid_mask`` becomes ``False`` and contribution is zero — rejected
   suffix tokens do not enter the numerator.  With ``leaky_eta > 0``,
   rejected suffixes retain weight η (see above).
5. ``capacity`` does **not** use the alive mask — rejected suffix positions
   remain in the denominator, penalising early rejection.
6. ``leaky_eta`` only affects the **contribution weight** of valid
   rejected positions, not invalid/padding positions and not the alive
   propagation — subsequent steps still use the hard alive mask.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def sampled_path_acceptance_objective(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_samples: int = 2,
    topk: int = 32,
    sampling_temperature: float = 1.0,
    reward_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Optimize complete greedy-verification paths with sampled advantages.

    Samples are scored only until their first mismatch with the target
    continuation.  Before that mismatch the sampled prefix is identical to
    the target prefix, so the observed prefix length is an exact greedy
    verification reward and does not require another target-model rollout.
    A sample-set mean baseline makes the objective a low-variance path-ranking
    policy gradient.  The primary D-PACE loss remains responsible for
    token-level calibration.
    """
    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, N, K, V]")
    if target_ids.shape != logits.shape[:-1]:
        raise ValueError("target_ids must match logits without the vocab axis")
    if valid_mask.shape != target_ids.shape:
        raise ValueError("valid_mask must match target_ids")
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if sampling_temperature <= 0:
        raise ValueError("sampling_temperature must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")

    valid = valid_mask.bool()
    valid_input = valid.to(torch.int64).clone()
    valid_input[..., 0] = 1
    chain_valid = valid_input.cumprod(dim=-1).bool()
    chain_valid[..., 0] = False

    # Sampling from the full vocabulary would materialize another float32
    # tensor as large as the training logits.  Restricting the proposal to the
    # model's top-k keeps the path objective practical while preserving the
    # alternatives that carry nearly all probability mass.
    proposal_topk = min(int(topk), logits.size(-1))
    top_values, top_ids = torch.topk(logits, k=proposal_topk, dim=-1)
    target_values = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    target_present = top_ids.eq(target_ids.unsqueeze(-1)).any(dim=-1)
    top_values = torch.cat(
        [
            top_values[..., :-1],
            torch.where(
                target_present, top_values[..., -1], target_values
            ).unsqueeze(-1),
        ],
        dim=-1,
    )
    top_ids = torch.cat(
        [
            top_ids[..., :-1],
            torch.where(
                target_present, top_ids[..., -1], target_ids
            ).unsqueeze(-1),
        ],
        dim=-1,
    )
    log_probs = F.log_softmax(
        top_values.float() / float(sampling_temperature), dim=-1
    )
    flat_probs = log_probs.exp().reshape(-1, proposal_topk)
    sampled_local_ids = torch.multinomial(
        flat_probs,
        num_samples=num_samples,
        replacement=True,
    ).view(*target_ids.shape, num_samples)
    sampled_ids = top_ids.unsqueeze(-2).expand(
        *top_ids.shape[:-1], num_samples, proposal_topk
    ).gather(-1, sampled_local_ids.unsqueeze(-1)).squeeze(-1)
    sampled_log_probs = log_probs.unsqueeze(-2).expand(
        *log_probs.shape[:-1], num_samples, log_probs.shape[-1]
    ).gather(-1, sampled_local_ids.unsqueeze(-1)).squeeze(-1)

    matches = sampled_ids.eq(target_ids.unsqueeze(-1))
    accepted_or_invalid = matches | (~chain_valid.unsqueeze(-1))
    previous_accept = torch.cat(
        [
            torch.ones_like(accepted_or_invalid[..., :1, :]),
            accepted_or_invalid[..., :-1, :],
        ],
        dim=-2,
    )
    alive_before = previous_accept.to(torch.int64).cumprod(dim=-2).bool()
    decisions = alive_before & chain_valid.unsqueeze(-1)
    accepted = decisions & matches

    valid_depth = chain_valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    rewards = accepted.float().sum(dim=-2) / valid_depth
    active_blocks = chain_valid.any(dim=-1)
    baseline = rewards.mean(dim=-1, keepdim=True)
    advantages = (rewards - baseline) / float(reward_temperature)

    decision_count = decisions.float().sum(dim=-2).clamp_min(1.0)
    path_log_prob = (
        sampled_log_probs * decisions.to(sampled_log_probs.dtype)
    ).sum(dim=-2) / decision_count
    per_block_loss = -(advantages.detach() * path_log_prob).mean(dim=-1)
    active_count = active_blocks.float().sum().clamp_min(1.0)
    loss = (
        per_block_loss * active_blocks.to(per_block_loss.dtype)
    ).sum() / active_count

    active_f = active_blocks.unsqueeze(-1).to(rewards.dtype)
    sample_count = active_f.sum().clamp_min(1.0) * float(num_samples)
    diagnostics = {
        "path_acceptance_loss": loss.detach(),
        "path_acceptance_reward_mean": (rewards * active_f).sum()
        / sample_count,
        "path_acceptance_reward_best": (
            rewards.max(dim=-1).values * active_blocks.float()
        ).sum()
        / active_count,
        "path_acceptance_advantage_abs": (
            advantages.abs() * active_f
        ).sum()
        / sample_count,
        "path_acceptance_active_block_fraction": active_blocks.float().mean(),
        "path_acceptance_decision_depth": (
            decisions.float().sum(dim=-2) * active_f
        ).sum()
        / sample_count,
        "path_acceptance_topk": torch.tensor(
            proposal_topk, device=logits.device, dtype=torch.float32
        ),
    }
    return loss, {key: value.detach() for key, value in diagnostics.items()}


def reachability_gated_dpace_weights(
    dpace_weights: torch.Tensor,
    hard_accept: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    suffix_retention: float = 0.0,
    preserve_weight_mass: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Gate D-PACE credit by the prefix reachable before first failure.

    DFlash position zero is the context anchor and is excluded from
    ``valid_mask``.  For every block, the AUF support contains all contiguous
    valid draft positions through (and including) the first greedy failure.
    Positions after that failure retain ``suffix_retention`` of their original
    D-PACE weight.  The hard AUF special case is ``suffix_retention=0``.

    When ``preserve_weight_mass`` is enabled, weights are rescaled per block so
    the gated and original D-PACE masses agree.  This changes where the block's
    gradient is spent without changing its total credit or the batch loss
    scale.  All reachability decisions and rescaling factors are detached.
    """
    if dpace_weights.shape != hard_accept.shape:
        raise ValueError("dpace_weights and hard_accept must have the same shape")
    if dpace_weights.shape != valid_mask.shape:
        raise ValueError("dpace_weights and valid_mask must have the same shape")
    if dpace_weights.ndim < 1 or dpace_weights.shape[-1] < 2:
        raise ValueError("D-PACE AUF gating requires a draft-depth dimension")
    if not 0.0 <= suffix_retention <= 1.0:
        raise ValueError("suffix_retention must be in [0, 1]")

    with torch.no_grad():
        valid = valid_mask.bool()

        # Labels may terminate before the configured block size.  Make the
        # supervised draft region contiguous so an internal gap cannot create
        # a false, later AUF frontier.  Depth zero is the context anchor.
        chain_valid_input = valid.to(torch.int64).clone()
        chain_valid_input[..., 0] = 1
        chain_valid = chain_valid_input.cumprod(dim=-1).bool()
        chain_valid[..., 0] = False

        accepted_for_prefix = torch.where(
            chain_valid, hard_accept.bool(), torch.ones_like(chain_valid)
        )
        accepted_for_prefix[..., 0] = True
        previous_accept = torch.cat(
            [
                torch.ones_like(accepted_for_prefix[..., :1]),
                accepted_for_prefix[..., :-1],
            ],
            dim=-1,
        )
        alive_before = previous_accept.to(torch.int64).cumprod(dim=-1).bool()
        reachable = chain_valid & alive_before

        gate = torch.full_like(dpace_weights, float(suffix_retention))
        gate = torch.where(reachable, torch.ones_like(gate), gate)
        gate = gate * chain_valid.to(gate.dtype)
        gated_weights = dpace_weights.detach() * gate

        # ``dpace_weights`` is already validity-masked by the caller.  Count
        # its complete original mass so even a pathological internal label
        # gap cannot silently change the batch loss scale.
        base_mass = dpace_weights.detach().sum(dim=-1, keepdim=True)
        gated_mass = gated_weights.sum(dim=-1, keepdim=True)
        active_block = base_mass.gt(0)
        if preserve_weight_mass:
            mass_scale = torch.where(
                active_block,
                base_mass / gated_mass.clamp_min(1e-12),
                torch.ones_like(base_mass),
            )
            gated_weights = gated_weights * mass_scale
        else:
            mass_scale = torch.ones_like(base_mass)

        suffix = chain_valid & (~reachable)
        valid_count = chain_valid.float().sum().clamp_min(1.0)
        active_count = active_block.float().sum().clamp_min(1.0)
        total_base_mass = base_mass.sum().clamp_min(1e-12)
        diagnostics = {
            "auf_reachable_fraction": reachable.float().sum() / valid_count,
            "auf_suffix_fraction": suffix.float().sum() / valid_count,
            "auf_suffix_weight_mass_fraction": (
                dpace_weights.detach() * suffix.to(dpace_weights.dtype)
            ).sum()
            / total_base_mass,
            "auf_mass_scale_mean": (
                mass_scale * active_block.to(mass_scale.dtype)
            ).sum()
            / active_count,
            "auf_original_weight_mass": base_mass.sum(),
            "auf_gated_weight_mass": gated_weights.sum(),
            "auf_active_block_count": active_block.float().sum(),
        }

    return gated_weights.to(dtype=dpace_weights.dtype), {
        key: value.detach() for key, value in diagnostics.items()
    }


@dataclass
class GreedySurvivalState:
    """Per-position state for a greedy speculative chain.

    Attributes:
        alive:
            Hard mask [B, S].  True if every earlier draft token was
            hard-accepted (argmax == target).  Detached from the
            computation graph.  ``None`` in soft survival mode
            (``hard_alive=False``).
        log_survival:
            Accumulated soft log-survival [B, S].  Differentiable
            w.r.t. draft logits via the top-2 margin surrogate.
    """

    alive: Optional[torch.Tensor]
    log_survival: torch.Tensor


def initialize_greedy_survival_state(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    *,
    hard_alive: bool = True,
) -> GreedySurvivalState:
    """Create an initial survival state (all positions alive, zero log-survival).

    Args:
        batch_size: Number of sequences in the batch.
        sequence_length: Length of each sequence.
        device: Device for the tensors.
        hard_alive: If True (default), initialise a boolean alive mask.
            If False (soft survival), ``alive`` is set to ``None``.
    """
    return GreedySurvivalState(
        alive=(
            torch.ones(
                batch_size,
                sequence_length,
                device=device,
                dtype=torch.bool,
            )
            if hard_alive
            else None
        ),
        log_survival=torch.zeros(
            batch_size,
            sequence_length,
            device=device,
            dtype=torch.float32,
        ),
    )


def greedy_survival_step(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    state: GreedySurvivalState,
    temperature: float,
    hard_alive: bool = True,
    leaky_eta: float = 0.0,
):
    """Compute one differentiable greedy-survival step.

    Args:
        logits:
            Base draft logits, shape [B, S, V].
        target_ids:
            Target greedy ids in draft-vocab space, shape [B, S].
        valid_mask:
            Positions whose full-vocab target top-1 is contained in the
            draft vocabulary and whose dataset loss mask is valid.
        state:
            Hard-alive mask and accumulated soft log-survival.
        temperature:
            Sigmoid temperature for the top-1 margin surrogate.
        hard_alive:
            If True (default), use the hard alive mask to zero out
            contributions from rejected suffix positions.  If False
            (soft survival), all valid positions contribute and gradient
            flows through the entire chain.
        leaky_eta:
            Leaky factor for the hard-alive mask (only used when
            ``hard_alive=True``).  When 0 (default), the mask is purely
            hard.  When >0, rejected suffixes retain a small weight η::

                H̃_{d-1} = H_{d-1} + η(1 - H_{d-1})

            This preserves the chain-aware inductive bias while allowing
            the model to learn from a small fraction of counterfactual
            suffix signal.  Recommended: 0.05–0.1.

    Returns:
        contribution:
            Differentiable soft accepted-token numerator for this depth.
        capacity:
            Number of possible accepted tokens at this depth.  Does not
            use the alive mask, so early rejection remains penalised.
        next_state:
            State for the next draft depth.
        diagnostics:
            Detached tensors used for logging/tests.
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be positive, got {temperature}"
        )

    if logits.dim() != 3:
        raise ValueError(
            f"logits must be [B, S, V], got {tuple(logits.shape)}"
        )

    logits_fp32 = logits.float()
    valid_mask = valid_mask.bool()

    target_logits = logits_fp32.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    # Efficiently find max_{y != target} z(y) without cloning/masking
    # the complete [B, S, V] tensor.
    top2_values, top2_ids = torch.topk(
        logits_fp32,
        k=2,
        dim=-1,
    )

    best_other = torch.where(
        top2_ids[..., 0].eq(target_ids),
        top2_values[..., 1],
        top2_values[..., 0],
    )

    margin = target_logits - best_other
    log_accept = F.logsigmoid(margin / temperature)

    # Invalid positions must not alter accumulated survival.
    next_log_survival = state.log_survival + torch.where(
        valid_mask,
        log_accept,
        torch.zeros_like(log_accept),
    )

    soft_survival = torch.exp(next_log_survival)

    # --- Contribution computation ---
    # hard_alive mode: zero out rejected suffix (no gradient through
    #   the hard alive mask).
    #   With leaky_eta > 0, rejected suffixes retain weight η instead
    #   of zero, allowing gradient from counterfactual suffixes while
    #   preserving chain-aware inductive bias.  The alive propagation
    #   remains hard regardless of eta.
    # soft mode: all valid positions contribute, allowing gradient
    #   to flow through rejected suffix positions.
    if hard_alive:
        eligible = state.alive & valid_mask
        if leaky_eta > 0:
            # H̃ = H + η(1-H), but ONLY for valid positions that are
            # dead due to hard rejection — NOT for padding / OOV positions
            # (they must remain zero so numerator/denominator stay aligned).
            # dead_but_valid = positions that are valid but already
            # hard-rejected (not padding, not OOV).
            dead_but_valid = (~state.alive) & valid_mask
            eligible_weight = (
                eligible.float()
                + leaky_eta * dead_but_valid.float()
            )  # [B, S]
        else:
            eligible_weight = eligible.to(soft_survival.dtype)
        contribution = (
            soft_survival * eligible_weight.detach()
        ).sum()
    else:
        contribution = (
            soft_survival * valid_mask.to(soft_survival.dtype)
        ).sum()

    # Do not use `eligible` in the denominator.  Rejected suffix positions
    # represent lost accepted tokens and therefore remain in the capacity.
    capacity = valid_mask.float().sum()

    hard_prediction = logits_fp32.argmax(dim=-1)
    hard_accept = hard_prediction.eq(target_ids)

    # --- Alive mask update ---
    if hard_alive:
        # No gradient is intended through discrete rollout or the
        # hard-alive mask.
        eligible_for_alive = state.alive & valid_mask
        next_alive = eligible_for_alive & hard_accept.detach()
    else:
        next_alive = None

    next_state = GreedySurvivalState(
        alive=next_alive,
        log_survival=next_log_survival,
    )

    diagnostics = {
        "margin": margin.detach(),
        "soft_accept": torch.exp(log_accept).detach(),
        "soft_survival": soft_survival.detach(),
        "hard_accept": hard_accept.detach(),
        "eligible": (
            (state.alive & valid_mask).detach()
            if state.alive is not None
            else valid_mask.detach()
        ),
    }

    # Return margin WITH gradient separately so that callers (e.g.,
    # Frontier Loss) can back-propagate through it without coupling
    # the training graph into the diagnostics dict.
    return contribution, capacity, next_state, diagnostics, margin


def block_survival_loss(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    hard_alive: bool = True,
    leaky_eta: float = 0.0,
):
    """Vectorized block-wise Survival Loss for parallel speculative drafters.

    Whereas :func:`greedy_survival_step` is a sequential, step-wise API
    designed for autoregressive drafters like EagleSpark, this function
    computes the entire survival chain in one pass using ``cumprod``/``cumsum``
    along the block dimension.  It is the natural formulation for parallel
    drafters like DFlash that produce logits for all positions in a block
    simultaneously.

    The loss is::

        L_block-surv = 1 - (Σ_d H_{d-1} · Ŝ_d · M_d) / (Σ_d M_d + ε)

    where:

    -  ``Ŝ_d = Π_{j=1}^{d} σ(m_j / τ)``  is the **soft survival probability**,
       computed via ``cumsum(log_accept, dim=-1)`` then ``exp``.
    -  ``m_d = z_d(y*_d) - max_{y ≠ y*_d} z_d(y)`` is the **greedy margin**.
    -  ``H_d = Π_{j=1}^{d} 1[argmax(z_j) = y*_j]`` is the **hard alive mask**
       (detached, computed via ``cumprod`` of hard accepts along the block dim).
    -  ``M_d`` is the valid token mask (``valid_mask``).

    The loss is differentiable w.r.t. ``logits`` through the soft survival
    path (sigmoid margin).  The hard alive mask is detached — gradient does
    not flow through discrete argmax decisions.

    Args:
        logits:
            Draft logits, shape ``[B, N_blocks, K, V]`` where K is the
            block size and V is the vocabulary size.
        target_ids:
            Target token ids (ground truth), shape ``[B, N_blocks, K]``.
        valid_mask:
            Valid position mask, shape ``[B, N_blocks, K]``.  Positions
            with ``valid_mask = 0`` are excluded from both numerator and
            denominator.  The caller is responsible for ensuring this mask
            represents a **contiguous supervised prefix** within each block
            (e.g., using ``cumprod`` truncation like DSpark's ``eval_mask``).
        temperature:
            Sigmoid temperature ``τ > 0`` for the margin surrogate.
        hard_alive:
            If True (default), use the hard alive mask to zero out
            contributions from rejected suffix positions.  If False
            (soft survival), all valid positions contribute.
        leaky_eta:
            Leaky factor for the hard-alive mask (only used when
            ``hard_alive=True``).  When 0 (default), the mask is purely
            hard.  When >0, rejected suffixes retain weight η.  See
            :func:`greedy_survival_step` for details.

    Returns:
        survival_loss:
            Scalar loss ``1 - score``, differentiable w.r.t. ``logits``.
        diagnostics:
            Dict of detached tensors for logging:
            ``margin``, ``soft_accept``, ``soft_survival``, ``hard_accept``
            (per-position), ``mean_margin``, ``mean_hard_accept``,
            ``hard_accept_length`` (scalars).

    Raises:
        ValueError: If ``temperature <= 0`` or ``logits`` is not 4-D.
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be positive, got {temperature}"
        )

    if logits.dim() != 4:
        raise ValueError(
            f"logits must be [B, N_blocks, K, V], got {tuple(logits.shape)}"
        )

    logits_fp32 = logits.float()
    valid_mask_bool = valid_mask.bool()
    valid_mask_f = valid_mask_bool.to(torch.float32)

    # ---- Margin m_d = z(y*) - max_{y≠y*} z(y) ----
    target_logits = logits_fp32.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)  # [B, N, K]

    top2_values, top2_ids = torch.topk(
        logits_fp32,
        k=2,
        dim=-1,
    )

    best_other = torch.where(
        top2_ids[..., 0].eq(target_ids),
        top2_values[..., 1],
        top2_values[..., 0],
    )  # [B, N, K]

    margin = target_logits - best_other  # [B, N, K]

    # ---- Soft accept â_d = σ(m_d / τ) ----
    log_accept = F.logsigmoid(margin / temperature)  # [B, N, K]

    # Invalid positions act as multiplicative identity (0 in log-space)
    # so they don't affect the cumulative product.
    masked_log_accept = torch.where(
        valid_mask_bool,
        log_accept,
        torch.zeros_like(log_accept),
    )

    # ---- Soft survival Ŝ_d = Π_{j=1}^{d} â_j ----
    # cumsum in log-space, then exp — numerically stable.
    log_survival = torch.cumsum(masked_log_accept, dim=-1)  # [B, N, K]
    soft_survival = torch.exp(log_survival)  # [B, N, K]

    # ---- Hard alive mask H_{d-1} ----
    # H_{d-1} = 1 if ALL *valid* positions 0..d-1 were hard-accepted.
    # Position d uses H_{d-1} as its eligible weight, so position 0 always
    # uses H_{-1} = 1 (always alive before any prediction).
    #
    # Invalid positions (e.g. the DFlash anchor at pos 0) must act as
    # multiplicative identity in the hard chain — they are NOT rejections,
    # they simply don't participate.  This mirrors how the soft survival
    # already treats invalid positions (zeroed log_accept = identity).
    hard_accept = logits_fp32.argmax(dim=-1).eq(target_ids)  # [B, N, K]
    hard_accept_masked = hard_accept & valid_mask_bool

    if hard_alive:
        # Invalid positions are neutral elements in the hard chain:
        #   (hard_accept OR ~valid) = True for invalid → doesn't break chain.
        # This prevents the DFlash anchor (valid_mask=0) from killing the
        # entire cumprod.
        hard_accept_or_invalid = hard_accept | (~valid_mask_bool)

        # Shift-right cumprod: prepend 1 (H_{-1}=1), compute inclusive
        # prefix over hard_accept_or_invalid from 0..d-1, then shift so
        # that position d gets the cumprod of indices < d.
        prefix_cumprod = torch.cumprod(
            hard_accept_or_invalid.float(), dim=-1
        )  # [B, N, K]

        # Shift right by 1: H_right[d] = prefix[d-1], H_right[0] = 1
        ones = torch.ones(
            prefix_cumprod.shape[0],
            prefix_cumprod.shape[1],
            1,
            device=prefix_cumprod.device,
            dtype=prefix_cumprod.dtype,
        )
        H = torch.cat([ones, prefix_cumprod[:, :, :-1]], dim=-1).bool()  # [B, N, K]

        eligible = H & valid_mask_bool  # [B, N, K]

        if leaky_eta > 0:
            # H̃ = H + η(1-H) for valid-but-dead positions.
            dead_but_valid = (~H) & valid_mask_bool
            eligible_weight = (
                eligible.float()
                + leaky_eta * dead_but_valid.float()
            )  # [B, N, K]
        else:
            eligible_weight = eligible.float()  # [B, N, K]

        # Detach: gradient flows only through soft_survival, not through
        # the discrete hard-alive mask.
        eligible_weight = eligible_weight.detach()

        contribution_per_pos = soft_survival * eligible_weight  # [B, N, K]
    else:
        # Soft survival mode: all valid positions contribute.
        contribution_per_pos = soft_survival * valid_mask_f  # [B, N, K]

    # ---- Numerator / Denominator ----
    numerator = contribution_per_pos.sum()  # scalar, differentiable
    denominator = valid_mask_f.sum()  # scalar, includes rejected suffixes

    survival_score = numerator / denominator.clamp(min=1e-6)
    survival_loss = 1.0 - survival_score

    # ---- Diagnostics ----
    valid_count = valid_mask_f.sum().clamp(min=1.0)

    diagnostics = {
        # Per-position tensors (detached)
        "margin": margin.detach(),
        "soft_accept": torch.exp(log_accept).detach(),
        "soft_survival": soft_survival.detach(),
        "hard_accept": hard_accept.detach(),
        # Scalar summaries
        "mean_margin": (margin.detach() * valid_mask_f).sum() / valid_count,
        "mean_hard_accept": (
            hard_accept.detach().float() * valid_mask_f
        ).sum() / valid_count,
        "hard_accept_length": _mean_hard_accept_length(
            hard_accept.detach(),
            valid_mask_bool.detach(),
        ),
    }

    return survival_loss, diagnostics


def first_rejection_continuation_loss(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.0,
    temperature: float = 1.0,
    continuation_power: float = 1.0,
    reduction: str = "normalized",
    reference_weight_mass: torch.Tensor | None = None,
    mass_match_min: float = 0.0,
    mass_match_max: float = 256.0,
    max_depth: int = 0,
    frontier_scale: torch.Tensor | None = None,
    adaptive_survival_rate: torch.Tensor | None = None,
    adaptive_gamma: float = 1.0,
    adaptive_margin_temperature: float = 1.0,
    adaptive_scale_min: float = 0.25,
    adaptive_scale_max: float = 4.0,
):
    loss, _, diagnostics = first_rejection_continuation_objective(
        logits=logits,
        target_ids=target_ids,
        valid_mask=valid_mask,
        margin=margin,
        temperature=temperature,
        continuation_power=continuation_power,
        reduction=reduction,
        reference_weight_mass=reference_weight_mass,
        mass_match_min=mass_match_min,
        mass_match_max=mass_match_max,
        max_depth=max_depth,
        frontier_scale=frontier_scale,
        adaptive_survival_rate=adaptive_survival_rate,
        adaptive_gamma=adaptive_gamma,
        adaptive_margin_temperature=adaptive_margin_temperature,
        adaptive_scale_min=adaptive_scale_min,
        adaptive_scale_max=adaptive_scale_max,
    )
    return loss, diagnostics


def first_rejection_continuation_objective(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.0,
    temperature: float = 1.0,
    continuation_power: float = 1.0,
    reduction: str = "normalized",
    reference_weight_mass: torch.Tensor | None = None,
    mass_match_min: float = 0.0,
    mass_match_max: float = 256.0,
    max_depth: int = 0,
    frontier_scale: torch.Tensor | None = None,
    adaptive_survival_rate: torch.Tensor | None = None,
    adaptive_gamma: float = 1.0,
    adaptive_margin_temperature: float = 1.0,
    adaptive_scale_min: float = 0.25,
    adaptive_scale_max: float = 4.0,
):
    """Penalize the first rejection, weighted by its lost continuation value.

    Only the first valid position whose greedy prediction differs from the
    target receives gradient.  Its weight is the valid suffix capacity, so an
    early rejection is more expensive because repairing it can recover more
    accepted tokens::

        L_FRC = sum_d R_d C_d softplus((delta - m_d) / tau)
                / (sum_d R_d C_d + eps)

    ``reduction="dpace-matched"`` rescales each sample's frontier credit to
    the detached D-PACE weight mass before applying the batch-size reduction.
    This keeps the auxiliary objective on the same scale as D-PACE, whose
    implementation sums over anchors and depths and divides only by batch.

    The discrete first-rejection mask ``R_d`` and continuation value ``C_d``
    are detached.  Gradients flow only through the target-vs-best-other margin
    ``m_d``.  Invalid positions are neutral to the alive chain, which handles
    DFlash's invalid anchor at depth zero and a contiguous invalid suffix.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if continuation_power < 0:
        raise ValueError(
            "continuation_power must be non-negative, "
            f"got {continuation_power}"
        )
    valid_reductions = {"normalized", "batch", "dpace-matched"}
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {sorted(valid_reductions)}, got {reduction!r}"
        )
    if mass_match_min < 0 or mass_match_max < mass_match_min:
        raise ValueError(
            "mass-match bounds must satisfy 0 <= min <= max, got "
            f"{mass_match_min} and {mass_match_max}"
        )
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")
    if logits.dim() != 4:
        raise ValueError(
            f"logits must be [B, N_blocks, K, V], got {tuple(logits.shape)}"
        )

    logits_fp32 = logits.float()
    valid = valid_mask.bool()
    target_logits = logits_fp32.gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)
    top2_values, top2_ids = torch.topk(logits_fp32, k=2, dim=-1)
    best_other = torch.where(
        top2_ids[..., 0].eq(target_ids),
        top2_values[..., 1],
        top2_values[..., 0],
    )
    target_margin = target_logits - best_other

    hard_accept = top2_ids[..., 0].eq(target_ids)
    accept_or_invalid = hard_accept | (~valid)
    alive_before = torch.ones_like(accept_or_invalid)
    if logits.size(-2) > 1:
        alive_before[..., 1:] = torch.cumprod(
            accept_or_invalid[..., :-1].to(torch.int32), dim=-1
        ).bool()
    first_rejection = alive_before & valid & (~hard_accept)
    if max_depth > 0:
        # DFlash reserves depth zero for the anchor.  A max_depth of three
        # therefore permits repair at draft depths 1, 2, and 3 only, while
        # retaining the full-chain rollout used to identify the true first
        # rejection.
        depth = torch.arange(logits.size(-2), device=logits.device)
        first_rejection = first_rejection & depth.le(int(max_depth)).view(
            *((1,) * (first_rejection.dim() - 1)), -1
        )

    continuation = torch.flip(
        torch.cumsum(torch.flip(valid.float(), dims=[-1]), dim=-1),
        dims=[-1],
    )
    if continuation_power != 1.0:
        continuation = continuation.pow(continuation_power)
    adaptive_diag = {}
    if adaptive_survival_rate is not None:
        adaptive_scale, adaptive_diag = survival_adaptive_frontier_scale(
            target_margin=target_margin,
            first_rejection=first_rejection,
            survival_rate=adaptive_survival_rate,
            gamma=adaptive_gamma,
            margin_target=margin,
            margin_temperature=adaptive_margin_temperature,
            scale_min=adaptive_scale_min,
            scale_max=adaptive_scale_max,
        )
        frontier_scale = (
            adaptive_scale
            if frontier_scale is None
            else frontier_scale * adaptive_scale
        )
    if frontier_scale is not None:
        if frontier_scale.shape != first_rejection.shape:
            raise ValueError(
                "frontier_scale must match logits without vocabulary axis, got "
                f"{tuple(frontier_scale.shape)} and {tuple(first_rejection.shape)}"
            )
        scale = frontier_scale.detach().to(
            device=continuation.device,
            dtype=continuation.dtype,
        )
    else:
        scale = torch.ones_like(continuation)
    credit_weight = (
        first_rejection.float() * continuation * scale
    ).detach()

    per_position = F.softplus(
        (float(margin) - target_margin) / temperature
    )
    reduce_dims = tuple(range(1, credit_weight.dim()))
    credit_mass_per_sample = credit_weight.sum(dim=reduce_dims)
    numerator_per_sample = (per_position * credit_weight).sum(dim=reduce_dims)
    denominator = credit_mass_per_sample.sum()
    numerator = numerator_per_sample.sum()
    batch_size = max(logits.size(0), 1)
    mass_scale = torch.ones_like(credit_mass_per_sample)
    if reduction == "batch":
        loss = numerator / float(batch_size)
    elif reduction == "dpace-matched":
        if reference_weight_mass is None:
            raise ValueError(
                "reference_weight_mass is required for reduction='dpace-matched'"
            )
        if reference_weight_mass.shape != credit_mass_per_sample.shape:
            raise ValueError(
                "reference_weight_mass must have shape [B], got "
                f"{tuple(reference_weight_mass.shape)}"
            )
        mass_scale = (
            reference_weight_mass.detach().float()
            / credit_mass_per_sample.clamp_min(1e-6)
        ).clamp(min=float(mass_match_min), max=float(mass_match_max))
        mass_scale = torch.where(
            credit_mass_per_sample > 0,
            mass_scale,
            torch.zeros_like(mass_scale),
        )
        loss = (numerator_per_sample * mass_scale).sum() / float(batch_size)
    else:
        # Preserve the historical normalized reduction.
        loss = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1.0),
            numerator * 0.0,
        )

    frontier_count = first_rejection.float().sum()
    diagnostics = {
        "margin": target_margin.detach(),
        "hard_accept": hard_accept.detach(),
        "first_rejection": first_rejection.detach(),
        "continuation_value": continuation.detach(),
        "frontier_count": frontier_count.detach(),
        "mean_frontier_margin": (
            (target_margin.detach() * first_rejection.float()).sum()
            / frontier_count.clamp_min(1.0)
        ),
        "mean_continuation": (
            (continuation.detach() * first_rejection.float()).sum()
            / frontier_count.clamp_min(1.0)
        ),
        "credit_mass": denominator.detach(),
        "credit_mass_per_sample": credit_mass_per_sample.detach().mean(),
        "mass_scale_mean": mass_scale.detach().sum()
        / mass_scale.detach().gt(0).float().sum().clamp_min(1.0),
        "mass_scale_max": mass_scale.detach().max(),
        "normalized_loss": torch.where(
            denominator > 0,
            numerator.detach() / denominator.detach().clamp_min(1.0),
            numerator.detach() * 0.0,
        ),
        "batch_reduced_loss": numerator.detach() / float(batch_size),
        **adaptive_diag,
    }
    # Expose the compact live margin only to callers that need gradient
    # surgery.  The public compatibility wrapper above continues returning
    # the historical (loss, diagnostics) pair.
    return loss, target_margin, diagnostics


def survival_adaptive_frontier_scale(
    *,
    target_margin: torch.Tensor,
    first_rejection: torch.Tensor,
    survival_rate: torch.Tensor,
    gamma: float = 1.0,
    margin_target: float = 0.0,
    margin_temperature: float = 1.0,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
):
    """Build detached, mean-one credit scales for shallow FRC frontiers.

    The depth factor focuses residual credit on depths whose empirical chain
    survival is weak, while the margin factor focuses it on rejected tokens
    that remain below the requested verification margin.  Normalising over
    the active frontiers preserves the overall FRC mass; the mechanism changes
    *where* residual credit is spent rather than silently changing its alpha.
    """
    if target_margin.shape != first_rejection.shape:
        raise ValueError(
            "target_margin and first_rejection must have the same shape"
        )
    if target_margin.dim() != 3:
        raise ValueError("frontier tensors must be [B, N_blocks, K]")
    if survival_rate.dim() != 1 or survival_rate.numel() != target_margin.size(-1):
        raise ValueError("survival_rate must have shape [K]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if margin_temperature <= 0:
        raise ValueError("margin_temperature must be positive")
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError("adaptive scales must satisfy 0 < min <= max")

    with torch.no_grad():
        depth_factor = (1.0 - survival_rate.float()).clamp_min(1e-4).pow(gamma)
        depth_factor = depth_factor.view(1, 1, -1)
        margin_factor = torch.sigmoid(
            (float(margin_target) - target_margin.detach().float())
            / float(margin_temperature)
        )
        raw = depth_factor * margin_factor
        frontier = first_rejection.bool()
        active_count = frontier.float().sum()
        active_mean = (raw * frontier.float()).sum() / active_count.clamp_min(1.0)
        normalized = (raw / active_mean.clamp_min(1e-6)).clamp(
            min=float(scale_min), max=float(scale_max)
        )
        scale = torch.where(frontier, normalized, torch.ones_like(normalized))
        scale_mean = (
            (scale * frontier.float()).sum() / active_count.clamp_min(1.0)
        )
        scale_min_value = torch.where(
            frontier, scale, torch.full_like(scale, float("inf"))
        ).amin()
        scale_max_value = torch.where(
            frontier, scale, torch.zeros_like(scale)
        ).amax()
        diagnostics = {
            "adaptive_scale_mean": torch.where(
                active_count > 0, scale_mean, scale_mean.new_ones(())
            ),
            "adaptive_scale_min": torch.where(
                active_count > 0,
                scale_min_value,
                scale_min_value.new_ones(()),
            ),
            "adaptive_scale_max": torch.where(
                active_count > 0,
                scale_max_value,
                scale_max_value.new_ones(()),
            ),
            "adaptive_frontier_count": active_count,
        }
    return scale, diagnostics


def deep_full_accept_margin_quantiles(
    *,
    target_margin: torch.Tensor,
    valid_mask: torch.Tensor,
    hard_accept: torch.Tensor,
    min_depth: int,
    quantile: float,
    fallback: torch.Tensor,
):
    """Estimate per-depth safety floors from currently full-accepted blocks."""
    if target_margin.shape != valid_mask.shape or target_margin.shape != hard_accept.shape:
        raise ValueError("deep-chain tensors must have matching shapes")
    if target_margin.dim() != 3:
        raise ValueError("deep-chain tensors must be [B, N_blocks, K]")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if fallback.dim() != 1 or fallback.numel() != target_margin.size(-1):
        raise ValueError("fallback must have shape [K]")

    with torch.no_grad():
        valid = valid_mask.bool()
        valid_depth = valid.sum(dim=-1)
        full_accept = (
            (hard_accept.bool() | (~valid)).all(dim=-1)
            & valid_depth.ge(int(min_depth))
        )
        floors = fallback.detach().float().clone()
        counts = torch.zeros_like(floors)
        for depth in range(target_margin.size(-1)):
            selected = valid[..., depth] & full_accept
            values = target_margin.detach().float()[..., depth][selected]
            counts[depth] = float(values.numel())
            if values.numel() > 0:
                floors[depth] = torch.quantile(values, float(quantile))
    return floors, counts


def deep_chain_dpace_anchor_loss(
    *,
    target_margin: torch.Tensor,
    dpace_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    hard_accept: torch.Tensor,
    min_depth: int,
    margin_floor: float | torch.Tensor = 0.0,
    temperature: float = 1.0,
):
    """Reinforce D-PACE on blocks that currently survive a deep full chain.

    FRC acts only on rejected blocks, but its shared-parameter update can still
    erode margins on blocks that are already fully accepted.  This anchor
    selects blocks whose complete valid chain is greedily accepted and whose
    valid draft depth is at least ``min_depth``.  It then reuses the original
    D-PACE-weighted margin-floor loss on those blocks.  The selector is
    discrete/detached; gradients flow only through ``target_margin``.  A
    margin floor avoids spending much protection capacity on already robust
    deep chains while reacting strongly when a previously accepted token is
    close to the greedy decision boundary.

    The batch reduction deliberately matches D-PACE's sum-over-blocks divided
    by batch-size convention, so the protection coefficient is interpretable
    as an additional relative weight on already-good deep chains.
    """
    if target_margin.shape != dpace_weights.shape:
        raise ValueError(
            "target_margin and dpace_weights must have the same shape, got "
            f"{tuple(target_margin.shape)} and {tuple(dpace_weights.shape)}"
        )
    if target_margin.shape != valid_mask.shape:
        raise ValueError(
            "target_margin and valid_mask must have the same shape, got "
            f"{tuple(target_margin.shape)} and {tuple(valid_mask.shape)}"
        )
    if target_margin.shape != hard_accept.shape:
        raise ValueError(
            "target_margin and hard_accept must have the same shape, got "
            f"{tuple(target_margin.shape)} and {tuple(hard_accept.shape)}"
        )
    if target_margin.dim() != 3:
        raise ValueError(
            "deep-chain tensors must be [B, N_blocks, K], got "
            f"{tuple(target_margin.shape)}"
        )
    if min_depth <= 0:
        raise ValueError(f"min_depth must be positive, got {min_depth}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    valid = valid_mask.bool()
    accepted_or_invalid = hard_accept.bool() | (~valid)
    valid_depth = valid.sum(dim=-1)
    full_accept_block = (
        accepted_or_invalid.all(dim=-1) & valid_depth.ge(int(min_depth))
    ).detach()
    protected_mask = (valid & full_accept_block.unsqueeze(-1)).detach()
    protected_weights = dpace_weights * protected_mask.to(dpace_weights.dtype)
    if isinstance(margin_floor, torch.Tensor):
        if margin_floor.dim() != 1 or margin_floor.numel() != target_margin.size(-1):
            raise ValueError("tensor margin_floor must have shape [K]")
        floor = margin_floor.to(
            device=target_margin.device, dtype=target_margin.dtype
        ).view(1, 1, -1)
    else:
        floor = float(margin_floor)
    per_position = F.softplus(
        (floor - target_margin) / float(temperature)
    )
    batch_size = max(target_margin.size(0), 1)
    loss = (per_position * protected_weights).sum() / float(batch_size)

    protected_block_count = full_accept_block.float().sum()
    protected_token_count = protected_mask.float().sum()
    eligible_block_count = valid_depth.ge(int(min_depth)).float().sum()
    diagnostics = {
        "protected_block_count": protected_block_count,
        "protected_token_count": protected_token_count,
        "eligible_block_count": eligible_block_count,
        "protected_block_fraction": (
            protected_block_count / eligible_block_count.clamp_min(1.0)
        ),
        "protected_weight_mass": protected_weights.detach().sum(),
        "protected_mean_margin": (
            (target_margin.detach() * protected_mask.float()).sum()
            / protected_token_count.clamp_min(1.0)
        ),
    }
    return loss, diagnostics


def first_rejection_boundary_loss(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
    focal_gamma: float = 2.0,
):
    """Optimize the *soft* first-rejection boundary of a parallel block.

    Greedy speculative verification stops at the first position whose target
    token loses the top-1 decision.  For target-vs-best-other margin ``m_d``,
    this objective uses ``a_d = sigmoid(m_d / temperature)`` as a smooth
    acceptance proxy and assigns position ``d`` the detached probability that
    all earlier valid positions survive::

        B_d = stopgrad(prod_{j < d} a_j)
        L_boundary = sum_d B_d (1-a_d)^gamma [-log(a_d)] / sum_d B_d

    Detaching ``B_d`` prevents the model from reducing a later position's
    credit by making an earlier position worse.  Unlike the hard FRC loss,
    every block supplies a useful boundary signal before and after its current
    discrete first rejection.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if focal_gamma < 0:
        raise ValueError(f"focal_gamma must be non-negative, got {focal_gamma}")
    if logits.dim() != 4:
        raise ValueError(
            f"logits must be [B, N_blocks, K, V], got {tuple(logits.shape)}"
        )

    logits_fp32 = logits.float()
    valid = valid_mask.bool()
    valid_f = valid.float()
    target_logits = logits_fp32.gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)
    top2_values, top2_ids = torch.topk(logits_fp32, k=2, dim=-1)
    best_other = torch.where(
        top2_ids[..., 0].eq(target_ids),
        top2_values[..., 1],
        top2_values[..., 0],
    )
    target_margin = target_logits - best_other

    log_accept = F.logsigmoid(target_margin / temperature)
    soft_accept = log_accept.exp()
    accept_or_identity = torch.where(
        valid, soft_accept, torch.ones_like(soft_accept)
    )
    inclusive_survival = torch.cumprod(accept_or_identity, dim=-1)
    alive_before = torch.ones_like(inclusive_survival)
    if logits.size(-2) > 1:
        alive_before[..., 1:] = inclusive_survival[..., :-1]

    boundary_credit = (alive_before * valid_f).detach()
    focal = (1.0 - soft_accept).pow(focal_gamma)
    per_position = focal * (-log_accept)
    credit_mass = boundary_credit.sum()
    loss = torch.where(
        credit_mass > 0,
        (per_position * boundary_credit).sum() / credit_mass.clamp_min(1.0),
        per_position.sum() * 0.0,
    )

    with torch.no_grad():
        hard_accept = top2_ids[..., 0].eq(target_ids)
        accept_or_invalid = hard_accept | (~valid)
        hard_alive_before = torch.ones_like(accept_or_invalid)
        if logits.size(-2) > 1:
            hard_alive_before[..., 1:] = torch.cumprod(
                accept_or_invalid[..., :-1].to(torch.int32), dim=-1
            ).bool()
        hard_frontier = hard_alive_before & valid & (~hard_accept)
        valid_count = valid_f.sum().clamp_min(1.0)
        diagnostics = {
            "margin": target_margin.detach(),
            "soft_accept": soft_accept.detach(),
            "alive_before": alive_before.detach(),
            "boundary_credit": boundary_credit.detach(),
            "credit_mass": credit_mass.detach(),
            "hard_frontier_count": hard_frontier.float().sum(),
            "mean_margin": (target_margin * valid_f).sum() / valid_count,
            "mean_soft_accept": (soft_accept * valid_f).sum() / valid_count,
        }
    return loss, diagnostics


def frbo_objective(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
    focal_gamma: float = 2.0,
):
    """Compute FRBO survival and boundary losses from one top-2 margin pass.

    The returned live margin is a compact ``[B, N, K]`` interface used by the
    optional conflict projection.  This avoids materializing multiple
    vocabulary-sized gradients for ``[B, N, K, V]`` logits.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if focal_gamma < 0:
        raise ValueError(f"focal_gamma must be non-negative, got {focal_gamma}")
    if logits.dim() != 4:
        raise ValueError(
            f"logits must be [B, N_blocks, K, V], got {tuple(logits.shape)}"
        )

    logits_fp32 = logits.float()
    valid = valid_mask.bool()
    valid_f = valid.float()
    target_logits = logits_fp32.gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)
    top2_values, top2_ids = torch.topk(logits_fp32, k=2, dim=-1)
    best_other = torch.where(
        top2_ids[..., 0].eq(target_ids),
        top2_values[..., 1],
        top2_values[..., 0],
    )
    target_margin = target_logits - best_other
    log_accept = F.logsigmoid(target_margin / temperature)
    soft_accept = log_accept.exp()
    accept_or_identity = torch.where(
        valid, soft_accept, torch.ones_like(soft_accept)
    )
    soft_survival = torch.cumprod(accept_or_identity, dim=-1)

    valid_count = valid_f.sum().clamp_min(1.0)
    survival_score = (soft_survival * valid_f).sum() / valid_count
    survival_loss = 1.0 - survival_score

    alive_before = torch.ones_like(soft_survival)
    if logits.size(-2) > 1:
        alive_before[..., 1:] = soft_survival[..., :-1]
    boundary_credit = (alive_before * valid_f).detach()
    credit_mass = boundary_credit.sum()
    focal = (1.0 - soft_accept).pow(focal_gamma)
    boundary_loss = torch.where(
        credit_mass > 0,
        (boundary_credit * focal * (-log_accept)).sum()
        / credit_mass.clamp_min(1.0),
        target_margin.sum() * 0.0,
    )

    with torch.no_grad():
        hard_accept = top2_ids[..., 0].eq(target_ids)
        active_blocks = valid.any(dim=-1).float().sum().clamp_min(1.0)
        hard_accept_length = _mean_hard_accept_length(hard_accept, valid)
        diagnostics = {
            "soft_accept": soft_accept.detach(),
            "soft_survival": soft_survival.detach(),
            "boundary_credit": boundary_credit.detach(),
            "boundary_credit_mass": credit_mass.detach(),
            "mean_soft_accept": (soft_accept * valid_f).sum() / valid_count,
            "soft_accept_length": (soft_survival * valid_f).sum()
            / active_blocks,
            "hard_accept_length": hard_accept_length,
        }
    return survival_loss, boundary_loss, target_margin, diagnostics


def project_conflicting_gradient(
    primary_grad: torch.Tensor,
    auxiliary_grad: torch.Tensor,
):
    """Project an auxiliary gradient away from a conflicting primary gradient.

    The projection is performed at a shared differentiable interface (FRBO
    uses the compact verification margin).  It preserves the auxiliary
    gradient when the dot product is non-negative and otherwise removes its
    component along the primary gradient.
    """
    primary_fp32 = primary_grad.float()
    auxiliary_fp32 = auxiliary_grad.float()
    dot = (primary_fp32 * auxiliary_fp32).sum()
    primary_norm_sq = primary_fp32.square().sum().clamp_min(1e-12)
    auxiliary_norm = auxiliary_fp32.square().sum().sqrt()
    primary_norm = primary_norm_sq.sqrt()
    cosine = dot / (primary_norm * auxiliary_norm).clamp_min(1e-12)
    conflict = dot.lt(0).float()
    projection = torch.clamp(dot, max=0.0) / primary_norm_sq
    projected = auxiliary_fp32 - projection * primary_fp32
    return projected.to(auxiliary_grad.dtype), cosine, conflict


def counterfactual_frontier_utility_loss(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str = "soft",
    frontier_temperature: float = 1.0,
    loss_temperature: float = 1.0,
    target_margin: float = 0.0,
    continuation_power: float = 1.0,
    utility_transform: str = "linear",
    advantage_beta: float = 0.0,
    advantage_temperature: float = 0.5,
    advantage_min_factor: float = 0.75,
    advantage_max_factor: float = 1.25,
):
    """Repair the verification frontier using observed suffix utility.

    This objective is independent of D-PACE.  It estimates the probability
    that depth ``d`` is the first rejection and weights a greedy-margin repair
    loss by the number of tokens that would become accepted if ``d`` were
    repaired while the already-produced parallel suffix stayed fixed.

    Modes:

    - ``frontier``: soft first-rejection credit without suffix utility.  This
      isolates whether gains come merely from frontier localization.
    - ``hard``: exact detached first-rejection indicator times observed
      recoverable suffix utility.
    - ``soft``: soft first-rejection probability times observed recoverable
      suffix utility (the main CFU objective).
    - ``expected``: soft first-rejection probability times a smooth expected
      continuation baseline inspired by D-PACE.
    - ``advantage``: expected credit with a bounded observed-vs-expected
      counterfactual advantage correction.

    The credit is detached.  Gradients flow only through the target-vs-best-
    other margin loss, preventing the model from manipulating its own weight.
    """
    valid_modes = {"frontier", "hard", "soft", "expected", "advantage"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")
    if frontier_temperature <= 0:
        raise ValueError("frontier_temperature must be positive")
    if loss_temperature <= 0:
        raise ValueError("loss_temperature must be positive")
    if continuation_power < 0:
        raise ValueError("continuation_power must be non-negative")
    if utility_transform not in {"linear", "log1p"}:
        raise ValueError("utility_transform must be 'linear' or 'log1p'")
    if advantage_beta < 0:
        raise ValueError("advantage_beta must be non-negative")
    if advantage_temperature <= 0:
        raise ValueError("advantage_temperature must be positive")
    if advantage_min_factor <= 0:
        raise ValueError("advantage_min_factor must be positive")
    if advantage_max_factor < advantage_min_factor:
        raise ValueError(
            "advantage_max_factor must be >= advantage_min_factor"
        )
    if logits.dim() != 4:
        raise ValueError(
            f"logits must be [B, N_blocks, K, V], got {tuple(logits.shape)}"
        )

    logits_fp32 = logits.float()
    valid = valid_mask.bool()
    target_logits = logits_fp32.gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)
    top2_values, top2_ids = torch.topk(logits_fp32, k=2, dim=-1)
    best_other = torch.where(
        top2_ids[..., 0].eq(target_ids),
        top2_values[..., 1],
        top2_values[..., 0],
    )
    target_margin_value = target_logits - best_other

    with torch.no_grad():
        hard_accept = top2_ids[..., 0].eq(target_ids)
        accept_or_invalid = hard_accept | (~valid)
        hard_alive_before = torch.ones_like(accept_or_invalid)
        if logits.size(-2) > 1:
            hard_alive_before[..., 1:] = torch.cumprod(
                accept_or_invalid[..., :-1].to(torch.int32), dim=-1
            ).bool()
        hard_frontier = hard_alive_before & valid & (~hard_accept)

        soft_accept = torch.sigmoid(
            target_margin_value.detach() / frontier_temperature
        )
        soft_accept_or_identity = torch.where(
            valid, soft_accept, torch.ones_like(soft_accept)
        )
        soft_reach_before = torch.ones_like(soft_accept)
        if logits.size(-2) > 1:
            soft_reach_before[..., 1:] = torch.cumprod(
                soft_accept_or_identity[..., :-1], dim=-1
            )
        soft_frontier = (
            soft_reach_before * (1.0 - soft_accept) * valid.float()
        )

        # C_d^+ = 1 plus the consecutive already-correct run after d.
        continuation = torch.zeros_like(soft_accept, dtype=torch.float32)
        running = torch.zeros_like(soft_accept[..., 0], dtype=torch.float32)
        for depth in range(logits.size(-2) - 1, -1, -1):
            if depth == logits.size(-2) - 1:
                future = torch.zeros_like(running)
            else:
                next_good = (
                    valid[..., depth + 1] & hard_accept[..., depth + 1]
                )
                future = torch.where(next_good, running, torch.zeros_like(running))
            running = torch.where(
                valid[..., depth], 1.0 + future, torch.zeros_like(future)
            )
            continuation[..., depth] = running

        # E_d = 1 + a_{d+1} E_{d+1}: a smooth continuation baseline.  This
        # borrows D-PACE's low-variance expectation idea without replacing the
        # base DFlash CE with D-PACE token weights.
        expected_continuation = torch.zeros_like(soft_accept)
        for depth in range(logits.size(-2) - 1, -1, -1):
            if depth == logits.size(-2) - 1:
                expected_future = torch.zeros_like(running)
            else:
                expected_future = torch.where(
                    valid[..., depth + 1],
                    soft_accept[..., depth + 1]
                    * expected_continuation[..., depth + 1],
                    torch.zeros_like(running),
                )
            expected_continuation[..., depth] = torch.where(
                valid[..., depth],
                1.0 + expected_future,
                torch.zeros_like(running),
            )

        if utility_transform == "log1p":
            utility = torch.log1p(continuation)
            expected_utility = torch.log1p(expected_continuation)
        else:
            utility = continuation
            expected_utility = expected_continuation
        if continuation_power != 1.0:
            utility = utility.pow(continuation_power)
            expected_utility = expected_utility.pow(continuation_power)

        relative_advantage = (
            (utility - expected_utility)
            / expected_utility.clamp_min(1e-6)
        )
        bounded_advantage = torch.tanh(
            relative_advantage / advantage_temperature
        )
        raw_advantage_factor = 1.0 + advantage_beta * bounded_advantage
        advantage_factor = raw_advantage_factor.clamp(
            min=advantage_min_factor,
            max=advantage_max_factor,
        )

        if mode == "frontier":
            frontier_credit = soft_frontier
            utility_for_credit = torch.ones_like(utility)
        elif mode == "hard":
            frontier_credit = hard_frontier.float()
            utility_for_credit = utility
        elif mode == "soft":
            frontier_credit = soft_frontier
            utility_for_credit = utility
        elif mode == "expected":
            frontier_credit = soft_frontier
            utility_for_credit = expected_utility
        else:
            frontier_credit = soft_frontier
            utility_for_credit = expected_utility * advantage_factor
        credit = (frontier_credit * utility_for_credit).detach()

    per_position = F.softplus(
        (float(target_margin) - target_margin_value) / loss_temperature
    )
    credit_mass = credit.sum()
    weighted_loss = (per_position * credit).sum()
    # Keep a valid autograd graph for hard-CFU batches with no rejection.
    loss = torch.where(
        credit_mass > 0,
        weighted_loss / credit_mass.clamp_min(1e-6),
        weighted_loss * 0.0,
    )

    valid_count = valid.float().sum().clamp_min(1.0)
    hard_frontier_count = hard_frontier.float().sum()
    soft_frontier_mass = soft_frontier.sum()
    active_blocks = credit.sum(dim=-1).gt(0).float()
    observed_mean = (utility * valid.float()).sum() / valid_count
    expected_mean = (expected_utility * valid.float()).sum() / valid_count
    observed_centered = utility - observed_mean
    expected_centered = expected_utility - expected_mean
    continuation_covariance = (
        observed_centered * expected_centered * valid.float()
    ).sum() / valid_count
    observed_variance = (
        observed_centered.square() * valid.float()
    ).sum() / valid_count
    expected_variance = (
        expected_centered.square() * valid.float()
    ).sum() / valid_count
    continuation_correlation = continuation_covariance / (
        torch.sqrt(observed_variance * expected_variance).clamp_min(1e-6)
    )
    advantage_mean = (
        relative_advantage * valid.float()
    ).sum() / valid_count
    advantage_variance = (
        (relative_advantage - advantage_mean).square() * valid.float()
    ).sum() / valid_count
    advantage_clip_fraction = (
        raw_advantage_factor.ne(advantage_factor).float() * valid.float()
    ).sum() / valid_count
    credit_effective_sample_size = credit_mass.square() / (
        credit.square().sum().clamp_min(1e-6)
    )
    diagnostics = {
        "margin": target_margin_value.detach(),
        "hard_accept": hard_accept.detach(),
        "hard_frontier": hard_frontier.detach(),
        "soft_frontier": soft_frontier.detach(),
        "continuation_value": continuation.detach(),
        "expected_continuation_value": expected_continuation.detach(),
        "expected_utility": expected_utility.detach(),
        "relative_advantage": relative_advantage.detach(),
        "advantage_factor": advantage_factor.detach(),
        "credit": credit.detach(),
        "credit_mass": credit_mass.detach(),
        "hard_frontier_count": hard_frontier_count.detach(),
        "soft_frontier_mass": soft_frontier_mass.detach(),
        "active_block_fraction": active_blocks.mean().detach(),
        "mean_margin": (
            target_margin_value.detach() * valid.float()
        ).sum() / valid_count,
        "mean_frontier_margin": (
            target_margin_value.detach() * hard_frontier.float()
        ).sum() / hard_frontier_count.clamp_min(1.0),
        "mean_frontier_continuation": (
            continuation.detach() * hard_frontier.float()
        ).sum() / hard_frontier_count.clamp_min(1.0),
        "mean_observed_utility": observed_mean.detach(),
        "mean_expected_utility": expected_mean.detach(),
        "continuation_correlation": continuation_correlation.detach(),
        "advantage_mean": advantage_mean.detach(),
        "advantage_std": torch.sqrt(advantage_variance).detach(),
        "advantage_clip_fraction": advantage_clip_fraction.detach(),
        "credit_effective_sample_size": (
            credit_effective_sample_size.detach()
        ),
    }
    return loss, diagnostics


def _mean_hard_accept_length(
    hard_accept: torch.Tensor,
    valid_mask_bool: torch.Tensor,
) -> torch.Tensor:
    """Compute mean hard-accept chain length per block, averaged over all blocks.

    Invalid positions (e.g. the DFlash anchor) are treated as neutral
    elements in the chain — they don't break it — and then masked out
    from the length count.

    Args:
        hard_accept: Bool tensor ``[B, N, K]`` — argmax == target.
        valid_mask_bool: Bool tensor ``[B, N, K]`` — valid positions.

    Returns:
        Scalar tensor (detached) — mean chain length across all blocks.
    """
    hard_accept_or_invalid = hard_accept | (~valid_mask_bool)
    prefix = torch.cumprod(hard_accept_or_invalid.float(), dim=-1)  # [B, N, K]
    # Only count valid positions in the chain length.
    accepted_prefix = prefix * valid_mask_bool.float()
    chain_length = accepted_prefix.sum(dim=-1)  # [B, N]
    total_blocks = chain_length.numel()
    if total_blocks == 0:
        return torch.tensor(0.0)
    return chain_length.sum() / total_blocks
