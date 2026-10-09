"""Training-only vocabulary-integrated BV objective (arXiv:2609.34832, B.4).

This is a full-offline CARH loss replacement experiment, not the paper's recipe
or an exact objective for SGLang's token verifier. Target-path identities require
the stored paths to have been sampled from the same target distribution used
here. No on-policy rollout, sampling operator, or serving parameters are added.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from specforge.legacy.dspark_training.options import add_bv_args as add_bv_args




def validate_bv_options(alpha, temperature, anneal_ratio, chunk_size,
                        l1_alpha=0.0, offline_objective="none", vat_enabled=False):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("bv-loss-alpha must be finite and non-negative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("bv-temperature must be finite and positive")
    if not math.isfinite(anneal_ratio) or not 0 <= anneal_ratio <= 1:
        raise ValueError("bv-anneal-ratio must be in [0, 1]")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("bv-block-chunk-size must be a positive integer")
    if alpha > 0 and (l1_alpha != 0 or offline_objective != "none" or vat_enabled):
        raise ValueError("BV replaces L1: require l1-loss-alpha=0, offline-acceptance-objective=none, VAT disabled")


def validate_bv_args(args):
    validate_bv_options(args.bv_loss_alpha, args.bv_temperature, args.bv_anneal_ratio,
                        args.bv_block_chunk_size, args.l1_loss_alpha,
                        args.offline_acceptance_objective, getattr(args, "vat_enabled", False))


def validate_bv_resume(saved_args, args):
    saved = saved_args if isinstance(saved_args, dict) else vars(saved_args)
    if saved.get("bv_loss_alpha", 0.0) > 0 or args.bv_loss_alpha > 0:
        # A loss change is a new experiment, not a silent optimizer-state resume.
        for key in ("bv_loss_alpha", "bv_temperature", "bv_anneal_ratio",
                    "ce_loss_alpha", "l1_loss_alpha", "num_epochs", "max_steps"):
            if saved.get(key) != getattr(args, key):
                raise ValueError(f"BV resume setting changed: {key}")


def bv_beta(step, total_steps, anneal_ratio):
    if anneal_ratio == 0:
        return 1.0
    if total_steps <= 0:
        return 0.0
    return min(max(step / (total_steps * anneal_ratio), 0.0), 1.0)


class BVTerms(NamedTuple):
    numerator: torch.Tensor
    denominator: torch.Tensor
    length_sum: torch.Tensor
    block_log_sum: torch.Tensor
    floored_positions: torch.Tensor
    valid_positions: torch.Tensor
    overlap: torch.Tensor
    target_entropy: torch.Tensor


def _bv_chunk(draft_logits, target_logits, labels, valid, beta, temperature):
    # Each row is an independent block; slot zero predicts anchor+1, not anchor.
    # Invalid tails must neither enter cumulative ratios nor receive gradients.
    logq = F.log_softmax(
        draft_logits.float().masked_fill(~valid[..., None], 0.0) / temperature, dim=-1)
    logp = F.log_softmax(
        target_logits.detach().float().masked_fill(~valid[..., None], 0.0) / temperature, dim=-1)
    safe_labels = labels.masked_fill(~valid, 0)
    log_ratio = (logq.gather(-1, safe_labels[..., None]).squeeze(-1)
                 - logp.gather(-1, safe_labels[..., None]).squeeze(-1))
    log_ratio = log_ratio.masked_fill(~valid, 0.0)
    log_r = log_ratio.cumsum(-1)
    log_r_before = F.pad(log_r[..., :-1], (1, 0), value=0.0)
    # Sampled-path states, NOT integrated scores, propagate to the next depth.
    log_a_before = log_r_before.cummin(-1).values
    log_score = torch.logsumexp(torch.minimum(
        log_a_before[..., None] + logp, log_r_before[..., None] + logq), dim=-1)
    # Explicit numerical safeguard for this adaptation, reported in diagnostics.
    floored = ((log_score < -80.0) & valid).sum(-1).float()
    log_score = log_score.clamp(min=-80.0, max=0.0).masked_fill(~valid, 0.0)
    count = valid.sum(-1).clamp_min(1).float()
    active = valid.any(-1)
    mean_log = log_score.sum(-1) / count
    weights = torch.softmax((beta * log_score.detach()).masked_fill(~valid, -1e9), dim=-1)
    weights = weights * valid
    surrogate = -(weights * log_score).sum(-1)
    if beta == 0:
        value = -mean_log
    else:
        # log1p/expm1 avoids cancellation near beta=0. Scores lie in [-80,0].
        centered = (log_score - mean_log[..., None]).masked_fill(~valid, 0.0)
        value = -(mean_log + torch.log1p(torch.expm1(beta * centered).sum(-1) / count) / beta)
    loss = (surrogate + (value - surrogate).detach()) * active
    with torch.no_grad():
        log_sum = torch.logsumexp(log_score.masked_fill(~valid, -1e9), dim=-1)
        block_log = (count.log() - log_sum) * active
        length = (log_score.exp() * valid).sum(-1)
        # Reuse these small diagnostics for the existing confidence/anchor logic;
        # do not retain a second full-vocabulary L1 backward graph.
        if temperature != 1.0:
            # Existing confidence/anchor diagnostics stay at T=1 even when the
            # BV training temperature is changed explicitly.
            logp = F.log_softmax(target_logits.detach().float().masked_fill(~valid[..., None], 0), dim=-1)
            logq = F.log_softmax(draft_logits.detach().float().masked_fill(~valid[..., None], 0), dim=-1)
        overlap = torch.logsumexp(torch.minimum(logp, logq), dim=-1).exp() * valid
        entropy = -(logp.exp() * logp).sum(-1) * valid
    return loss, block_log, length, floored, overlap, entropy


def bv_loss_terms(draft_logits, target_logits, labels, valid, *, beta,
                  temperature=1.0, block_chunk_size=8):
    """Local block-pooled numerator/count plus detached diagnostics.

    Shapes: logits [..., depth, vocab], labels/mask [..., depth]. No depth decay
    or D-PACE multipliers are applied. Prefix holes truncate the remainder. Full
    target distributions are integrated, without top-k truncation. Chunk-local
    log-softmax/minimum activations are recomputed in backward, bounding *added*
    storage; existing model logits/LM-head allocation is unchanged.
    """
    validate_bv_options(1.0, temperature, 0.5, block_chunk_size)
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("BV beta must be in [0, 1]")
    if (draft_logits.shape != target_logits.shape or draft_logits.ndim < 3
            or labels.shape != draft_logits.shape[:-1] or valid.shape != labels.shape):
        raise ValueError("BV expects matched [..., depth, vocab] logits and [..., depth] labels/mask")
    depth, vocab = draft_logits.shape[-2:]
    if depth < 1 or vocab < 1 or labels.numel() == 0:
        raise ValueError("BV requires non-empty blocks and vocabulary")
    q = draft_logits.reshape(-1, depth, vocab)
    p = target_logits.detach().reshape_as(q)
    y = labels.reshape(-1, depth)
    keep = valid.bool().reshape(-1, depth).int().cumprod(-1).bool()
    outputs = []
    for start in range(0, q.shape[0], block_chunk_size):
        stop = start + block_chunk_size
        args = (q[start:stop], p[start:stop], y[start:stop], keep[start:stop], beta, temperature)
        if torch.is_grad_enabled() and q.requires_grad:
            outputs.append(checkpoint(_bv_chunk, *args, use_reentrant=False, preserve_rng_state=False))
        else:
            outputs.append(_bv_chunk(*args))
    loss, block_log, length, floor, overlap, entropy = (
        torch.cat([part[i] for part in outputs], dim=0) for i in range(6))
    return BVTerms(loss.sum(), keep.any(-1).sum().float(), length.sum(), block_log.sum(),
                   floor.sum(), keep.sum().float(), overlap.reshape_as(valid), entropy.reshape_as(valid))
