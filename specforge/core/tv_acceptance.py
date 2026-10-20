"""TV acceptance-length proxy on ground-truth training-sequence prefixes.

Reuse the target's single sequence forward for every anchor. The prefix-product
objective is a teacher-forced proxy, not measured speculative acceptance length.
"""

import math
from typing import NamedTuple

import torch
import torch.distributed as dist

from specforge.core.chunking import checkpointed_chunk_reduce


class TVAcceptanceTerms(NamedTuple):
    loss_sum: torch.Tensor
    block_count: torch.Tensor
    length_sum: torch.Tensor
    overlap_position_sum: torch.Tensor
    correct_position_sum: torch.Tensor
    position_count: torch.Tensor


def sample_tv_anchors(loss_mask, attention_mask, num_anchors):
    """Keep partial tails and a masked placeholder on locally empty ranks.

    A rank must still enter the collective loss reduction when other ranks
    have valid blocks. The legacy sampler raises early in that case.
    """
    batch, length = loss_mask.shape
    if length < 1:
        raise ValueError("TV anchor sampling requires a nonempty input sequence")
    eligible = (
        (loss_mask[:, :-1] > 0.5)
        & (loss_mask[:, 1:] > 0.5)
        & attention_mask[:, :-1].bool()
        & attention_mask[:, 1:].bool()
    )
    counts = eligible.sum(-1)
    width = max(1, min(num_anchors, int(counts.max().item())))
    if eligible.shape[1] == 0:
        anchors = torch.zeros((batch, 1), device=loss_mask.device, dtype=torch.long)
        return anchors, torch.zeros_like(anchors, dtype=torch.bool)
    scores = torch.rand(eligible.shape, device=loss_mask.device)
    scores = torch.where(eligible, scores, 2.0)
    selected = scores.argsort(-1)[:, :width]
    keep = eligible.gather(1, selected)
    selected = torch.where(keep, selected, length)
    anchors = selected.sort(-1).values
    keep = anchors < length
    return torch.where(keep, anchors, 0), keep


def tv_acceptance_terms(draft_logits, target_logits, valid_mask, temperature=1.0):
    """Additive block terms for [..., K, V] logits on identical prefixes.

    Full-vocabulary softmax uses a shared positive temperature for both models.
    An invalid position terminates its block; EOS is included by
    the caller, and subsequent positions are invalid. Empty blocks have zero
    weight. Only target logits are detached; all prefix products retain grad.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("TV acceptance temperature must be finite and positive")
    if draft_logits.ndim < 3 or draft_logits.shape != target_logits.shape:
        raise ValueError("draft/target logits must have matching [..., K, V] shapes")
    if valid_mask.shape != draft_logits.shape[:-1]:
        raise ValueError("valid_mask must match the non-vocabulary logit dimensions")
    valid = valid_mask.bool().int().cumprod(dim=-1).bool()
    dtype = torch.float64 if draft_logits.dtype == torch.float64 else torch.float32
    # Padding may contain arbitrary values, including NaNs. Mask before softmax
    # to keep padding out of both the value and backward computation.
    q_logits = torch.where(valid[..., None], draft_logits.to(dtype), 0.0)
    p_logits = torch.where(valid[..., None], target_logits.detach().to(dtype), 0.0)
    q = torch.softmax(q_logits / temperature, dim=-1)
    p = torch.softmax(p_logits / temperature, dim=-1)
    a = (1 - 0.5 * (p - q).abs().sum(dim=-1)).clamp(0, 1)
    a = torch.where(valid, a, 0.0)
    length = a.cumprod(dim=-1).sum(dim=-1)
    k = valid.sum(dim=-1)
    nonempty = k > 0
    block_loss = torch.where(nonempty, 1 - length / k.clamp_min(1), 0.0)
    reduce_dims = tuple(range(valid.ndim - 1))
    correct = (q_logits.argmax(-1) == p_logits.argmax(-1)) & valid
    return TVAcceptanceTerms(
        block_loss.sum(),
        nonempty.to(dtype).sum(),
        length.sum(),
        a.sum(dim=reduce_dims),
        correct.to(dtype).sum(dim=reduce_dims),
        valid.to(dtype).sum(dim=reduce_dims),
    )


def teacher_forced_tv_forward(
    model, input_ids, attention_mask, anchors, hidden, valid_mask, last_hidden_states
):
    """Score all blocks using target states and predecessors from the same sequence.

    Slot j at anchor s predicts input_ids[s+j+1]. Its Markov predecessor is
    input_ids[s+j], and its target distribution comes from final hidden row s+j.
    No token sampling or target transformer forward occurs here or in backward.
    """
    batch, blocks, width, hidden_size = hidden.shape
    if last_hidden_states is None:
        raise ValueError("tv-acceptance requires target final hidden states")
    if (
        input_ids.ndim != 2
        or input_ids.size(0) != batch
        or input_ids.size(1) < 1
        or attention_mask.shape != input_ids.shape
        or anchors.shape != (batch, blocks)
        or valid_mask.shape != hidden.shape[:-1]
        or last_hidden_states.shape != (*input_ids.shape, hidden_size)
    ):
        raise ValueError(
            "teacher-forced TV inputs and target hidden states are misaligned"
        )
    flat_hidden = hidden.reshape(-1, width, hidden_size)
    with torch.no_grad():
        positions = anchors[..., None] + torch.arange(width, device=input_ids.device)
        label_positions = positions + 1
        length = input_ids.size(1)
        in_bounds = (positions >= 0) & (label_positions < length)
        safe_positions = positions.clamp(0, length - 1)
        safe_labels = label_positions.clamp(0, length - 1)
        rows = torch.arange(batch, device=input_ids.device)[:, None, None]
        previous = input_ids[rows, safe_positions]
        labels = input_ids[rows, safe_labels]
        valid = (
            valid_mask.bool()
            & in_bounds
            & attention_mask[rows, safe_positions].bool()
            & attention_mask[rows, safe_labels].bool()
        )
        eos = torch.zeros_like(valid)
        for token_id in model.tv_eos_token_ids:
            eos |= labels == token_id
        # Include the ground-truth EOS, but never score the following turn.
        valid &= (eos.int().cumsum(-1) - eos.int()) == 0
        valid = valid.int().cumprod(-1).bool().reshape(-1, width)
        teacher = last_hidden_states.detach()[rows, safe_positions].to(
            device=hidden.device, dtype=hidden.dtype
        )

    def chunk_terms(h, prev, target_h, mask):
        logits = model.lm_head(h)
        if model.draft_model.markov_head is not None:
            logits = model.draft_model.markov_head.apply_block_logits(
                logits,
                token_ids=prev,
                hidden_states=h,
            )
        with torch.no_grad():
            target_logits = model.lm_head(target_h)
        return tuple(
            tv_acceptance_terms(
                logits,
                target_logits,
                mask,
                model.tv_temperature,
            )
        )

    terms = TVAcceptanceTerms(
        *checkpointed_chunk_reduce(
            chunk_terms,
            flat_hidden,
            previous.reshape(-1, width),
            teacher.reshape_as(flat_hidden),
            valid,
            chunk_size=model.tv_objective_chunk_blocks,
        )
    )
    return reduce_tv_terms(terms)


def reduce_tv_terms(terms: TVAcceptanceTerms, *, allow_empty=False):
    """Normalize across ranks and return DSpark's loss/telemetry contract."""
    scalars = torch.stack(
        [terms.loss_sum.detach(), terms.length_sum.detach(), terms.block_count.detach()]
    )
    positions = torch.stack(
        [
            terms.overlap_position_sum.detach(),
            terms.correct_position_sum.detach(),
            terms.position_count.detach(),
        ]
    )
    world = (
        dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    )
    if world > 1:
        dist.all_reduce(scalars)
        dist.all_reduce(positions)
    global_loss, global_length, global_count = scalars.unbind(0)
    if global_count.item() <= 0 and not allow_empty:
        raise ValueError("tv-acceptance has no valid blocks on any rank")
    # FSDP averages rank gradients: compensate to weight every nonempty block
    # equally. The same reduced count also normalizes the detached telemetry.
    block_denominator = global_count.clamp_min(1)
    loss = world * terms.loss_sum / block_denominator
    overlap, correct, count = positions.unbind(0)
    position_denominator = count.clamp_min(1)
    total_positions = count.sum().clamp_min(1)
    components = {
        "tv_acceptance_loss": global_loss / block_denominator,
        "tv_acceptance_length_proxy": global_length / block_denominator,
        "tv_mean_overlap": overlap.sum() / total_positions,
        "tv_valid_blocks": global_count,
        "tv_mean_valid_length": count.sum() / block_denominator,
    }
    return (
        loss,
        correct.sum() / total_positions,
        (count - overlap) / position_denominator,
        correct / position_denominator,
        count,
        components,
    )


def configure_tv_acceptance(
    model,
    *,
    temperature,
    chunk_blocks,
    eos_token_ids,
    beta=False,
    verification_batch_size=None,
    anchor_mode="random",
    rollout_max_tokens=256,
    max_sequence_length=4096,
):
    """Enable the pure objective before FSDP/optimizer creation.

    Beta also supports single-predecessor CARH without predecessor context or
    sampled-prefix memory: its step sampler and block replay use the same
    token, hidden state and depth. Stateful CARH variants, recurrent, selector
    and refiner heads need their own rollout/replay integration.
    """
    if not math.isfinite(temperature) or temperature <= 0 or chunk_blocks < 1:
        raise ValueError("temperature must be finite and > 0; chunk_blocks must be > 0")
    draft = model.draft_model
    if beta and (
        model.block_size != 7 or getattr(draft.config, "block_size", None) != 7
    ):
        raise ValueError("tv-acceptance-beta requires actual model block_size=7")
    if beta and verification_batch_size is not None and verification_batch_size < 1:
        raise ValueError("TV beta verification batch size must be positive")
    if anchor_mode not in ("random", "on-policy"):
        raise ValueError("Unknown TV beta anchor mode")
    if anchor_mode == "on-policy":
        if not beta:
            raise ValueError("On-policy anchors require tv-acceptance-beta")
        if model.attention_backend not in ("sdpa", "flex_attention"):
            raise ValueError("On-policy DSpark beta requires sdpa or flex_attention")
        if rollout_max_tokens < 1 or max_sequence_length < 2 or model.num_anchors < 1:
            raise ValueError(
                "On-policy rollout, sequence and anchor budgets must be positive"
            )
        method = getattr(draft.config, "dflash_config", None) or {}
        if (
            getattr(draft.config, "attention_dropout", 0.0) != 0.0
            or getattr(draft, "target_layer_fusion_source_dropout", 0.0) != 0.0
            or method.get("cross_depth_mixer_dropout", 0.0) != 0.0
            or any(
                getattr(layer.self_attn, "sliding_window", None) is not None
                for layer in draft.layers
            )
        ):
            raise ValueError(
                "On-policy DSpark beta requires zero attention/fusion/mixer dropout "
                "and full attention"
            )
    head = draft.markov_head
    head_type = getattr(head, "markov_head_type", "vanilla")
    if head is not None and beta and head_type == "carh":
        if (
            getattr(head, "predecessor_count", None) != 1
            or getattr(head, "predecessor_context_mode", None) != "none"
            or getattr(head, "sampled_prefix_memory_rank", None) != 0
            or getattr(head, "sampled_prefix_memory", None) is not None
        ):
            raise ValueError(
                "tv-acceptance-beta CARH requires carh_predecessor_count=1, "
                "carh_predecessor_context_mode='none' and "
                "carh_sampled_prefix_memory_rank=0"
            )
    elif head is not None and head_type != "vanilla":
        raise ValueError(
            "tv-acceptance supports only the vanilla Markov head; "
            "tv-acceptance-beta also supports stateless single-predecessor CARH"
        )
    for name in (
        "candidate_selector",
        "parallel_refiner",
        "recall_correction",
        "prefix_state_mixer",
        "block_summary",
    ):
        if getattr(draft, name, None) is not None:
            raise ValueError(f"tv-acceptance does not support {name}")
    confidence = getattr(draft, "confidence_head", None)
    if confidence is not None:
        confidence.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    model.tv_temperature = float(temperature)
    model.tv_objective_chunk_blocks = int(chunk_blocks)
    model.tv_eos_token_ids = tuple(eos_token_ids)
    model.tv_acceptance_enabled = True
    model.tv_acceptance_beta = bool(beta)
    model.tv_verification_batch_size = verification_batch_size
    model.tv_anchor_mode = anchor_mode
    model.tv_rollout_max_tokens = int(rollout_max_tokens)
    model.tv_max_sequence_length = int(max_sequence_length)
