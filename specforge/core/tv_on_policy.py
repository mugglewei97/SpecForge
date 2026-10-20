"""Prompt extraction and exact stochastic verification for on-policy TV training.

The verifier consumes the distributions actually used for sampling, after any
temperature or logits processing. It does not alter the complete proposal block:
rejected suffixes remain available to the acceptance-length training objective.
"""

from typing import NamedTuple, Optional, Sequence, Tuple

import torch


class OnPolicyPrompts(NamedTuple):
    prefixes: Tuple[torch.Tensor, ...]
    budgets: torch.Tensor
    active: torch.Tensor


class OnPolicyStep(NamedTuple):
    accepted_counts: torch.Tensor
    next_tokens: torch.Tensor
    emit_next: torch.Tensor
    active_next: torch.Tensor
    emitted_counts: torch.Tensor


def prepare_on_policy_prompts(
    input_ids,
    attention_mask,
    loss_mask,
    *,
    max_seq_len,
    max_new_tokens,
):
    """Discard reference continuations, retaining only each row's first prompt.

    Only right padding is supported. The first supervised token starts generation;
    all reference answers and subsequent conversation turns are discarded, even if
    the loss mask contains multiple supervised spans. If supervision starts at the
    first token, that token is retained as the mandatory anchor. Inactive rows keep
    one token (zero for an entirely padded row) so distributed callers can still run
    the same model operations, but their generation budget is zero.
    """
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("on-policy input_ids must have nonempty [B, S] shape")
    if attention_mask.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
        raise ValueError("on-policy attention and loss masks must align with input_ids")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("on-policy input_ids must contain integer token IDs")
    if (
        attention_mask.device != input_ids.device
        or loss_mask.device != input_ids.device
    ):
        raise ValueError("on-policy inputs and masks must share a device")
    if not isinstance(max_seq_len, int) or max_seq_len < 1:
        raise ValueError("on-policy max_seq_len must be a positive integer")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("on-policy max_new_tokens must be a nonnegative integer")
    if not torch.isfinite(attention_mask).all() or not torch.isfinite(loss_mask).all():
        raise ValueError("on-policy masks must be finite")
    attention = attention_mask.bool()
    if (attention[:, 1:] & ~attention[:, :-1]).any():
        raise ValueError(
            "on-policy attention_mask must use right padding without holes"
        )
    supervised = (loss_mask > 0.5) & attention
    has_supervision = supervised.any(-1)
    first = supervised.to(torch.int64).argmax(-1)
    lengths = torch.where(has_supervision, first.clamp(min=1), 1)
    lengths = lengths.clamp(max=max_seq_len)
    budgets = torch.minimum(
        torch.full_like(lengths, max_new_tokens), max_seq_len - lengths
    )
    budgets = torch.where(has_supervision, budgets, 0)
    prefixes = []
    # One metadata transfer; never inspect or copy tokens after the prompt.
    metadata = torch.stack((lengths, attention.any(-1).long()), -1).cpu().tolist()
    for row, (length, has_tokens) in enumerate(metadata):
        prefix = (
            input_ids[row, :length].clone() if has_tokens else input_ids.new_zeros(1)
        )
        prefixes.append(prefix)
    return OnPolicyPrompts(tuple(prefixes), budgets, budgets > 0)


def _check_probabilities(probabilities, required, name):
    if not required.any():
        return
    invalid = ~torch.isfinite(probabilities).all(-1) | (probabilities < 0).any(-1)
    if (invalid & required).any():
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    # Reduce first, avoiding an additional [B, K, V] floating-point copy.
    totals = probabilities.sum(-1)[required]
    tolerance = 2e-3 if probabilities.dtype in (torch.float16, torch.bfloat16) else 1e-5
    if not torch.allclose(
        totals, torch.ones_like(totals), rtol=tolerance, atol=tolerance
    ):
        raise ValueError(f"{name} must be normalized over the full vocabulary")


def _uniforms(values, shape, *, device, dtype, generator, name):
    if values is None:
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)
    if values.shape != shape or values.device != device:
        raise ValueError(f"{name} must have shape {shape} on the proposal device")
    if not torch.isfinite(values).all() or ((values < 0) | (values >= 1)).any():
        raise ValueError(f"{name} must lie in [0, 1)")
    return values.to(dtype)


@torch.no_grad()
def verify_on_policy_block(
    proposal_tokens,
    proposal_valid,
    draft_probs,
    target_probs,
    remaining_tokens,
    *,
    eos_token_ids: Sequence[int] = (),
    accept_uniforms: Optional[torch.Tensor] = None,
    sample_uniforms: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """Verify a block with exact rejection sampling and return its state update.

    ``draft_probs`` is [B, K, V], ``target_probs`` is [B, K + 1, V]; row j
    of each distribution conditions on the same complete proposal prefix. Target
    row K supplies the all-accepted bonus. Sparse draft candidate distributions are
    represented by zeros outside their support, without renormalizing the target.

    Commit the first ``accepted_counts`` proposals and, when ``emit_next`` is true,
    ``next_tokens``. The latter is sampled from (p-q)+ at the first rejection, or p
    after complete acceptance. Accepted EOS emits no correction/bonus. Budget tails
    may consume only part of a block; no token is emitted past the budget. A zero
    budget requires an empty valid mask. ``next_tokens`` is zero when not emitted.

    Injected uniforms make exact finite-distribution tests possible. They are
    independent acceptance draws [B, K] and a categorical draw [B].
    """
    if proposal_tokens.ndim != 2 or proposal_tokens.size(1) < 1:
        raise ValueError("proposal_tokens must have shape [B, K] with K >= 1")
    batch, width = proposal_tokens.shape
    if proposal_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("proposal_tokens must contain integer token IDs")
    if proposal_valid.shape != (batch, width) or remaining_tokens.shape != (batch,):
        raise ValueError("proposal validity and remaining token budgets must align")
    if remaining_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("remaining_tokens must contain integer budgets")
    if draft_probs.ndim != 3 or draft_probs.shape[:2] != (batch, width):
        raise ValueError("draft_probs must have shape [B, K, V]")
    vocab = draft_probs.size(-1)
    if vocab < 1 or target_probs.shape != (batch, width + 1, vocab):
        raise ValueError("target_probs must have shape [B, K + 1, V]")
    if not draft_probs.is_floating_point() or not target_probs.is_floating_point():
        raise ValueError("draft and target probabilities must be floating point")
    device = proposal_tokens.device
    if any(
        value.device != device
        for value in (proposal_valid, draft_probs, target_probs, remaining_tokens)
    ):
        raise ValueError("on-policy verification tensors must share a device")
    if (remaining_tokens < 0).any():
        raise ValueError("remaining token budgets must be nonnegative")
    valid = proposal_valid.bool()
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("proposal_valid must be a contiguous prefix without holes")
    if (valid & (remaining_tokens == 0)[:, None]).any():
        raise ValueError("zero-budget rows must have no valid proposals")
    if (((proposal_tokens < 0) | (proposal_tokens >= vocab)) & valid).any():
        raise ValueError("valid proposal token IDs must be inside the vocabulary")

    dtype = (
        torch.float64
        if draft_probs.dtype == torch.float64 or target_probs.dtype == torch.float64
        else torch.float32
    )
    tokens = torch.where(valid, proposal_tokens, 0).long()
    eos = torch.zeros_like(valid)
    for token_id in eos_token_ids:
        eos |= tokens == token_id
    eos &= valid
    eos_before = torch.cat((torch.zeros_like(eos[:, :1]), eos[:, :-1]), -1)
    eos_before = eos_before.long().cumsum(-1) > 0
    positions = torch.arange(width, device=device)
    allowed = valid & (positions < remaining_tokens[:, None]) & ~eos_before
    lengths = allowed.sum(-1)
    ends_in_eos = eos.gather(-1, (lengths - 1).clamp(min=0)[:, None]).squeeze(-1)
    ends_in_eos &= lengths > 0
    target_positions = torch.arange(width + 1, device=device)
    needed_target = target_positions < lengths[:, None]
    needs_bonus = (remaining_tokens > lengths) & ~ends_in_eos
    needed_target |= (target_positions == lengths[:, None]) & needs_bonus[:, None]
    _check_probabilities(draft_probs, allowed, "draft_probs")
    _check_probabilities(target_probs, needed_target, "target_probs")

    q_tokens = draft_probs.gather(-1, tokens[..., None]).squeeze(-1).to(dtype)
    p_tokens = (
        target_probs[:, :width].gather(-1, tokens[..., None]).squeeze(-1).to(dtype)
    )
    if ((q_tokens <= 0) & allowed).any():
        raise ValueError("a sampled proposal must have positive draft probability")
    q_tokens = torch.where(allowed, q_tokens, 1.0)
    p_tokens = torch.where(allowed, p_tokens, 0.0)
    acceptance = (p_tokens / q_tokens).clamp(max=1)
    draws = _uniforms(
        accept_uniforms,
        (batch, width),
        device=device,
        dtype=dtype,
        generator=generator,
        name="accept_uniforms",
    )
    accepted = (allowed & (draws < acceptance)).long().cumprod(-1)
    accepted_counts = accepted.sum(-1)
    accepted_eos = (accepted.bool() & eos).any(-1)
    emit_next = (remaining_tokens > accepted_counts) & ~accepted_eos
    next_tokens = torch.zeros(batch, dtype=torch.long, device=device)
    rows = emit_next.nonzero(as_tuple=True)[0]
    if rows.numel():
        depth = accepted_counts[rows]
        p_next = target_probs[rows, depth].to(dtype)
        rejected = depth < lengths[rows]
        next_distribution = p_next.clone()
        if rejected.any():
            rejected_rows = rows[rejected]
            rejected_depth = depth[rejected]
            residual = (
                p_next[rejected] - draft_probs[rejected_rows, rejected_depth].to(dtype)
            ).clamp(min=0)
            # Never replace a small but positive residual by p: that changes
            # the speculative decoding distribution.
            if (residual.sum(-1) <= 0).any():
                raise ValueError("rejected proposals require positive residual mass")
            next_distribution[rejected] = residual
        if sample_uniforms is None:
            next_tokens[rows] = torch.multinomial(
                next_distribution, 1, generator=generator
            ).squeeze(-1)
        else:
            categorical = _uniforms(
                sample_uniforms,
                (batch,),
                device=device,
                dtype=dtype,
                generator=generator,
                name="sample_uniforms",
            )[rows]
            cdf = (
                next_distribution / next_distribution.sum(-1, keepdim=True)
            ).cumsum(-1)
            cdf[:, -1] = 1
            next_tokens[rows] = torch.searchsorted(
                cdf.contiguous(), categorical[:, None].contiguous(), right=True
            ).squeeze(-1)
    elif sample_uniforms is not None:
        _uniforms(
            sample_uniforms,
            (batch,),
            device=device,
            dtype=dtype,
            generator=generator,
            name="sample_uniforms",
        )
    emitted_counts = accepted_counts + emit_next.long()
    next_is_eos = torch.zeros(batch, dtype=torch.bool, device=device)
    for token_id in eos_token_ids:
        next_is_eos |= emit_next & (next_tokens == token_id)
    active_next = (emitted_counts < remaining_tokens) & ~accepted_eos & ~next_is_eos
    return OnPolicyStep(
        accepted_counts, next_tokens, emit_next, active_next, emitted_counts
    )
