"""On-policy single-round accepted length, conditional on fixed anchors.

The sampled-prefix factors include the proposal-distribution derivative. Target
verification and sampling happen once, outside activation checkpoint replay.
"""

import math
from typing import NamedTuple

import torch

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.tv_acceptance import TVAcceptanceTerms, reduce_tv_terms

BETA_BLOCK_SIZE = 7


class ProposalRollout(NamedTuple):
    tokens: torch.Tensor
    log_probs: torch.Tensor
    previous: torch.Tensor
    valid: torch.Tensor


def tv_acceptance_beta_terms(
    draft_logits,
    target_logits,
    valid_mask,
    proposal_tokens,
    old_log_probs,
    temperature=1.0,
):
    """R = a1 + r1*a2 + ...; r = min(p(d), q(d))/stopgrad(q_old(d)).

    The denominator is always seven, including truncated blocks. Short tensors
    are supported here for formula tests; the training model must have K=7.
    Do not clamp r to one or detach its prefix products: either changes the
    policy gradient, even when the on-policy forward value is unchanged.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("TV beta temperature must be finite and positive")
    if draft_logits.ndim < 3 or draft_logits.shape != target_logits.shape:
        raise ValueError("draft/target logits must have matching [..., K, V] shapes")
    shape = draft_logits.shape[:-1]
    if not 1 <= shape[-1] <= BETA_BLOCK_SIZE:
        raise ValueError("TV beta requires 1 <= K <= 7")
    if any(x.shape != shape for x in (valid_mask, proposal_tokens, old_log_probs)):
        raise ValueError("mask, proposal tokens and old log probabilities must align")
    valid = valid_mask.bool().int().cumprod(-1).bool()
    dtype = torch.float64 if draft_logits.dtype == torch.float64 else torch.float32
    q_logits = torch.where(valid[..., None], draft_logits.to(dtype), 0.0)
    p_logits = torch.where(valid[..., None], target_logits.detach().to(dtype), 0.0)
    log_q = torch.log_softmax(q_logits / temperature, -1)
    log_p = torch.log_softmax(p_logits / temperature, -1)
    overlap = torch.minimum(log_p.exp(), log_q.exp()).sum(-1).clamp(0, 1)
    overlap = torch.where(valid, overlap, 0.0)
    tokens = torch.where(valid, proposal_tokens, 0).long()[..., None]
    log_mass = torch.minimum(log_p.gather(-1, tokens), log_q.gather(-1, tokens))
    old = torch.where(valid, old_log_probs.detach().to(dtype), 0.0)
    ratios = torch.exp(log_mass.squeeze(-1) - old)
    ratios = torch.where(valid, ratios, 0.0)
    prefix = torch.cat((torch.ones_like(ratios[..., :1]), ratios[..., :-1]), -1)
    length = (prefix.cumprod(-1) * overlap).sum(-1)
    nonempty = valid.any(-1)
    loss = torch.where(nonempty, 1 - length / BETA_BLOCK_SIZE, 0.0)
    dims = tuple(range(valid.ndim - 1))
    correct = (q_logits.argmax(-1) == p_logits.argmax(-1)) & valid
    return TVAcceptanceTerms(
        loss.sum(),
        nonempty.to(dtype).sum(),
        length.sum(),
        overlap.sum(dims),
        correct.to(dtype).sum(dims),
        valid.to(dtype).sum(dims),
    )


def _proposal_logits(model, hidden, previous):
    logits = model.lm_head(hidden)
    if model.draft_model.markov_head is not None:
        logits = model.draft_model.markov_head.apply_block_logits(
            logits,
            token_ids=previous,
            hidden_states=hidden,
        )
    return logits


@torch.no_grad()
def sample_beta_proposals(model, hidden, anchor_tokens, valid):
    """Sample autoregressive Markov predecessors from the SAME backbone states."""
    count, width, _ = hidden.shape
    tokens = torch.zeros_like(valid, dtype=torch.long)
    previous = torch.zeros_like(tokens)
    dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    old_log_probs = torch.zeros(valid.shape, device=hidden.device, dtype=dtype)
    valid = valid.bool().int().cumprod(-1).bool().clone()
    chunk = model.tv_objective_chunk_blocks
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        h = hidden[start:end]
        # One base projection per block; only the small Markov head is sequential.
        base = model.lm_head(h)
        prev = anchor_tokens[start:end]
        head = model.draft_model.markov_head
        for depth in range(width):
            previous[start:end, depth] = prev
            logits = base[:, depth]
            if head is not None:
                logits = logits + head.compute_step_bias(prev, h[:, depth], depth)
            active = valid[start:end, depth]
            logits = torch.where(active[:, None], logits.to(dtype), 0.0)
            log_q = torch.log_softmax(logits / model.tv_temperature, -1)
            sampled = torch.multinomial(log_q.exp(), 1).squeeze(-1)
            tokens[start:end, depth] = sampled
            old_log_probs[start:end, depth] = log_q.gather(
                -1, sampled[:, None]
            ).squeeze(-1)
            if depth + 1 < width:
                eos = torch.zeros_like(active)
                for token_id in model.tv_eos_token_ids:
                    eos |= sampled == token_id
                valid[start:end, depth + 1 :] &= (~eos & active)[:, None]
            prev = sampled
    return ProposalRollout(tokens, old_log_probs, previous, valid)


@torch.no_grad()
def verify_beta_proposals(
    session,
    input_ids,
    anchors,
    rollout,
    last_hidden_states,
    batch_size,
):
    """Reuse hidden rows up to the first divergent token, then score its suffix.

    Hidden at anchor+j predicts d_j, so a mismatch at d_j still permits reuse
    at j; only j+1 onward needs a new target forward. No bonus-token scoring.
    """
    batch, blocks = anchors.shape
    width = rollout.tokens.shape[-1]
    positions = anchors[..., None] + torch.arange(width, device=anchors.device)
    rows = torch.arange(batch, device=anchors.device)[:, None, None]
    teacher = last_hidden_states.detach()[
        rows, positions.clamp(0, input_ids.size(1) - 1)
    ]
    teacher = teacher.reshape(-1, width, last_hidden_states.size(-1)).clone()
    # Transfer small token/length metadata once rather than synchronize per anchor.
    reference = input_ids.detach().cpu().tolist()
    anchor_list = anchors.detach().cpu().reshape(-1).tolist()
    proposals = rollout.tokens.detach().cpu().tolist()
    lengths = rollout.valid.sum(-1).cpu().tolist()
    requests = []
    for block, (anchor, tokens, length) in enumerate(
        zip(anchor_list, proposals, lengths)
    ):
        row = block // blocks
        first = 0
        while (
            first < length - 1 and tokens[first] == reference[row][anchor + first + 1]
        ):
            first += 1
        if first < length - 1:
            requests.append((block, row, anchor + 1 + first, first + 1, length))
    size = batch_size or session.default_batch_size
    for start in range(0, len(requests), size):
        group = requests[start : start + size]
        states = session.score_suffixes(
            [item[1] for item in group],
            [item[2] for item in group],
            [
                rollout.tokens[b, first - 1 : length - 1]
                for b, _, _, first, length in group
            ],
        )
        for (block, _, _, first, length), value in zip(group, states, strict=True):
            if value.shape != teacher[block, first:length].shape:
                raise ValueError(
                    "target verification returned misaligned final hidden states"
                )
            teacher[block, first:length] = value.to(teacher)
    return teacher


def sampled_tv_beta_forward(
    model,
    input_ids,
    attention_mask,
    anchors,
    hidden,
    valid_mask,
    last_hidden_states,
    verification_session,
):
    batch, blocks, width, hidden_size = hidden.shape
    if width != BETA_BLOCK_SIZE:
        raise ValueError("tv-acceptance-beta requires actual model block_size=7")
    if last_hidden_states is None or last_hidden_states.shape != (
        *input_ids.shape,
        hidden_size,
    ):
        raise ValueError(
            "tv-acceptance-beta requires aligned target final hidden states"
        )
    if verification_session is None:
        raise ValueError(
            "tv-acceptance-beta requires a target proposal verification session"
        )
    if anchors.shape != (batch, blocks) or valid_mask.shape != hidden.shape[:-1]:
        raise ValueError("TV beta anchors and valid mask are misaligned")
    with torch.no_grad():
        positions = anchors[..., None] + torch.arange(width, device=input_ids.device)
        rows = torch.arange(batch, device=input_ids.device)[:, None, None]
        labels = positions + 1
        valid = valid_mask.bool() & (positions >= 0) & (labels < input_ids.size(1))
        valid &= attention_mask[rows, positions.clamp(0, input_ids.size(1) - 1)].bool()
        valid &= attention_mask[rows, labels.clamp(0, input_ids.size(1) - 1)].bool()
        valid = valid.int().cumprod(-1).bool().reshape(-1, width)
        anchor_tokens = input_ids.gather(1, anchors).reshape(-1)
    flat_hidden = hidden.reshape(-1, width, hidden_size)
    rollout = sample_beta_proposals(model, flat_hidden, anchor_tokens, valid)
    teacher = verify_beta_proposals(
        verification_session,
        input_ids,
        anchors,
        rollout,
        last_hidden_states,
        model.tv_verification_batch_size,
    ).to(hidden)

    def chunk_terms(h, prev, target_h, mask, tokens, old):
        logits = _proposal_logits(model, h, prev)
        with torch.no_grad():
            target_logits = model.lm_head(target_h)
        return tuple(
            tv_acceptance_beta_terms(
                logits,
                target_logits,
                mask,
                tokens,
                old,
                model.tv_temperature,
            )
        )

    terms = TVAcceptanceTerms(
        *checkpointed_chunk_reduce(
            chunk_terms,
            flat_hidden,
            rollout.previous,
            teacher,
            rollout.valid,
            rollout.tokens,
            rollout.log_probs,
            chunk_size=model.tv_objective_chunk_blocks,
        )
    )
    result = reduce_tv_terms(terms)
    # Distinguish the sampled single-round estimate from the teacher-forced proxy.
    metrics = result[-1]
    metrics["tv_acceptance_length_estimate"] = metrics.pop("tv_acceptance_length_proxy")
    return result
