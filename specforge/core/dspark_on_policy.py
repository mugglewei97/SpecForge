"""Fresh DSpark speculative rounds and same-step full-vocabulary beta replay.

The seven DSpark slots all predict tokens. Only the root token enters the
backbone; sampled predecessors enter the Markov/CARH head. Rejected proposal
suffixes retain their original verification hidden states for the loss.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.dflash import create_dflash_block_mask, create_dflash_sdpa_mask
from specforge.core.tv_acceptance import TVAcceptanceTerms, reduce_tv_terms
from specforge.core.tv_acceptance_beta import (
    BETA_BLOCK_SIZE,
    _proposal_logits,
    sample_beta_proposals,
    tv_acceptance_beta_terms,
)
from specforge.core.tv_on_policy import (
    prepare_on_policy_prompts,
    verify_on_policy_block,
)
from specforge.core.tv_rollout_cache import BlockRolloutCache


class DSparkRolloutCache(BlockRolloutCache):
    def __init__(self):
        # Unlike native DFlash2, slot zero predicts the first proposal.
        super().__init__(proposal_start=0)


@dataclass
class DSparkBlockReplay:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_hidden: torch.Tensor
    anchors: torch.Tensor
    keep: torch.Tensor
    tokens: torch.Tensor
    previous: torch.Tensor
    valid: torch.Tensor
    behavior_log_probs: torch.Tensor
    teacher_hidden: torch.Tensor
    statistics: torch.Tensor


def forward_dspark_blocks(
    model, input_ids, target_hidden, attention_mask, anchors, keep
):
    """Replay isolated blocks against committed context strictly before each root."""
    batch, length = input_ids.shape
    noise = model._create_noise_embed(input_ids, anchors, keep)
    context_positions = torch.arange(length, device=input_ids.device).expand(batch, -1)
    positions = torch.cat((context_positions, model._create_position_ids(anchors)), -1)
    mask_factory = (
        create_dflash_block_mask
        if model.attention_backend == "flex_attention"
        else create_dflash_sdpa_mask
    )
    mask = mask_factory(
        anchor_positions=anchors,
        block_keep_mask=keep,
        S=length,
        block_size=model.block_size,
        device=input_ids.device,
    )
    # The session produces right-padded committed context. Each kept anchor is
    # inside that context, so the strict <anchor mask also excludes all padding.
    hidden = model.draft_model(
        position_ids=positions,
        noise_embedding=noise,
        target_hidden=target_hidden,
        attention_mask=mask,
    )
    return hidden.reshape(batch, anchors.size(1), model.block_size, -1)


def _globally_active(active):
    flag = active.any().to(torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


@torch.no_grad()
def collect_dspark_rollout(model, input_ids, attention_mask, loss_mask, session):
    prompts = prepare_on_policy_prompts(
        input_ids,
        attention_mask,
        loss_mask,
        max_seq_len=model.tv_max_sequence_length,
        max_new_tokens=model.tv_rollout_max_tokens,
    )
    ids = pad_sequence(prompts.prefixes, batch_first=True, padding_value=0)
    lengths = torch.tensor([x.numel() for x in prompts.prefixes], device=ids.device)
    mask = torch.arange(ids.size(1), device=ids.device)[None] < lengths[:, None]
    session.reserve_capacity(int((lengths + prompts.budgets).max().item()))
    output = session.capture(ids, mask.long(), mask.float())
    rows = torch.arange(ids.size(0), device=ids.device)
    initial_hidden = output.last_hidden_states[rows, lengths - 1].to(
        model.lm_head.weight
    )
    initial_probs = (
        model.lm_head(initial_hidden).float() / model.tv_temperature
    ).softmax(-1)
    seed = torch.multinomial(initial_probs, 1).squeeze(-1)
    output = session.seed(seed, prompts.active)
    remaining = prompts.budgets - prompts.active.long()
    eos = torch.zeros_like(prompts.active)
    for token in model.tv_eos_token_ids:
        eos |= seed == token
    active = prompts.active & ~eos & (remaining > 0)
    # accepted, proposed, rounds, generated (including seed), episodes, capped
    statistics = torch.zeros(6, device=ids.device, dtype=torch.float64)
    statistics[3] = prompts.active.sum()
    statistics[4] = prompts.active.sum()
    history = []
    cache = DSparkRolloutCache()
    for _ in range(model.num_anchors):
        # Keep nested FSDP forwards aligned even after a local rank finishes.
        if not _globally_active(active):
            break
        output.hidden_states = output.hidden_states.to(model.lm_head.weight)
        anchors, hidden = cache.forward(model, output, active)
        valid = (
            torch.arange(BETA_BLOCK_SIZE, device=ids.device)[None] < remaining[:, None]
        ) & active[:, None]
        rollout = sample_beta_proposals(
            model,
            hidden,
            output.input_ids.gather(1, anchors[:, None]).squeeze(-1),
            valid,
            retain_probs=True,
        )
        # These are final-normalized states on the actual proposed prefix,
        # including the rejected suffix; never replace them with committed states.
        teacher = session.verify(rollout.tokens, rollout.valid, active).to(hidden)
        accepted = torch.zeros_like(remaining)
        next_tokens = torch.zeros_like(remaining)
        emit_next = torch.zeros_like(active)
        active_next = torch.zeros_like(active)
        emitted = torch.zeros_like(remaining)
        for start in range(0, ids.size(0), model.tv_objective_chunk_blocks):
            end = min(start + model.tv_objective_chunk_blocks, ids.size(0))
            p = (
                model.lm_head(teacher[start:end]).float() / model.tv_temperature
            ).softmax(-1)
            step = verify_on_policy_block(
                rollout.tokens[start:end],
                rollout.valid[start:end],
                rollout.probabilities[start:end],
                p,
                torch.where(active[start:end], remaining[start:end], 0),
                eos_token_ids=model.tv_eos_token_ids,
            )
            accepted[start:end] = step.accepted_counts
            next_tokens[start:end] = step.next_tokens
            emit_next[start:end] = step.emit_next
            active_next[start:end] = step.active_next
            emitted[start:end] = step.emitted_counts
        history.append(
            (
                anchors,
                active,
                rollout.tokens,
                rollout.previous,
                rollout.valid,
                rollout.log_probs,
                teacher[:, :BETA_BLOCK_SIZE].clone(),
            )
        )
        statistics[0] += accepted.sum()
        statistics[1] += rollout.valid.sum()
        statistics[2] += active.sum()
        statistics[3] += emitted.sum()
        output = session.commit(accepted, next_tokens, active, emit_next)
        remaining = remaining - emitted
        active = active_next
        # Full-vocabulary sampling probabilities are needed only for this round.
        del rollout, teacher, p
    statistics[5] = active.sum()
    if not history:
        # FSDP still needs a connected zero graph when every seed ends an episode.
        shape = (ids.size(0), BETA_BLOCK_SIZE)
        zeros = torch.zeros(shape, device=ids.device, dtype=torch.long)
        history.append(
            (
                output.attention_mask.sum(-1).long() - 1,
                torch.zeros(ids.size(0), device=ids.device, dtype=torch.bool),
                zeros,
                zeros.clone(),
                zeros.bool(),
                zeros.float(),
                initial_hidden.new_zeros((*shape, initial_hidden.size(-1))),
            )
        )
    fields = [torch.stack(parts, dim=1) for parts in zip(*history)]
    return DSparkBlockReplay(
        output.input_ids,
        output.attention_mask,
        output.hidden_states.to(model.lm_head.weight),
        *fields,
        statistics,
    )


def replay_dspark_blocks(model, replay):
    hidden = forward_dspark_blocks(
        model,
        replay.input_ids,
        replay.target_hidden,
        replay.attention_mask,
        replay.anchors,
        replay.keep,
    ).flatten(0, 1)

    def chunk_terms(h, previous, teacher, valid, tokens, old):
        logits = _proposal_logits(model, h, previous)
        with torch.no_grad():
            target_logits = model.lm_head(teacher)
        return tuple(
            tv_acceptance_beta_terms(
                logits, target_logits, valid, tokens, old, model.tv_temperature
            )
        )

    terms = TVAcceptanceTerms(
        *checkpointed_chunk_reduce(
            chunk_terms,
            hidden,
            replay.previous.flatten(0, 1),
            replay.teacher_hidden.flatten(0, 1),
            replay.valid.flatten(0, 1),
            replay.tokens.flatten(0, 1),
            replay.behavior_log_probs.flatten(0, 1),
            chunk_size=model.tv_objective_chunk_blocks,
        )
    )
    result = reduce_tv_terms(terms, allow_empty=True)
    metrics = result[-1]
    metrics["tv_acceptance_length_estimate"] = metrics.pop("tv_acceptance_length_proxy")
    stats = replay.statistics.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats)
    accepted, proposed, rounds, generated, episodes, capped = stats.unbind()
    metrics.update(
        tv_rollout_accepted_length=accepted / rounds.clamp_min(1),
        tv_rollout_proposed_length=proposed / rounds.clamp_min(1),
        tv_rollout_generated_tokens=generated,
        tv_rollout_rounds=rounds,
        tv_rollout_episodes=episodes,
        tv_rollout_round_limit_count=capped,
        tv_rollout_empty_batch=(rounds == 0).float(),
    )
    return result


def on_policy_dspark_forward(model, input_ids, attention_mask, loss_mask, session):
    replay = collect_dspark_rollout(
        model, input_ids, attention_mask, loss_mask, session
    )
    # Replay owns detached context and verification states, independent of KV.
    session.close()
    return replay_dspark_blocks(model, replay)
