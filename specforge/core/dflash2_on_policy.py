"""Fresh DFlash2 speculative rollouts and exact conditional beta replay.

The target and draft use one shared positive temperature. Rollout records are
consumed once, before any parameter update: no stale top-k support is replayed.
Acceptance uniforms determine the next real anchor, never the beta loss mask.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers.cache_utils import DynamicCache

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.dflash2_tv_acceptance import (
    dflash2_candidate_scores,
    sample_dflash2_beta_proposals,
)
from specforge.core.tv_acceptance import TVAcceptanceTerms, reduce_tv_terms
from specforge.core.tv_acceptance_beta import BETA_BLOCK_SIZE
from specforge.core.tv_on_policy import (
    prepare_on_policy_prompts,
    verify_on_policy_block,
)
from specforge.modeling.draft.flex_attention_backend import flex_attention_backend


def candidate_beta_terms(
    scores,
    target_log_probs,
    valid,
    candidate_ids,
    tokens,
    behavior_log_probs,
    target_argmax,
    temperature,
):
    """Exact full-vocabulary beta for a proposal with finite candidate support.

    Target probabilities retain the full-vocabulary normalizer from verification.
    Outside the proposal support min(p,q)=0, so no dense vocabulary scatter or
    repeated target projection is necessary. All prefix products retain gradient.
    """
    valid = valid.bool().int().cumprod(-1).bool()
    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    logits = torch.where(valid[..., None], scores.to(dtype), 0.0)
    log_q = torch.log_softmax(logits / temperature, -1)
    log_p = torch.where(valid[..., None], target_log_probs.detach().to(dtype), 0.0)
    overlap = torch.minimum(log_p.exp(), log_q.exp()).sum(-1).clamp(0, 1)
    overlap = torch.where(valid, overlap, 0.0)
    match = candidate_ids == tokens[..., None]
    index = match.long().argmax(-1, keepdim=True)
    mass = torch.minimum(log_p, log_q).gather(-1, index).squeeze(-1)
    old = torch.where(valid, behavior_log_probs.detach().to(dtype), 0.0)
    ratios = torch.where(valid, (mass - old).exp(), 0.0)
    prefix = torch.cat((torch.ones_like(ratios[..., :1]), ratios[..., :-1]), -1)
    length = (prefix.cumprod(-1) * overlap).sum(-1)
    nonempty = valid.any(-1)
    loss = torch.where(nonempty, 1 - length / BETA_BLOCK_SIZE, 0.0)
    predicted = candidate_ids.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
    correct = (predicted == target_argmax) & valid
    dims = tuple(range(valid.ndim - 1))
    return TVAcceptanceTerms(
        loss.sum(),
        nonempty.to(dtype).sum(),
        length.sum(),
        overlap.sum(dims),
        correct.to(dtype).sum(dims),
        valid.to(dtype).sum(dims),
    )


class DFlashRolloutCache:
    """Reuse frozen-within-rollout draft context KV; discard block noise KV.

    Different rows can advance by different counts. Physical cache padding is
    masked explicitly, and RoPE uses each row's original token positions.
    This cache never crosses an optimizer update or enters backward replay.
    """

    def __init__(self):
        self.kv = DynamicCache()
        self.lengths = None
        self.valid_keys = None

    @torch.no_grad()
    def forward(self, model, output, active):
        ids, features = output.input_ids, output.hidden_states
        anchors = output.attention_mask.long().sum(-1) - 1
        batch = ids.size(0)
        if self.lengths is None:
            self.lengths = torch.zeros_like(anchors)
            self.valid_keys = torch.zeros(
                (batch, 0), device=ids.device, dtype=torch.bool
            )
        delta = anchors - self.lengths
        if (delta < 0).any():
            raise ValueError("Committed draft context cannot move backwards")
        width = int(delta.max().item())
        offsets = torch.arange(width, device=ids.device)[None]
        positions = self.lengths[:, None] + offsets
        new_valid = offsets < delta[:, None]
        rows = torch.arange(batch, device=ids.device)[:, None]
        context = features[rows, positions.clamp(max=ids.size(1) - 1)]
        context = torch.where(new_valid[..., None], context, 0.0)
        context_keys = torch.cat((self.valid_keys, new_valid), -1)
        keys = (
            torch.cat((context_keys, active[:, None].expand(-1, model.block_size)), -1)
            & active[:, None]
        )
        noise = model._create_noise_embed(ids, anchors[:, None], active[:, None])
        noise_positions = anchors[:, None] + torch.arange(
            model.block_size, device=ids.device
        )
        position_ids = torch.cat((positions, noise_positions), -1)
        kwargs = {}
        if model.attention_backend == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            key_width = keys.size(1)

            def mask_mod(b, h, q, k):
                return (
                    (q < model.block_size)
                    & (k < key_width)
                    & keys[b, k.clamp(max=key_width - 1)]
                )

            mask = create_block_mask(
                mask_mod,
                B=batch,
                H=None,
                Q_LEN=model.block_size,
                KV_LEN=key_width,
                device=ids.device,
                BLOCK_SIZE=(256, 128) if flex_attention_backend() == "FLASH" else 128,
            )
            kwargs["kernel_options"] = (
                {"BACKEND": "TRITON"}
                if torch.__version__ >= "2.11"
                else {"FORCE_USE_FLEX_ATTENTION": True}
            )
        else:
            mask = keys[:, None, None, :].expand(-1, 1, model.block_size, -1)
        hidden = model.draft_model(
            position_ids=position_ids,
            attention_mask=mask,
            noise_embedding=noise,
            target_hidden=context,
            past_key_values=self.kv,
            use_cache=True,
            **kwargs,
        )
        self.kv.crop(context_keys.size(1))
        self.lengths = anchors
        self.valid_keys = context_keys
        return anchors, hidden[:, 1:]


@dataclass
class DFlash2BlockReplay:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_hidden: torch.Tensor
    anchors: torch.Tensor
    keep: torch.Tensor
    tokens: torch.Tensor
    previous: torch.Tensor
    valid: torch.Tensor
    behavior_log_probs: torch.Tensor
    candidate_ids: torch.Tensor
    target_log_probs: torch.Tensor
    target_argmax: torch.Tensor
    statistics: torch.Tensor


def _globally_active(active):
    flag = active.any().to(torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


@torch.no_grad()
def collect_dflash2_rollout(model, input_ids, attention_mask, loss_mask, session):
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
    # Native DFlash first obtains one target token, then uses it as the anchor.
    rows = torch.arange(ids.size(0), device=ids.device)
    initial_hidden = output.last_hidden_states[rows, lengths - 1].to(
        model.lm_head.weight
    )
    initial_probs = (
        model.lm_head(initial_hidden).float() / model.tv_temperature
    ).softmax(-1)
    seed = torch.multinomial(initial_probs, 1).squeeze(-1)
    active = prompts.active
    output = session.seed(seed, active)
    remaining = prompts.budgets - active.long()
    eos = torch.zeros_like(active)
    for token in model.tv_eos_token_ids:
        eos |= seed == token
    active = active & ~eos & (remaining > 0)
    # accepted, proposals, rounds, generated (incl. seed), episodes, capped rows
    statistics = torch.zeros(6, device=ids.device, dtype=torch.float64)
    statistics[3] = prompts.active.sum()
    statistics[4] = prompts.active.sum()
    history = []
    cache = DFlashRolloutCache()
    for _ in range(model.num_anchors):
        # All ranks execute the same number of nested FSDP forwards, including
        # ranks whose local prompts have already reached EOS or their budget.
        if not _globally_active(active):
            break
        output.hidden_states = output.hidden_states.to(model.lm_head.weight)
        anchors, hidden = cache.forward(model, output, active)
        valid = (
            torch.arange(BETA_BLOCK_SIZE, device=ids.device)[None] < remaining[:, None]
        ) & active[:, None]
        rollout = sample_dflash2_beta_proposals(
            model,
            hidden,
            output.input_ids.gather(1, anchors[:, None]).squeeze(-1),
            valid,
            retain_candidate_probs=True,
        )
        teacher = session.verify(rollout.tokens, rollout.valid, active).to(hidden)
        target_candidates = torch.zeros_like(rollout.candidate_probs)
        target_argmax = torch.zeros_like(rollout.tokens)
        accepted = torch.zeros_like(remaining)
        next_tokens = torch.zeros_like(remaining)
        emit_next = torch.zeros_like(active)
        active_next = torch.zeros_like(active)
        emitted = torch.zeros_like(remaining)
        for start in range(0, ids.size(0), model.tv_objective_chunk_blocks):
            end = min(start + model.tv_objective_chunk_blocks, ids.size(0))
            logits = model.lm_head(teacher[start:end]).float() / model.tv_temperature
            log_p = logits.log_softmax(-1)
            target_candidates[start:end] = log_p[:, :BETA_BLOCK_SIZE].gather(
                -1, rollout.candidate_ids[start:end]
            )
            target_argmax[start:end] = logits[:, :BETA_BLOCK_SIZE].argmax(-1)
            q = torch.zeros_like(log_p[:, :BETA_BLOCK_SIZE]).scatter(
                -1, rollout.candidate_ids[start:end], rollout.candidate_probs[start:end]
            )
            step = verify_on_policy_block(
                rollout.tokens[start:end],
                rollout.valid[start:end],
                q,
                log_p.exp(),
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
                rollout.candidate_ids,
                target_candidates,
                target_argmax,
            )
        )
        statistics[0] += accepted.sum()
        statistics[1] += rollout.valid.sum()
        statistics[2] += active.sum()
        statistics[3] += emitted.sum()
        output = session.commit(accepted, next_tokens, active, emit_next)
        remaining = remaining - emitted
        active = active_next
    statistics[5] = active.sum()
    if not history:
        # A zero graph is still needed for FSDP if every initial token was EOS.
        batch = ids.size(0)
        shape = (batch, BETA_BLOCK_SIZE)
        candidates = model.draft_model.candidate_selector.top_k
        zeros = torch.zeros(shape, device=ids.device, dtype=torch.long)
        history.append(
            (
                output.attention_mask.sum(-1).long() - 1,
                torch.zeros(batch, device=ids.device, dtype=torch.bool),
                zeros,
                zeros.clone(),
                zeros.bool(),
                zeros.float(),
                torch.zeros((*shape, candidates), device=ids.device, dtype=torch.long),
                torch.zeros((*shape, candidates), device=ids.device),
                zeros.clone(),
            )
        )
    fields = [torch.stack(parts, dim=1) for parts in zip(*history)]
    return DFlash2BlockReplay(
        output.input_ids,
        output.attention_mask,
        output.hidden_states.to(model.lm_head.weight),
        *fields,
        statistics,
    )


def replay_dflash2_blocks(model, replay):
    _, _, output = model._forward_draft_blocks(
        replay.input_ids,
        replay.target_hidden,
        replay.attention_mask,
        anchor_positions=replay.anchors,
        block_keep_mask=replay.keep,
    )
    batch, blocks = replay.anchors.shape
    hidden = output.reshape(batch, blocks, model.block_size, -1)[:, :, 1:]
    hidden = hidden.reshape(-1, BETA_BLOCK_SIZE, hidden.size(-1))

    def flat(value):
        return value.flatten(0, 1)

    def chunk_terms(h, previous, candidate_ids, target_p, valid, tokens, old, argmax):
        scores = dflash2_candidate_scores(model, h, previous, candidate_ids)
        return tuple(
            candidate_beta_terms(
                scores,
                target_p,
                valid,
                candidate_ids,
                tokens,
                old,
                argmax,
                model.tv_temperature,
            )
        )

    terms = TVAcceptanceTerms(
        *checkpointed_chunk_reduce(
            chunk_terms,
            hidden,
            flat(replay.previous),
            flat(replay.candidate_ids),
            flat(replay.target_log_probs),
            flat(replay.valid),
            flat(replay.tokens),
            flat(replay.behavior_log_probs),
            flat(replay.target_argmax),
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
        {
            "tv_rollout_accepted_length": accepted / rounds.clamp_min(1),
            "tv_rollout_proposed_length": proposed / rounds.clamp_min(1),
            "tv_rollout_generated_tokens": generated,
            "tv_rollout_rounds": rounds,
            "tv_rollout_episodes": episodes,
            "tv_rollout_round_limit_count": capped,
            "tv_rollout_empty_batch": (rounds == 0).float(),
        }
    )
    return result


def on_policy_dflash2_forward(model, input_ids, attention_mask, loss_mask, session):
    replay = collect_dflash2_rollout(
        model, input_ids, attention_mask, loss_mask, session
    )
    # Neither replay nor backward depends on target KV. Release it before the
    # differentiable draft forward to reduce peak GPU memory.
    session.close()
    return replay_dflash2_blocks(model, replay)
