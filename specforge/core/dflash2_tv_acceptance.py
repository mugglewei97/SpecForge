"""DFlash2 sampled-prefix TV beta, isolated from the DSpark proposal policy.

The native block has one anchor and seven proposals. The selector defines a
conditional categorical distribution on the strict unary top-k; tokens outside
that set have zero proposal mass. Sampling and replay share the candidate IDs.
"""

import math
from typing import NamedTuple

import torch

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.tv_acceptance import (
    TVAcceptanceTerms,
    reduce_tv_terms,
    sample_tv_anchors,
)
from specforge.core.tv_acceptance_beta import (
    BETA_BLOCK_SIZE,
    tv_acceptance_beta_terms,
    verify_beta_proposals,
)


class DFlash2ProposalRollout(NamedTuple):
    tokens: torch.Tensor
    log_probs: torch.Tensor
    previous: torch.Tensor
    valid: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_probs: torch.Tensor | None = None


def dflash2_candidate_scores(model, hidden, previous, candidate_ids):
    """Current selector scores on the recorded support, without a dense scatter."""
    unary = model.draft_model.transform_unary_logits(model.lm_head(hidden))
    return model.draft_model.candidate_selector.score_candidates(
        candidate_ids=candidate_ids,
        unary_logits=unary.gather(-1, candidate_ids),
        hidden_states=hidden,
        predecessor_ids=previous,
    )


def dflash2_proposal_logits(model, hidden, previous, candidate_ids):
    """Replay the sampled support, with gradients through unary and selector.

    Scattering -inf outside the candidate set lets the existing full-vocabulary
    beta formula retain the target's full softmax normalization unchanged.
    Top-k membership is discrete; gradients are conditional on the sampled set.
    """
    unary = model.draft_model.transform_unary_logits(model.lm_head(hidden))
    scores = model.draft_model.candidate_selector.score_candidates(
        candidate_ids=candidate_ids,
        unary_logits=unary.gather(-1, candidate_ids),
        hidden_states=hidden,
        predecessor_ids=previous,
    )
    return torch.full_like(unary, -torch.inf).scatter(-1, candidate_ids, scores)


@torch.no_grad()
def sample_dflash2_beta_proposals(
    model, hidden, anchor_tokens, valid, *, retain_candidate_probs=False
):
    count, width, _ = hidden.shape
    selector = model.draft_model.candidate_selector
    tokens = torch.zeros_like(valid, dtype=torch.long)
    previous = torch.zeros_like(tokens)
    old_log_probs = torch.zeros(valid.shape, device=hidden.device, dtype=torch.float32)
    candidates = torch.empty(
        (*valid.shape, selector.top_k), device=hidden.device, dtype=torch.long
    )
    candidate_probs = (
        torch.zeros(candidates.shape, device=hidden.device, dtype=torch.float32)
        if retain_candidate_probs
        else None
    )
    valid = valid.bool().int().cumprod(-1).bool().clone()
    chunk = model.tv_objective_chunk_blocks
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        h = hidden[start:end]
        unary = model.draft_model.transform_unary_logits(model.lm_head(h))
        values, ids = unary.topk(selector.top_k, dim=-1)
        candidates[start:end] = ids
        prev = anchor_tokens[start:end]
        for depth in range(width):
            previous[start:end, depth] = prev
            scores = selector.score_candidates(
                candidate_ids=ids[:, depth],
                unary_logits=values[:, depth],
                hidden_states=h[:, depth],
                predecessor_ids=prev,
            )
            active = valid[start:end, depth]
            scores = torch.where(active[:, None], scores.float(), 0.0)
            log_q = torch.log_softmax(scores / model.tv_temperature, -1)
            if candidate_probs is not None:
                candidate_probs[start:end, depth] = log_q.exp()
            index = torch.multinomial(log_q.exp(), 1)
            sampled = ids[:, depth].gather(-1, index).squeeze(-1)
            tokens[start:end, depth] = sampled
            old_log_probs[start:end, depth] = log_q.gather(-1, index).squeeze(-1)
            if depth + 1 < width:
                eos = torch.zeros_like(active)
                for token_id in model.tv_eos_token_ids:
                    eos |= sampled == token_id
                valid[start:end, depth + 1 :] &= (~eos & active)[:, None]
            prev = sampled
    return DFlash2ProposalRollout(
        tokens, old_log_probs, previous, valid, candidates, candidate_probs
    )


class OnlineDFlash2TVBetaModel(OnlineDFlashModel):
    """Native DFlash2 backbone with the legacy beta trainer's return contract."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.block_size != BETA_BLOCK_SIZE + 1
            or self.draft_model.block_size != self.block_size
        ):
            raise ValueError(
                "DFlash2 tv-acceptance-beta requires block_size=8 (anchor + 7 proposals)"
            )
        if self.draft_model.candidate_selector is None:
            raise ValueError("DFlash2 tv-acceptance-beta requires a candidate selector")
        self.lm_head.requires_grad_(False)
        self.embed_tokens.requires_grad_(False)

    def configure_beta(
        self,
        *,
        temperature,
        chunk_blocks,
        eos_token_ids,
        verification_batch_size,
        anchor_mode="random",
        rollout_max_tokens=256,
        max_sequence_length=4096,
    ):
        if not math.isfinite(temperature) or temperature <= 0 or chunk_blocks < 1:
            raise ValueError(
                "temperature must be finite and > 0; chunk_blocks must be > 0"
            )
        if verification_batch_size is not None and verification_batch_size < 1:
            raise ValueError("TV beta verification batch size must be positive")
        self.tv_temperature = float(temperature)
        self.tv_objective_chunk_blocks = int(chunk_blocks)
        self.tv_eos_token_ids = tuple(eos_token_ids)
        self.tv_verification_batch_size = verification_batch_size
        if anchor_mode not in ("random", "on-policy"):
            raise ValueError("Unknown TV beta anchor mode")
        if rollout_max_tokens < 1 or max_sequence_length < 2:
            raise ValueError("TV rollout and sequence budgets must be positive")
        if anchor_mode == "on-policy" and (
            self.num_anchors < 1
            or getattr(self.draft_model.config, "attention_dropout", 0.0) != 0.0
            or self.draft_model.sliding_window is not None
        ):
            raise ValueError(
                "On-policy beta requires anchors > 0, zero draft dropout and full attention"
            )
        self.tv_anchor_mode = anchor_mode
        self.tv_rollout_max_tokens = int(rollout_max_tokens)
        self.tv_max_sequence_length = int(max_sequence_length)

    def set_training_progress(self, global_step, total_steps):
        # Pure beta has no auxiliary-loss schedule. The shared trainer calls
        # this lifecycle hook for both architectures.
        pass

    def _sample_anchor_positions(
        self, seq_len, loss_mask, device, max_valid_anchors=None
    ):
        # Preserve partial tails and collective participation on empty ranks.
        return sample_tv_anchors(
            loss_mask, torch.ones_like(loss_mask), self.num_anchors
        )

    def forward(
        self,
        input_ids,
        hidden_states,
        loss_mask,
        last_hidden_states=None,
        attention_mask=None,
        sample_ids=None,
        source_ids=None,
        tv_verification_session=None,
        tv_on_policy_session=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.tv_anchor_mode == "on-policy":
            from specforge.core.dflash2_on_policy import on_policy_dflash2_forward

            if tv_on_policy_session is None:
                raise ValueError("On-policy beta requires a target rollout session")
            return on_policy_dflash2_forward(
                self, input_ids, attention_mask, loss_mask, tv_on_policy_session
            )
        if (
            input_ids.ndim != 2
            or input_ids.size(1) < 1
            or loss_mask.shape != input_ids.shape
            or attention_mask.shape != input_ids.shape
            or last_hidden_states is None
            or last_hidden_states.shape
            != (*input_ids.shape, self.draft_model.config.hidden_size)
        ):
            raise ValueError(
                "DFlash2 TV beta requires aligned input masks and target final hidden states"
            )
        if tv_verification_session is None:
            raise ValueError(
                "DFlash2 TV beta requires a target proposal verification session"
            )
        anchors, keep, output = self._forward_draft_blocks(
            input_ids, hidden_states, loss_mask * attention_mask
        )
        batch, blocks = anchors.shape
        # Native DFlash predicts at the same position; slot zero is the anchor.
        hidden = output.reshape(batch, blocks, self.block_size, -1)[:, :, 1:]
        positions = anchors[..., None] + torch.arange(
            1, self.block_size, device=input_ids.device
        )
        rows = torch.arange(batch, device=input_ids.device)[:, None, None]
        safe = positions.clamp(max=input_ids.size(1) - 1)
        valid = (
            (
                keep[..., None]
                & (positions < input_ids.size(1))
                & attention_mask[rows, safe].bool()
                & (loss_mask[rows, safe] > 0.5)
            )
            .int()
            .cumprod(-1)
            .bool()
            .reshape(-1, BETA_BLOCK_SIZE)
        )
        flat_hidden = hidden.reshape(-1, BETA_BLOCK_SIZE, hidden.size(-1))
        rollout = sample_dflash2_beta_proposals(
            self, flat_hidden, input_ids.gather(1, anchors).reshape(-1), valid
        )
        teacher = verify_beta_proposals(
            tv_verification_session,
            input_ids,
            anchors,
            rollout,
            last_hidden_states,
            self.tv_verification_batch_size,
        ).to(hidden)

        def chunk_terms(h, prev, candidates, target_h, mask, tokens, old):
            logits = dflash2_proposal_logits(self, h, prev, candidates)
            with torch.no_grad():
                target_logits = self.lm_head(target_h)
            return tuple(
                tv_acceptance_beta_terms(
                    logits, target_logits, mask, tokens, old, self.tv_temperature
                )
            )

        terms = TVAcceptanceTerms(
            *checkpointed_chunk_reduce(
                chunk_terms,
                flat_hidden,
                rollout.previous,
                rollout.candidate_ids,
                teacher,
                rollout.valid,
                rollout.tokens,
                rollout.log_probs,
                chunk_size=self.tv_objective_chunk_blocks,
            )
        )
        result = reduce_tv_terms(terms)
        result[-1]["tv_acceptance_length_estimate"] = result[-1].pop(
            "tv_acceptance_length_proxy"
        )
        return result
