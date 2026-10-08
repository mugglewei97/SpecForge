"""DSpark replay from the exact anchor, including the rejected proposal suffix."""

import torch
from torch import nn

from .objective import block_loss, draft_distribution


def replay_inputs(
    context_length, block_size, anchor, mask_token, device, layer_types, window=None
):
    ids = torch.full((1, block_size), mask_token, device=device, dtype=torch.long)
    ids[0, 0] = anchor
    positions = torch.arange(context_length + block_size, device=device)[None]
    q = torch.arange(block_size, device=device)[:, None]
    kv = torch.arange(context_length + block_size, device=device)[None, :]
    masks = {}
    for kind in set(layer_types):
        allowed = torch.ones(
            (block_size, context_length + block_size), device=device, dtype=torch.bool
        )
        if kind == "sliding_attention":
            if window is None or window <= 0:
                raise ValueError("sliding replay requires a positive window")
            allowed = (kv >= context_length + q - (window - 1)) & (
                kv <= context_length + q
            )
        elif kind != "full_attention":
            raise ValueError(f"unsupported DSpark layer: {kind}")
        masks[kind] = allowed[None, None]
    return ids, positions, masks


class DSparkReplayModel(nn.Module):
    def __init__(self, draft_model, target_parts):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_parts.lm_head.requires_grad_(False).eval()
        self.embed_tokens = target_parts.embed_tokens.requires_grad_(False).eval()
        if draft_model.confidence_head is not None:
            # The requested objective supervises q, not the scheduling predictor.
            draft_model.confidence_head.requires_grad_(False)

    def forward(self, block, context_hidden, target_probs, rollout_q):
        draft = self.draft_model
        device = self.embed_tokens.weight.device
        k = len(block["proposal"])
        if k != draft.block_size:
            raise ValueError("trajectory block size differs from the checkpoint")
        context_length = block["context_length"]
        if context_hidden.shape[0] != context_length:
            raise ValueError("target context features do not cover the recorded prefix")
        ids, positions, masks = replay_inputs(
            context_length,
            k,
            block["anchor"],
            draft.mask_token_id,
            device,
            draft.layer_types,
            draft.sliding_window,
        )
        hidden = draft(
            position_ids=positions,
            attention_mask=masks,
            noise_embedding=self.embed_tokens(ids),
            target_hidden=context_hidden.to(
                device=device, dtype=self.embed_tokens.weight.dtype
            )[None],
            use_cache=False,
        )
        # Every preceding token comes from the original proposal, not the final
        # accepted text. RNN heads unroll once from the block's zero state.
        previous = torch.tensor(
            [[block["anchor"], *block["proposal"][:-1]]],
            device=device,
            dtype=torch.long,
        )
        logits = draft.apply_logits_head(
            self.lm_head(hidden), prev_token_ids=previous, hidden_states=hidden
        )[0]
        q = draft_distribution(logits, block["sampling"]["temperature"])
        mask = torch.tensor(block["valid_mask"], device=device, dtype=torch.bool)
        p = target_probs.to(device=device)
        loss = block_loss(p, q, mask)
        parity = (q.detach() - rollout_q.to(device=device)).abs().sum(-1).mul(0.5).max()
        return loss, parity
