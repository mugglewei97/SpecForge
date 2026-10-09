"""DSpark replay from the exact anchor, including the rejected proposal suffix."""

import torch
from torch import nn

from .objective import block_loss, draft_distribution


def create_replay_draft(config):
    """BF16 parameters with RoPE frequencies computed and retained in FP32."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    from specforge.modeling.auto import AutoDraftModel

    draft = AutoDraftModel.from_config(config, torch_dtype=torch.bfloat16)
    # AutoDraftModel casts the entire module, including nonpersistent inv_freq.
    # Casting that rounded buffer back to FP32 in FSDP cannot recover it. Rebuild
    # from the config, like SGLang, before any device-only move/FSDP wrapping.
    draft.rotary_emb = Qwen3RotaryEmbedding(config, device=draft.fc.weight.device)
    return draft


def replay_inputs(
    context_length,
    block_size,
    anchor,
    mask_token,
    device,
    layer_types,
    window=None,
    attention_backend="sdpa",
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
        if attention_backend == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            # Use arithmetic masks, not indexing into a dense tensor: flex mask
            # construction evaluates padded query/KV indices at block boundaries.
            def full_mask(b, h, qi, ki):
                return (qi < block_size) & (ki < context_length + block_size)

            def sliding_mask(b, h, qi, ki):
                return (
                    full_mask(b, h, qi, ki)
                    & (ki >= context_length + qi - (window - 1))
                    & (ki <= context_length + qi)
                )

            masks[kind] = create_block_mask(
                sliding_mask if kind == "sliding_attention" else full_mask,
                B=1,
                H=None,
                Q_LEN=block_size,
                KV_LEN=context_length + block_size,
                device=str(device),
            )
        else:
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

    def draft_probabilities(self, block, context_hidden):
        """Recompute q without a loss; also used by read-only trajectory audits."""
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
            draft.config._attn_implementation,
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
        return draft_distribution(logits, block["sampling"]["temperature"])

    def forward(self, block, context_hidden, target_probs, rollout_q):
        q = self.draft_probabilities(block, context_hidden)
        device = q.device
        mask = torch.tensor(block["valid_mask"], device=device, dtype=torch.bool)
        p = target_probs.to(device=device)
        loss = block_loss(p, q, mask)
        # Parity checks only valid positions: invalid suffix tokens have no
        # gradient significance and their numerical drift is irrelevant.
        rollout_q_dev = rollout_q.to(device=device)
        if mask.any():
            parity = (q.detach()[mask] - rollout_q_dev[mask]).abs().sum(-1).mul(0.5).max()
        else:
            parity = torch.tensor(0.0, device=device)
        return loss, parity
