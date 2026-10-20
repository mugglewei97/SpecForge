"""Step-local context KV shared by native DFlash2 and DSpark rollouts.

DSpark uses every output slot; native DFlash2 drops its first anchor slot.
"""

import torch
from transformers.cache_utils import DynamicCache

from specforge.modeling.draft.flex_attention_backend import flex_attention_backend


class BlockRolloutCache:
    """Reuse frozen-within-rollout draft context KV; discard block noise KV.

    Different rows can advance by different counts. Physical cache padding is
    masked explicitly, and RoPE uses each row's original token positions.
    This cache never crosses an optimizer update or enters backward replay.
    """

    def __init__(self, *, proposal_start=1):
        self.proposal_start = proposal_start
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
        return anchors, hidden[:, self.proposal_start :]
