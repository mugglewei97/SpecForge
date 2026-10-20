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
        self.lengths_cpu = None
        self.valid_keys = None
        self._block_layout = None

    def _block_mask(self, keys, query_length):
        """Build a forward-only mask without evaluating a dense Q x KV mask."""
        from torch.nn.attention.flex_attention import BlockMask

        block_size = (256, 128) if flex_attention_backend() == "FLASH" else (128, 128)
        batch, key_width = keys.shape
        query_blocks = (query_length + block_size[0] - 1) // block_size[0]
        kv_blocks = (key_width + block_size[1] - 1) // block_size[1]
        layout_key = (keys.device, batch, query_blocks, kv_blocks, block_size)
        if self._block_layout is None or self._block_layout[0] != layout_key:
            # All queries share the same visible keys. Visit every KV tile as
            # partial: marking tiles full would bypass the physical-padding and
            # inactive-row checks in mask_mod. No device values need inspection.
            counts = torch.full(
                (batch, 1, query_blocks),
                kv_blocks,
                dtype=torch.int32,
                device=keys.device,
            )
            indices = torch.arange(kv_blocks, dtype=torch.int32, device=keys.device)
            # Triton's sparse kernel flattens batch/head assuming contiguous
            # metadata. A zero-stride expand would read past storage for batch > 1.
            indices = (
                indices[None, None, None, :]
                .expand(batch, 1, query_blocks, -1)
                .contiguous()
            )
            # FLASH requires full-block metadata even when there are no full
            # tiles. Zero counts make the shared indices unused in that path.
            full_counts = torch.zeros_like(counts)
            self._block_layout = (layout_key, counts, indices, full_counts)
        _, counts, indices, full_counts = self._block_layout

        def mask_mod(b, h, q, k):
            return (
                (q < query_length)
                & (k < key_width)
                & keys[b, k.clamp(max=key_width - 1)]
            )

        # Reuse only the layout within a tile-size bucket, never the closure:
        # keys (including active rows) and exact sequence lengths change each round.
        return BlockMask.from_kv_blocks(
            counts,
            indices,
            full_kv_num_blocks=full_counts,
            full_kv_indices=indices,
            BLOCK_SIZE=block_size,
            mask_mod=mask_mod,
            seq_lengths=(query_length, key_width),
            compute_q_blocks=False,  # Rollout runs under no_grad; replay has its own mask.
        )

    @torch.no_grad()
    def forward(self, model, output, active, *, lengths, lengths_cpu):
        """Use matching device/CPU committed-length snapshots from the session."""
        ids, features = output.input_ids, output.hidden_states
        batch = ids.size(0)
        if lengths.shape != (batch,) or len(lengths_cpu) != batch:
            raise ValueError("Committed lengths must have one entry per row")
        if any(length < 1 or length > ids.size(1) for length in lengths_cpu):
            raise ValueError("Committed lengths must fit the nonempty token rows")
        anchors_cpu = tuple(length - 1 for length in lengths_cpu)
        previous_cpu = self.lengths_cpu or (0,) * batch
        delta_cpu = tuple(
            current - previous
            for current, previous in zip(anchors_cpu, previous_cpu, strict=True)
        )
        if any(delta < 0 for delta in delta_cpu):
            raise ValueError("Committed draft context cannot move backwards")
        width = max(delta_cpu)
        anchors = lengths - 1
        if self.lengths is None:
            self.lengths = torch.zeros_like(anchors)
            self.valid_keys = torch.zeros(
                (batch, 0), device=ids.device, dtype=torch.bool
            )
        delta = anchors - self.lengths
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
            mask = self._block_mask(keys, model.block_size)
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
        self.lengths_cpu = anchors_cpu
        self.valid_keys = context_keys
        return anchors, hidden[:, self.proposal_start :]
