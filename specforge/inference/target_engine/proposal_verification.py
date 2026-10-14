"""Per-forward target verification sessions; never retained by loss checkpoints."""

import torch


def supports_qwen3_prefix_cache(config):
    """Only full-attention Qwen3 has a validated reusable-cache implementation."""
    if getattr(config, "model_type", None) != "qwen3":
        return False
    layers = getattr(config, "layer_types", None)
    if layers:
        return all(kind == "full_attention" for kind in layers)
    return not getattr(config, "use_sliding_window", False) and not getattr(
        config, "sliding_window", None
    )


def right_padded_lengths(attention_mask):
    """SGLang's request API represents lengths rather than arbitrary masks."""
    lengths = attention_mask.bool().sum(-1)
    expected = (
        torch.arange(attention_mask.size(1), device=attention_mask.device)[None]
        < lengths[:, None]
    )
    if not torch.equal(attention_mask.bool(), expected) or (lengths == 0).any():
        raise ValueError("SGLang TV beta requires nonempty, right-padded sequences")
    return lengths


class ProposalVerificationSession:
    """Portable full-candidate fallback for engines with no reusable prefix KV."""

    default_batch_size = 4

    def __init__(self, engine):
        self.engine = engine
        self.input_ids = None
        self.attention_mask = None
        self.output = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        self.input_ids = self.attention_mask = self.output = None
        self.closed = True

    def _check_open(self):
        if self.closed or self.input_ids is None:
            raise RuntimeError(
                "proposal verification requires an active captured session"
            )

    @torch.no_grad()
    def capture(self, input_ids, attention_mask, loss_mask):
        if self.closed or self.input_ids is not None:
            raise RuntimeError("capture may only run once per proposal session")
        if getattr(self.engine, "backend", None) == "sglang":
            right_padded_lengths(attention_mask)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.output = self.engine.generate_dflash_data(
            input_ids, attention_mask, loss_mask
        )
        return self.output

    @torch.no_grad()
    def score_suffixes(self, rows, prefix_lengths, suffix_ids):
        self._check_open()
        lengths = [
            prefix + len(suffix)
            for prefix, suffix in zip(prefix_lengths, suffix_ids, strict=True)
        ]
        ids = self.input_ids.new_zeros((len(rows), max(lengths)))
        mask = self.attention_mask.new_zeros(ids.shape)
        for i, (row, prefix, suffix) in enumerate(
            zip(rows, prefix_lengths, suffix_ids, strict=True)
        ):
            ids[i, :prefix] = self.input_ids[row, :prefix]
            ids[i, prefix : prefix + len(suffix)] = suffix
            mask[i, :prefix] = self.attention_mask[row, :prefix]
            mask[i, prefix : prefix + len(suffix)] = 1
        output = self.engine.generate_dflash_data(ids, mask, mask.float())
        if output.last_hidden_states is None:
            raise ValueError(
                "proposal verification requires target final hidden states"
            )
        return [
            output.last_hidden_states[i, prefix:length].detach()
            for i, (prefix, length) in enumerate(zip(prefix_lengths, lengths))
        ]


class HFProposalVerificationSession(ProposalVerificationSession):
    """Copy only the needed prefix KV; batch ragged prefixes with masked padding."""

    @torch.no_grad()
    def capture(self, input_ids, attention_mask, loss_mask):
        from .target_capture_policy import DFlashCapturePolicy

        if self.closed or self.input_ids is not None:
            raise RuntimeError("capture may only run once per proposal session")
        self.input_ids, self.attention_mask = input_ids, attention_mask
        self.output = DFlashCapturePolicy().hf_capture(
            self.engine.model,
            self.engine.capture_layer_ids,
            input_ids,
            attention_mask,
            loss_mask,
            use_cache=True,
        )
        return self.output

    def close(self):
        if self.output is not None:
            self.output.past_key_values = None
        super().close()

    @torch.no_grad()
    def score_suffixes(self, rows, prefix_lengths, suffix_ids):
        from transformers.cache_utils import DynamicCache

        self._check_open()
        original = self.output.past_key_values
        if original is None:
            raise RuntimeError("Qwen3 capture did not return prefix KV")
        prefix_width = max(prefix_lengths)
        suffix_width = max(map(len, suffix_ids))
        row_indices = torch.tensor(rows, device=self.input_ids.device, dtype=torch.long)
        lengths = torch.tensor(prefix_lengths, device=self.input_ids.device)
        cache = DynamicCache()
        for layer, kv in enumerate(original):
            key, value = kv[:2]
            # index_select owns its storage: candidate updates never mutate the
            # original cache. Masked unused reference suffix KV is never attended.
            cache.update(
                key[..., :prefix_width, :].index_select(0, row_indices),
                value[..., :prefix_width, :].index_select(0, row_indices),
                layer,
            )
        ids = self.input_ids.new_zeros((len(rows), suffix_width))
        mask = self.attention_mask.new_zeros((len(rows), prefix_width + suffix_width))
        for i, (row, prefix, suffix) in enumerate(
            zip(rows, prefix_lengths, suffix_ids, strict=True)
        ):
            ids[i, : len(suffix)] = suffix
            mask[i, :prefix] = self.attention_mask[row, :prefix]
            mask[i, prefix_width : prefix_width + len(suffix)] = 1
        positions = lengths[:, None] + torch.arange(suffix_width, device=ids.device)
        outputs = self.engine.model.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
        )
        return [
            outputs.last_hidden_state[i, : len(suffix)].detach()
            for i, suffix in enumerate(suffix_ids)
        ]
