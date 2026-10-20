"""Incremental target sessions for exact speculative rollout and block replay.

The correction/bonus token is committed without running the target. It is the
next draft anchor, whose auxiliary features are not visible to the draft. The
next verification consumes this pending token and the new proposals together.
Only ``verify`` results, not pending rows in ``output.last_hidden_states``, may
be used as next-token distributions.
"""

from dataclasses import dataclass

import torch

from .proposal_verification import right_padded_lengths
from .target_capture_policy import DFlashCapturePolicy, DFlashTargetOutput


@dataclass
class _Branch:
    tokens: torch.Tensor
    aux: torch.Tensor
    final: torch.Tensor
    cache: object
    cpu_tokens: list[int] | None = None


class OnPolicyTargetSession:
    """State machine shared by cached HF and SGLang target implementations."""

    def __init__(self, engine):
        self.engine = engine
        self.output = None
        self.closed = False
        self._rows = []
        self._pending = []
        self._branches = None
        self._seed_available = True
        self._capacity = None
        self._buffers = None
        # Only backends with a CPU request API populate this mirror. It includes
        # the pending seed/correction/bonus, just like each committed tensor row.
        self._cpu_tokens = None
        # Read-only snapshots for rollout sizing, including any pending anchor.
        # Replace the device vector on append; consumers may retain old lengths.
        self.lengths = None
        self.lengths_cpu = ()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        self._rows = []
        self._pending = []
        self._branches = None
        self._buffers = None
        self._cpu_tokens = None
        self.lengths = None
        self.lengths_cpu = ()
        self.output = None
        self.closed = True

    def reserve_capacity(self, length):
        """Reserve committed feature storage before rollout begins.

        Capture/commit outputs then expose views of the same storage. Previously
        pending hidden rows are filled in place by later commits; callers that
        require historical snapshots must clone them explicitly.
        """
        if self.closed or not self._seed_available:
            raise RuntimeError("reserve capacity before seeding/verifying the rollout")
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise ValueError("on-policy storage capacity must be a positive integer")
        if self._rows and length < max(len(row[0]) for row in self._rows):
            raise ValueError("on-policy storage capacity is smaller than the prefix")
        self._capacity = length
        self._buffers = None
        if self.output is not None:
            self._pack()

    def _check_open(self):
        if self.closed or self.output is None:
            raise RuntimeError("on-policy rollout requires an active captured session")

    @torch.no_grad()
    def capture(self, input_ids, attention_mask, loss_mask):
        if self.closed or self.output is not None:
            raise RuntimeError("capture may only run once per on-policy session")
        if (
            input_ids.ndim != 2
            or input_ids.shape != attention_mask.shape
            or input_ids.shape != loss_mask.shape
        ):
            raise ValueError(
                "on-policy capture requires aligned [batch, sequence] tensors"
            )
        if input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("on-policy capture requires nonempty prefixes")
        device_lengths = right_padded_lengths(attention_mask)
        lengths = device_lengths.cpu().tolist()
        if self._capacity is not None and max(lengths) > self._capacity:
            raise ValueError("on-policy storage capacity is smaller than the prefix")
        try:
            captured = self._capture(
                input_ids, attention_mask, loss_mask, lengths=lengths
            )
            if captured.last_hidden_states is None:
                raise ValueError(
                    "on-policy capture requires target final hidden states"
                )
            self._rows = [
                tuple(
                    value[i, :length].detach().clone()
                    for value in (
                        input_ids,
                        captured.hidden_states,
                        captured.last_hidden_states,
                        loss_mask,
                    )
                )
                for i, length in enumerate(lengths)
            ]
            self._pending = [False] * len(lengths)
            self.lengths = device_lengths
            return self._pack()
        except BaseException:
            self.close()
            raise

    def _pack(self):
        # Tensor row shapes already reflect validated seed/commit metadata.
        # Reading their lengths needs neither a mask reduction nor a D2H copy.
        self.lengths_cpu = tuple(len(row[0]) for row in self._rows)
        batch, width = len(self._rows), max(self.lengths_cpu)
        ids, aux, final, loss = self._rows[0]
        if self._buffers is None:
            capacity = self._capacity or width
            packed_ids = ids.new_zeros((batch, capacity))
            packed_aux = aux.new_zeros((batch, capacity, aux.size(-1)))
            packed_final = final.new_zeros((batch, capacity, final.size(-1)))
            packed_loss = loss.new_zeros((batch, capacity))
            mask = packed_ids.new_zeros((batch, capacity))
            for row, (ids, aux, final, loss) in enumerate(self._rows):
                length = len(ids)
                packed_ids[row, :length] = ids
                packed_aux[row, :length] = aux
                packed_final[row, :length] = final
                packed_loss[row, :length] = loss
                mask[row, :length] = 1
            if self._capacity is not None:
                self._buffers = (
                    packed_ids,
                    packed_aux,
                    packed_final,
                    packed_loss,
                    mask,
                )
                self._rows = [
                    tuple(value[row, : len(data[0])] for value in self._buffers[:4])
                    for row, data in enumerate(self._rows)
                ]
        else:
            packed_ids, packed_aux, packed_final, packed_loss, mask = self._buffers
        self.output = DFlashTargetOutput(
            hidden_states=packed_aux[:, :width],
            input_ids=packed_ids[:, :width],
            attention_mask=mask[:, :width],
            loss_mask=packed_loss[:, :width],
            last_hidden_states=packed_final[:, :width],
        )
        return self.output

    def _append_row(self, row, accepted, processed_aux, processed_final, next_token):
        ids, aux, final, loss = self._rows[row]
        old_length = len(ids)
        prefix_length = old_length - int(self._pending[row])
        accepted_end = old_length + len(accepted)
        length = accepted_end + int(next_token is not None)
        if self._buffers is not None:
            packed_ids, packed_aux, packed_final, packed_loss, mask = self._buffers
            packed_ids[row, old_length:accepted_end] = accepted
            packed_aux[row, prefix_length:accepted_end] = processed_aux
            packed_final[row, prefix_length:accepted_end] = processed_final
            if next_token is not None:
                packed_ids[row, accepted_end] = next_token
                packed_aux[row, accepted_end].zero_()
                packed_final[row, accepted_end].zero_()
            packed_loss[row, old_length:length] = 1
            mask[row, old_length:length] = 1
            self._rows[row] = tuple(value[row, :length] for value in self._buffers[:4])
        else:
            ids = torch.cat((ids, accepted))
            aux = torch.cat((aux[:prefix_length], processed_aux))
            final = torch.cat((final[:prefix_length], processed_final))
            loss = torch.cat((loss, loss.new_ones(len(accepted))))
            if next_token is not None:
                ids = torch.cat((ids, next_token.reshape(1).to(ids)))
                aux = torch.cat((aux, aux.new_zeros((1, aux.size(-1)))))
                final = torch.cat((final, final.new_zeros((1, final.size(-1)))))
                loss = torch.cat((loss, loss.new_ones(1)))
            self._rows[row] = (ids, aux, final, loss)

    @torch.no_grad()
    def seed(self, next_tokens, active):
        """Append the target-prefill sample as the first pending draft anchor."""
        self._check_open()
        if not self._seed_available:
            raise RuntimeError(
                "seed is only allowed once, before the first verification"
            )
        if next_tokens.shape != (len(self._rows),) or active.shape != next_tokens.shape:
            raise ValueError("seed requires one token and active flag per row")
        if next_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("seed requires integer token IDs")
        metadata = torch.stack((active.bool(), next_tokens), -1).cpu().tolist()
        if self._capacity is not None and any(
            enabled and len(self._rows[row][0]) + 1 > self._capacity
            for row, (enabled, _) in enumerate(metadata)
        ):
            raise ValueError("seed would exceed reserved on-policy capacity")
        for row, (enabled, token) in enumerate(metadata):
            if enabled:
                ids, aux, final, _ = self._rows[row]
                self._append_row(row, ids[:0], aux[:0], final[:0], next_tokens[row])
                if self._cpu_tokens is not None:
                    self._cpu_tokens[row].append(token)
                self._pending[row] = True
        self.lengths = self.lengths + active.bool().to(self.lengths)
        self._seed_available = False
        return self._pack()

    @torch.no_grad()
    def verify(self, tokens, valid, active):
        """Return [B, K+1, H]; slots 0..n predict n proposals and the bonus.

        Invalid slots and inactive rows are zero. Validity is a contiguous prefix
        and is independent of the committed output length. A verify must be
        followed by one commit before another verify.
        """
        self._check_open()
        batch = len(self._rows)
        if self._branches is not None:
            raise RuntimeError("commit the verified block before verifying another")
        if (
            tokens.ndim != 2
            or tokens.shape != valid.shape
            or tokens.size(0) != batch
            or active.shape != (batch,)
        ):
            raise ValueError(
                "on-policy verification requires aligned tokens, valid and active"
            )
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("on-policy verification requires integer token IDs")
        contiguous = valid.bool().int().cumprod(-1).bool()
        # One transfer for all rows, including the validity check. SGLang also
        # needs the new proposal IDs, but never needs to reread the GPU prefix.
        metadata = torch.stack(
            (
                contiguous.sum(-1),
                active.bool(),
                (valid.bool() == contiguous).all(-1),
            ),
            -1,
        )
        if self._cpu_tokens is not None:
            metadata = torch.cat((metadata, tokens), -1)
        metadata = metadata.cpu().tolist()
        if not all(values[2] for values in metadata):
            raise ValueError("proposal validity must be a contiguous prefix")
        self._seed_available = False
        counts = [values[0] for values in metadata]
        suffixes, rows = [], []
        suffixes_cpu = [] if self._cpu_tokens is not None else None
        for row, values in enumerate(metadata):
            count, enabled = values[:2]
            if enabled:
                prefix = (
                    self._rows[row][0][-1:] if self._pending[row] else tokens[row, :0]
                )
                suffixes.append(torch.cat((prefix, tokens[row, :count])))
                if suffixes_cpu is not None:
                    prefix_cpu = (
                        self._cpu_tokens[row][-1:] if self._pending[row] else []
                    )
                    suffixes_cpu.append(prefix_cpu + values[3 : 3 + count])
                rows.append(row)
        try:
            branches = self._verify(rows, suffixes, suffixes_cpu=suffixes_cpu)
            final = self._rows[0][2]
            result = final.new_zeros((batch, tokens.size(1) + 1, final.size(-1)))
            self._branches = {}
            for row, branch in zip(rows, branches, strict=True):
                offset = int(self._pending[row])
                if offset:
                    result[row, : counts[row] + 1] = branch.final
                else:
                    result[row, 0] = self._rows[row][2][-1]
                    result[row, 1 : counts[row] + 1] = branch.final
                self._branches[row] = _Branch(
                    tokens[row, : counts[row]].detach().clone(),
                    branch.aux,
                    branch.final,
                    branch.cache,
                    metadata[row][3 : 3 + counts[row]]
                    if self._cpu_tokens is not None
                    else None,
                )
            return result
        except BaseException:
            self.close()
            raise

    @torch.no_grad()
    def commit(self, accepted_counts, next_tokens, active, emit_next):
        """Commit accepted proposals plus an optional pending correction/bonus."""
        self._check_open()
        if self._branches is None:
            raise RuntimeError("verify a proposal block before commit")
        batch = len(self._rows)
        if any(
            value.shape != (batch,)
            for value in (accepted_counts, next_tokens, active, emit_next)
        ):
            raise ValueError("commit requires one accepted count/token/flag per row")
        if accepted_counts.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "accepted counts must be integers within each proposal block"
            )
        if next_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("commit requires integer token IDs")
        # Keep integer IDs/counts exact; packing through float would lose large
        # token IDs. The next token is reused by the CPU mirror at commit time.
        metadata = (
            torch.stack(
                (active.bool(), accepted_counts, emit_next.bool(), next_tokens), -1
            )
            .cpu()
            .tolist()
        )
        enabled, counts, emits, next_ids = zip(*metadata)
        if {row for row, flag in enumerate(enabled) if flag} != set(self._branches):
            raise ValueError("commit active rows must match the verified block")
        if any(
            not isinstance(counts[row], int)
            or not 0 <= counts[row] <= len(branch.tokens)
            for row, branch in self._branches.items()
        ):
            raise ValueError(
                "accepted counts must be integers within each proposal block"
            )
        if self._capacity is not None and any(
            len(self._rows[row][0]) + counts[row] + int(emits[row]) > self._capacity
            for row in self._branches
        ):
            raise ValueError("commit would exceed reserved on-policy capacity")
        try:
            for row, branch in self._branches.items():
                old_length = len(self._rows[row][0])
                consumed = int(self._pending[row]) + counts[row]
                self._commit_cache(row, branch.cache, old_length + counts[row])
                self._append_row(
                    row,
                    branch.tokens[: counts[row]],
                    branch.aux[:consumed],
                    branch.final[:consumed],
                    next_tokens[row] if emits[row] else None,
                )
                if self._cpu_tokens is not None:
                    self._cpu_tokens[row].extend(branch.cpu_tokens[: counts[row]])
                    if emits[row]:
                        self._cpu_tokens[row].append(next_ids[row])
                self._pending[row] = bool(emits[row])
            increment = torch.where(
                active.bool(), accepted_counts + emit_next.bool(), 0
            )
            self.lengths = self.lengths + increment.to(self.lengths)
            self._branches = None
            return self._pack()
        except BaseException:
            self.close()
            raise


class HFOnPolicyTargetSession(OnPolicyTargetSession):
    """One batched Transformer extend per round, compacted per-row prefix KV."""

    def __init__(self, engine):
        super().__init__(engine)
        self._cache_rows = []

    def close(self):
        self._cache_rows = []
        super().close()

    def _capture(self, input_ids, attention_mask, loss_mask, *, lengths):
        output = DFlashCapturePolicy().hf_capture(
            self.engine.model,
            self.engine.capture_layer_ids,
            input_ids,
            attention_mask,
            loss_mask,
            use_cache=True,
        )
        self._cache_rows = [
            [
                (
                    key[row : row + 1, :, :length].clone(),
                    value[row : row + 1, :, :length].clone(),
                )
                for key, value, *_ in output.past_key_values
            ]
            for row, length in enumerate(lengths)
        ]
        output.past_key_values = None
        return output

    def _verify(self, rows, suffixes, *, suffixes_cpu=None):
        from transformers.cache_utils import DynamicCache

        branches = {}
        nonempty = [
            (row, suffix)
            for row, suffix in zip(rows, suffixes, strict=True)
            if len(suffix)
        ]
        for row, suffix in zip(rows, suffixes, strict=True):
            if not len(suffix):
                branches[row] = _Branch(
                    suffix,
                    self._rows[row][1][:0],
                    self._rows[row][2][:0],
                    self._cache_rows[row],
                )
        if nonempty:
            prefix_lengths = [
                len(self._rows[row][0]) - int(self._pending[row]) for row, _ in nonempty
            ]
            prefix_width = max(prefix_lengths)
            suffix_width = max(len(suffix) for _, suffix in nonempty)
            cache = DynamicCache()
            for layer in range(len(self._cache_rows[nonempty[0][0]])):
                packed = []
                for component in (0, 1):
                    example = self._cache_rows[nonempty[0][0]][layer][component]
                    value = example.new_zeros(
                        (len(nonempty), example.size(1), prefix_width, example.size(-1))
                    )
                    for i, ((row, _), length) in enumerate(
                        zip(nonempty, prefix_lengths, strict=True)
                    ):
                        value[i, :, :length] = self._cache_rows[row][layer][component][
                            0
                        ]
                    packed.append(value)
                cache.update(*packed, layer)
            ids = self.output.input_ids.new_zeros((len(nonempty), suffix_width))
            mask = ids.new_zeros((len(nonempty), prefix_width + suffix_width))
            for i, ((_, suffix), length) in enumerate(
                zip(nonempty, prefix_lengths, strict=True)
            ):
                ids[i, : len(suffix)] = suffix
                mask[i, :length] = 1
                mask[i, prefix_width : prefix_width + len(suffix)] = 1
            positions = torch.tensor(prefix_lengths, device=ids.device)[:, None]
            positions = positions + torch.arange(suffix_width, device=ids.device)
            output = self.engine.model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            layers = self.engine.capture_layer_ids
            aux = (
                torch.cat([output.hidden_states[layer + 1] for layer in layers], -1)
                if layers is not None
                else output.hidden_states[-1]
            )
            final = output.hidden_states[-1]
            for i, ((row, suffix), length) in enumerate(
                zip(nonempty, prefix_lengths, strict=True)
            ):
                row_cache = [
                    tuple(
                        torch.cat(
                            (
                                value[i : i + 1, :, :length],
                                value[
                                    i : i + 1,
                                    :,
                                    prefix_width : prefix_width + len(suffix),
                                ],
                            ),
                            dim=2,
                        )
                        for value in (key, val)
                    )
                    for key, val, *_ in output.past_key_values
                ]
                branches[row] = _Branch(
                    suffix,
                    aux[i, : len(suffix)].detach(),
                    final[i, : len(suffix)].detach(),
                    row_cache,
                )
        return [branches[row] for row in rows]

    def _commit_cache(self, row, cache, keep):
        self._cache_rows[row] = [
            tuple(value[:, :, :keep].clone() for value in layer) for layer in cache
        ]
