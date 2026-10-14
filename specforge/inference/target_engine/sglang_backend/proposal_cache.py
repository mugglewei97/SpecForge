"""Qwen3 proposal verification with session-owned, page-aligned prefix KV.

SGLang internals stay behind the capture backend boundary. The original batch
owns its KV until close(); candidates share full pages and own all new pages.
"""

from array import array

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from ..proposal_verification import ProposalVerificationSession, right_padded_lengths
from ..target_capture_policy import DFlashTargetOutput


class SGLangProposalVerificationSession(ProposalVerificationSession):
    default_batch_size = 64

    def __init__(self, engine):
        super().__init__(engine)
        self.backend = engine._backend
        self.prefix_indices = None
        self.reference_ids = None
        self.owns_pools = False

    def __enter__(self):
        if getattr(self.backend, "_proposal_session_active", False):
            raise RuntimeError("SGLang proposal verification sessions cannot overlap")
        self.backend._proposal_session_active = True
        self.owns_pools = True
        return self

    def close(self):
        try:
            if self.owns_pools:
                self.backend._clear_pools()
        finally:
            if self.owns_pools:
                self.backend._proposal_session_active = False
            self.owns_pools = False
            self.prefix_indices = self.reference_ids = None
            super().close()

    @staticmethod
    def _request(tokens, index, prefix_indices=None):
        request = Req(
            rid=f"tv-beta-{index}",
            origin_input_text="",
            origin_input_ids=tokens,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1, top_k=1),
        )
        if prefix_indices is not None:
            # write_cache_indices requires int64, even when the pool stores int32.
            request.prefix_indices = prefix_indices.to(dtype=torch.int64)
        request.full_untruncated_fill_ids = array("q", tokens)
        request.fill_len = len(tokens)
        request.extend_input_len = request.fill_len - len(request.prefix_indices)
        request.set_extend_range(len(request.prefix_indices), request.fill_len)
        return request

    def _forward(self, requests):
        self.backend._set_eagle3_logits_flags(True, False, False)
        return self.backend._forward_extend(requests)

    @torch.no_grad()
    def capture(self, input_ids, attention_mask, loss_mask):
        if not self.owns_pools or self.closed or self.input_ids is not None:
            raise RuntimeError(
                "capture requires a fresh active SGLang proposal session"
            )
        lengths = right_padded_lengths(attention_mask)
        self.input_ids, self.attention_mask = input_ids, attention_mask
        counts = lengths.cpu().tolist()
        self.reference_ids = [
            row[:n] for row, n in zip(input_ids.cpu().tolist(), counts)
        ]
        requests = [
            self._request(tokens, i) for i, tokens in enumerate(self.reference_ids)
        ]
        try:
            outputs = self._forward(requests)
            if outputs.last_hidden_states is None or outputs.aux_hidden_states is None:
                raise ValueError(
                    "SGLang beta capture requires auxiliary and final hidden states"
                )
            pool = self.backend.model_runner.req_to_token_pool
            self.prefix_indices = [
                pool.req_to_token[req.req_pool_idx, :n].to(torch.int64).clone()
                for req, n in zip(requests, counts)
            ]
            # Requests are no longer needed. Their KV remains session-owned.
            for req in requests:
                pool.free(req)
            shape = input_ids.shape
            hidden = outputs.aux_hidden_states.new_zeros(
                (*shape, outputs.aux_hidden_states.size(-1))
            )
            final = outputs.last_hidden_states.new_zeros(
                (*shape, outputs.last_hidden_states.size(-1))
            )
            aux_rows = outputs.aux_hidden_states.split(counts)
            final_rows = outputs.last_hidden_states.split(counts)
            for row, n in enumerate(counts):
                hidden[row, :n], final[row, :n] = aux_rows[row], final_rows[row]
            self.output = DFlashTargetOutput(
                hidden_states=hidden,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                last_hidden_states=final,
            )
            return self.output
        except BaseException:
            self.close()
            raise

    @torch.no_grad()
    def score_suffixes(self, rows, prefix_lengths, suffix_ids):
        self._check_open()
        runner = self.backend.model_runner
        page_size = runner.server_args.page_size
        requests, extend_lengths = [], []
        # One host transfer per batch, not one synchronization per anchor.
        suffix_lengths = [len(suffix) for suffix in suffix_ids]
        host_suffixes = torch.cat(suffix_ids).cpu().split(suffix_lengths)
        for i, (row, prefix, suffix) in enumerate(
            zip(rows, prefix_lengths, host_suffixes, strict=True)
        ):
            # A partial shared page could be overwritten by a candidate. Replay
            # its tail (at most page_size-1 tokens) into private allocated pages.
            shared = prefix // page_size * page_size
            tokens = self.reference_ids[row][:prefix] + suffix.tolist()
            req = self._request(tokens, i, self.prefix_indices[row][:shared])
            requests.append(req)
            extend_lengths.append(len(tokens) - shared)
        try:
            outputs = self._forward(requests)
            if outputs.last_hidden_states is None:
                raise ValueError(
                    "SGLang verification requires target final hidden states"
                )
            states = outputs.last_hidden_states.split(extend_lengths)
            result = [
                value[-len(suffix) :].clone()
                for value, suffix in zip(states, suffix_ids, strict=True)
            ]
            for req in requests:
                # Only free the candidate's new pages, never its shared prefix.
                allocated = runner.req_to_token_pool.req_to_token[
                    req.req_pool_idx, len(req.prefix_indices) : req.fill_len
                ]
                runner.token_to_kv_pool_allocator.free(allocated)
                runner.req_to_token_pool.free(req)
            return result
        except BaseException:
            # Allocation can fail part-way through batch preparation. Reset the
            # entire owned session rather than guessing which slots were written.
            self.close()
            raise
