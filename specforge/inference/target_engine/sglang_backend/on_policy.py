"""SGLang speculative-rollout sessions with owned, page-aligned prefix KV."""

import torch

from ..on_policy import OnPolicyTargetSession, _Branch
from ..target_capture_policy import DFlashTargetOutput
from .proposal_cache import SGLangProposalVerificationSession


class SGLangOnPolicyTargetSession(OnPolicyTargetSession):
    """Share immutable full prefix pages; commit/discard candidate pages exactly."""

    _request = staticmethod(SGLangProposalVerificationSession._request)
    _forward = SGLangProposalVerificationSession._forward

    def __init__(self, engine):
        super().__init__(engine)
        self.backend = engine._backend
        self._indices = []
        self.owns_pools = False

    def __enter__(self):
        if getattr(self.backend, "_proposal_session_active", False):
            raise RuntimeError("SGLang target sessions cannot overlap")
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
            self._indices = []
            super().close()

    def _capture(self, input_ids, attention_mask, loss_mask):
        if not self.owns_pools:
            raise RuntimeError("SGLang on-policy capture requires a context manager")
        lengths = attention_mask.bool().sum(-1).cpu().tolist()
        requests = [
            self._request(ids[:length], i)
            for i, (ids, length) in enumerate(
                zip(input_ids.cpu().tolist(), lengths, strict=True)
            )
        ]
        output = self._forward(requests)
        if output.last_hidden_states is None or output.aux_hidden_states is None:
            raise ValueError(
                "on-policy target must return auxiliary and final hidden states"
            )
        pool = self.backend.model_runner.req_to_token_pool
        self._indices = [
            pool.req_to_token[req.req_pool_idx, :length].to(torch.int64).clone()
            for req, length in zip(requests, lengths, strict=True)
        ]
        for req in requests:
            pool.free(req)
        shape = input_ids.shape
        aux = output.aux_hidden_states.new_zeros(
            (*shape, output.aux_hidden_states.size(-1))
        )
        final = output.last_hidden_states.new_zeros(
            (*shape, output.last_hidden_states.size(-1))
        )
        for i, (length, a, f) in enumerate(
            zip(
                lengths,
                output.aux_hidden_states.split(lengths),
                output.last_hidden_states.split(lengths),
                strict=True,
            )
        ):
            aux[i, :length], final[i, :length] = a, f
        return DFlashTargetOutput(aux, input_ids, attention_mask, loss_mask, final)

    def _verify(self, rows, suffixes):
        runner = self.backend.model_runner
        page_size = runner.server_args.page_size
        branches, requests, work = {}, [], []
        for row, suffix in zip(rows, suffixes, strict=True):
            if not len(suffix):
                branches[row] = _Branch(
                    suffix,
                    self._rows[row][1][:0],
                    self._rows[row][2][:0],
                    self._indices[row],
                )
                continue
            prefix = len(self._indices[row])
            shared = prefix // page_size * page_size
            tokens = self._rows[row][0][:prefix].cpu().tolist() + suffix.cpu().tolist()
            requests.append(self._request(tokens, row, self._indices[row][:shared]))
            work.append((row, suffix, len(tokens), len(tokens) - shared))
        if requests:
            output = self._forward(requests)
            if output.last_hidden_states is None or output.aux_hidden_states is None:
                raise ValueError(
                    "on-policy target must return auxiliary and final hidden states"
                )
            counts = [item[3] for item in work]
            aux_rows = output.aux_hidden_states.split(counts)
            final_rows = output.last_hidden_states.split(counts)
            pool = runner.req_to_token_pool
            for req, (row, suffix, length, _), aux, final in zip(
                requests, work, aux_rows, final_rows, strict=True
            ):
                indices = (
                    pool.req_to_token[req.req_pool_idx, :length].to(torch.int64).clone()
                )
                branches[row] = _Branch(
                    suffix,
                    aux[-len(suffix) :].clone(),
                    final[-len(suffix) :].clone(),
                    indices,
                )
                pool.free(req)
        return [branches[row] for row in rows]

    def _commit_cache(self, row, cache, keep):
        runner = self.backend.model_runner
        page_size = runner.server_args.page_size
        retained = cache[:keep].clone()
        # The page containing the last retained token remains live in its
        # entirety. Freeing cache[keep:] directly would free that partial page.
        owned_pages = torch.unique(torch.cat((self._indices[row], cache)) // page_size)
        kept_pages = torch.unique(retained // page_size)
        discarded = owned_pages[~torch.isin(owned_pages, kept_pages)]
        if discarded.numel():
            runner.token_to_kv_pool_allocator.free(discarded * page_size)
        self._indices[row] = retained
