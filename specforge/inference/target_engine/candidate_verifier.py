"""Score the target on each draft block's actual, independent candidate prefix."""

from itertools import islice
from typing import NamedTuple

import torch
from torch.nn.utils.rnn import pad_sequence


class _CandidateRequest(NamedTuple):
    batch: int
    block: int
    anchor: int
    count: int
    tokens: torch.Tensor


class CandidatePrefixVerifier:
    """Capture final hidden states on context[:anchor+1] + candidates[:K-1].

    Rows anchor..anchor+K-1 predict the K candidates. The frozen LM head is
    applied later in bounded objective chunks on the trainer.
    """

    def __init__(self, target_model, batch_size=4, pad_token_id=0):
        if batch_size < 1:
            raise ValueError("candidate verification batch_size must be positive")
        self.target_model = target_model
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id

    @torch.no_grad()
    def __call__(
        self, input_ids, attention_mask, anchors, proposals, valid, hidden_size
    ):
        requests = self._iter_requests(
            input_ids, attention_mask, anchors, proposals, valid
        )
        shape = (*proposals.shape, hidden_size)
        result = None
        while batch := list(islice(requests, self.batch_size)):
            last = self._capture(batch, input_ids.device, hidden_size)
            if result is None:
                result = last.new_zeros(shape, device=input_ids.device)
            for row, request in enumerate(batch):
                start, count = request.anchor, request.count
                result[request.batch, request.block, :count] = last[
                    row, start : start + count
                ].to(result.device)
        if result is None:
            return torch.zeros(shape, device=input_ids.device)
        return result

    def _capture(self, requests, device, hidden_size):
        # Pad on CPU, then transfer once per batch instead of once per prefix.
        ids = pad_sequence(
            [request.tokens for request in requests],
            batch_first=True,
            padding_value=self.pad_token_id,
        ).to(device)
        lengths = torch.tensor(
            [len(request.tokens) for request in requests], device=device
        )
        mask = (
            torch.arange(ids.size(1), device=device)[None] < lengths[:, None]
        ).long()
        output = self.target_model.generate_dflash_data(ids, mask, mask)
        last = output.last_hidden_states
        if last is None:
            raise ValueError(
                "tv-acceptance requires target final hidden states on sampled prefixes"
            )
        if last.shape != (*ids.shape, hidden_size):
            raise ValueError(
                "target capture returned unexpected candidate hidden-state shape"
            )
        return last

    @staticmethod
    def _iter_requests(input_ids, attention_mask, anchors, proposals, valid):
        if (
            proposals.ndim != 3
            or proposals.shape != valid.shape
            or anchors.shape != proposals.shape[:2]
            or input_ids.ndim != 2
            or input_ids.size(0) != anchors.size(0)
        ):
            raise ValueError(
                "candidate anchors, proposals and valid mask are misaligned"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        # Absolute token positions in this training path require right padding.
        lengths = attention_mask.long().sum(-1)
        expected = (
            torch.arange(input_ids.size(1), device=input_ids.device)[None]
            < lengths[:, None]
        )
        if not torch.equal(attention_mask.bool(), expected):
            raise ValueError("candidate verification requires right-padded inputs")
        counts = valid.bool().int().cumprod(-1).sum(-1).cpu()
        anchor_cpu, input_cpu = anchors.cpu(), input_ids.cpu()
        proposal_cpu, lengths_cpu = proposals.cpu(), lengths.cpu()
        for b in range(anchors.size(0)):
            for block in range(anchors.size(1)):
                count = int(counts[b, block])
                if count == 0:
                    continue
                anchor = int(anchor_cpu[b, block])
                if anchor < 0 or anchor + count >= int(lengths_cpu[b]):
                    raise ValueError(
                        "candidate block exceeds the original valid sequence"
                    )
                tokens = torch.cat(
                    [input_cpu[b, : anchor + 1], proposal_cpu[b, block, : count - 1]]
                )
                yield _CandidateRequest(b, block, anchor, count, tokens)
