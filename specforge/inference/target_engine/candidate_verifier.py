"""Score the target on each draft block's actual, independent candidate prefix."""

import torch


class CandidatePrefixVerifier:
    """Reuse HF/SGLang capture; never replace full distributions with top-k.

    Each request is original_context[:anchor+1] + candidates[:K-1]. Captured
    final hidden rows anchor..anchor+K-1 predict the K candidates. The frozen
    LM head is applied later in bounded objective chunks on the trainer.
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
        if proposals.shape != valid.shape or anchors.shape != proposals.shape[:2]:
            raise ValueError(
                "candidate anchors, proposals and valid mask are misaligned"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        # This legacy training path uses right padding and absolute token
        # positions; accepting left padding here would change the target prefix.
        lengths = attention_mask.long().sum(-1)
        expected = (
            torch.arange(input_ids.size(1), device=input_ids.device)[None]
            < lengths[:, None]
        )
        if not torch.equal(attention_mask.bool(), expected):
            raise ValueError("candidate verification requires right-padded inputs")
        valid = valid.bool().int().cumprod(-1).bool()
        counts = valid.sum(-1).cpu()
        anchor_cpu = anchors.cpu()
        input_cpu = input_ids.cpu()
        proposal_cpu = proposals.cpu()
        lengths_cpu = lengths.cpu()
        requests = []
        locations = []
        result = None

        def flush():
            nonlocal result
            if not requests:
                return
            width = max(len(tokens) for tokens in requests)
            ids = input_ids.new_full((len(requests), width), self.pad_token_id)
            mask = torch.zeros_like(ids)
            for row, tokens in enumerate(requests):
                ids[row, : len(tokens)] = tokens.to(ids.device)
                mask[row, : len(tokens)] = 1
            output = self.target_model.generate_dflash_data(ids, mask, mask)
            last = output.last_hidden_states
            if last is None:
                raise ValueError(
                    "tv-acceptance requires target final hidden states on sampled prefixes"
                )
            if last.shape != (len(requests), width, hidden_size):
                raise ValueError(
                    "target capture returned unexpected candidate hidden-state shape"
                )
            if result is None:
                result = last.new_zeros(
                    (*proposals.shape, hidden_size), device=input_ids.device
                )
            for row, (b, block, anchor, k) in enumerate(locations):
                result[b, block, :k] = last[row, anchor : anchor + k].to(result.device)
            requests.clear()
            locations.clear()

        for b in range(anchors.size(0)):
            for block in range(anchors.size(1)):
                k = int(counts[b, block])
                if k == 0:
                    continue
                anchor = int(anchor_cpu[b, block])
                if anchor < 0 or anchor + k >= int(lengths_cpu[b]):
                    raise ValueError(
                        "candidate block exceeds the original valid sequence"
                    )
                tokens = torch.cat(
                    [input_cpu[b, : anchor + 1], proposal_cpu[b, block, : k - 1]]
                )
                requests.append(tokens)
                locations.append((b, block, anchor, k))
                if len(requests) == self.batch_size:
                    flush()
        flush()
        if result is None:
            return torch.zeros((*proposals.shape, hidden_size), device=input_ids.device)
        return result
