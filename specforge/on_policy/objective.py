"""Fixed-trajectory acceptance-length proxy; no score-function gradient."""

import torch


def draft_distribution(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """SGLang 0.5.18 DSpark q: temperature only, including with target top-k/p.

    DSpark's rejection sampler receives the untruncated draft distribution.
    Applying the target's top-k/top-p to q would silently train another policy.
    """
    if temperature < 1e-5:
        raise ValueError("greedy draft sampling has no useful exact TV gradient")
    return torch.softmax(logits.float() / temperature, dim=-1)


def block_loss(
    p: torch.Tensor, q: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """1 - sum_k prod_{i<=k}(1-TV(p_i,q_i)) / K, in FP32.

    Masking precedes arithmetic, so padded NaNs do not poison the loss or
    gradients. A block with no valid candidates is excluded by the caller.
    Acceptance outcomes deliberately do not enter this objective.
    """
    if p.shape != q.shape or p.ndim != 2 or valid_mask.shape != q.shape[:1]:
        raise ValueError("expected matching [K,V] distributions and a [K] mask")
    mask = valid_mask.bool()
    if not bool(mask.any()) or bool((mask[1:] & ~mask[:-1]).any()):
        raise ValueError("valid candidates must form a nonempty prefix")
    pp, qq = p.detach()[mask].float(), q[mask].float()
    if not bool(torch.isfinite(pp).all() & torch.isfinite(qq).all()):
        raise ValueError("non-finite probability at a valid candidate")
    for probabilities in (pp, qq):
        if bool((probabilities < 0).any()) or not torch.allclose(
            probabilities.sum(-1),
            torch.ones_like(probabilities[:, 0]),
            atol=2e-4,
            rtol=0,
        ):
            raise ValueError("invalid sampling distribution")
    acceptance = (1.0 - 0.5 * (pp - qq).abs().sum(-1)).clamp(0, 1)
    return 1.0 - acceptance.cumprod(0).mean()


def block_losses(
    p: torch.Tensor, q: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Independent losses for packed [blocks, K, V] distributions.

    Each block keeps its own valid-prefix denominator and acceptance chain.
    The caller sums these losses with the original per-block sample weight;
    averaging here would overweight a trajectory's final, smaller group.
    """
    if p.shape != q.shape or p.ndim != 3 or valid_mask.shape != q.shape[:2]:
        raise ValueError("expected matching [B,K,V] distributions and a [B,K] mask")
    mask = valid_mask.bool()
    counts = mask.sum(-1)
    if bool((counts == 0).any()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("valid candidates must form a nonempty prefix in every block")
    pp, qq = p.detach()[mask].float(), q[mask].float()
    if not bool(torch.isfinite(pp).all() & torch.isfinite(qq).all()):
        raise ValueError("non-finite probability at a valid candidate")
    for probabilities in (pp, qq):
        if bool((probabilities < 0).any()) or not torch.allclose(
            probabilities.sum(-1),
            torch.ones_like(probabilities[:, 0]),
            atol=2e-4,
            rtol=0,
        ):
            raise ValueError("invalid sampling distribution")
    # TV only touches valid rows, excluding padded NaNs and avoiding extra
    # full-vocabulary masked copies. Rebuild just the small [B,K] acceptances.
    valid_acceptance = (1.0 - 0.5 * (pp - qq).abs().sum(-1)).clamp(0, 1)
    acceptance = torch.ones(mask.shape, device=q.device, dtype=torch.float32)
    acceptance = acceptance.masked_scatter(mask, valid_acceptance)
    return 1.0 - (acceptance.cumprod(-1) * mask).sum(-1) / counts


def candidate_mask(
    proposal: list[int], remaining: int, stop_ids: set[int]
) -> list[bool]:
    """Include the first EOS, stop after it, and ignore actual rejection length."""
    live = True
    mask = []
    for index, token in enumerate(proposal):
        mask.append(live and index < remaining)
        if token in stop_ids:
            live = False
    return mask
