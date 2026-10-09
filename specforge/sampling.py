"""Sampling transforms shared by training-time draft and target distributions."""

from __future__ import annotations

import torch


def processed_log_probs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    *,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Apply deployment-style temperature/Top-k/Top-p and normalize.

    ``temperatures`` is per batch row; all remaining non-vocabulary dimensions
    share that row's sampling configuration.  Temperature zero is represented
    as a one-hot greedy distribution so callers can mix greedy and stochastic
    trajectories without a separate full-vocabulary tensor.
    """
    if logits.ndim < 2:
        raise ValueError("logits must have a batch and vocabulary dimension")
    if temperatures.ndim != 1 or temperatures.size(0) != logits.size(0):
        raise ValueError("temperatures must have shape [batch]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    scores = logits.float()
    temperature_shape = (temperatures.size(0),) + (1,) * (scores.ndim - 1)
    stochastic = temperatures.gt(0)
    safe_temperatures = torch.where(
        stochastic, temperatures, torch.ones_like(temperatures)
    ).view(temperature_shape)
    scores = scores / safe_temperatures

    if top_k > 0 and top_k < scores.size(-1):
        topk_values, topk_indices = torch.topk(scores, k=top_k, dim=-1)
        filtered = torch.full_like(scores, float("-inf"))
        scores = filtered.scatter(-1, topk_indices, topk_values)

    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(
            scores, dim=-1, descending=True
        )
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs >= float(top_p)
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(scores, float("-inf"))
        scores = filtered.scatter(-1, sorted_indices, sorted_scores)

    log_probs = torch.log_softmax(scores, dim=-1)
    if not stochastic.all():
        greedy_ids = logits.float().argmax(dim=-1, keepdim=True)
        greedy_log_probs = torch.full_like(log_probs, float("-inf"))
        greedy_log_probs.scatter_(-1, greedy_ids, 0.0)
        row_shape = (stochastic.size(0),) + (1,) * (log_probs.ndim - 1)
        log_probs = torch.where(stochastic.view(row_shape), log_probs, greedy_log_probs)
    return log_probs
