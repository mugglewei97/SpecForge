"""Paired teacher-forced diagnostics, NOT live verifier acceptance metrics."""

import torch


@torch.no_grad()
def predecessor_pair_metrics(reference_logits, candidate_logits, target_logits, valid):
    """Compare heads on identical prefixes, returning counts plus local rates.

    reference_logits may be the current head with the second branch disabled,
    or a separate reference checkpoint evaluated on the SAME stored prefixes.
    Greedy regression uses target argmax, never the sampled dataset label.
    Stochastic diagnostics use full-vocabulary T=1 overlap (no top-k/top-p).
    """
    if not (reference_logits.shape == candidate_logits.shape == target_logits.shape):
        raise ValueError("Paired logits must have identical shapes")
    if valid.shape != target_logits.shape[:-1]:
        raise ValueError("valid must match the logits' non-vocabulary axes")
    old_correct = reference_logits.argmax(-1).eq(target_logits.argmax(-1))
    new_correct = candidate_logits.argmax(-1).eq(target_logits.argmax(-1))
    p = target_logits.float().softmax(-1)
    old_overlap = torch.minimum(p, reference_logits.float().softmax(-1)).sum(-1)
    new_overlap = torch.minimum(p, candidate_logits.float().softmax(-1)).sum(-1)
    metrics = {}
    # Include first position as an identity-path sanity check.
    for d in range(min(4, valid.shape[-1])):
        mask = valid[..., d].bool()
        correct = mask & old_correct[..., d]
        incorrect = mask & ~old_correct[..., d]
        regressions = (correct & ~new_correct[..., d]).float().sum()
        repairs = (incorrect & new_correct[..., d]).float().sum()
        count = mask.float().sum()
        prefix = f"carh_pair_tf_d{d + 1}_"
        metrics.update({
            prefix + "valid_count": count,
            prefix + "reference_correct_count": correct.float().sum(),
            prefix + "reference_incorrect_count": incorrect.float().sum(),
            prefix + "regression_count": regressions,
            prefix + "repair_count": repairs,
            prefix + "regression_rate": regressions / correct.float().sum().clamp_min(1),
            prefix + "repair_rate": repairs / incorrect.float().sum().clamp_min(1),
            prefix + "overlap_t1_delta_sum": (
                (new_overlap[..., d] - old_overlap[..., d]) * mask
            ).sum(),
            prefix + "overlap_t1_delta": (
                (new_overlap[..., d] - old_overlap[..., d]) * mask
            ).sum() / count.clamp_min(1),
        })
    return metrics
