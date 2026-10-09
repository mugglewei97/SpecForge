"""Training-only predictive supervision; no serving modules or vocabulary logits.

The semantic codebook is a chunked, randomized PCA-whitened projection of
normalized frozen target LM-head rows. This is an approximation, not an exact
PCA or a reproduction of another paper's full training recipe.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from specforge.legacy.dspark_training.options import add_predictive_auxiliary_args as add_predictive_auxiliary_args




def validate_predictive_auxiliary(args, draft_model):
    for key in ("conv_source_semantic_alpha", "carh_reference_calibration_alpha"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    warmup, ramp = args.predictive_aux_warmup_ratio, args.predictive_aux_ramp_ratio
    if not (0 <= warmup <= 1 and 0 <= ramp <= 1 and warmup + ramp <= 1):
        raise ValueError("predictive auxiliary warmup/ramp must fit within training")
    if args.conv_source_semantic_alpha > 0:
        ranks = set()
        for layer in draft_model.layers:
            conv = getattr(layer, "mlp_conv", None)
            if conv is None or getattr(conv, "source_down", None) is None:
                raise ValueError("Semantic supervision requires source-aware MLP conv in every draft layer")
            ranks.add(conv.source_rank)
        if len(ranks) != 1:
            raise ValueError("Semantic supervision requires one shared source rank")
    if args.carh_reference_calibration_alpha > 0:
        head = draft_model.markov_head
        if not getattr(head, "predecessor_context_mode", "none").startswith("innovation"):
            raise ValueError("Reference calibration requires an innovation CARH head")
        if getattr(head, "predecessor_count", 1) != 1:
            raise ValueError("Reference calibration requires single-predecessor CARH")
        if getattr(draft_model, "recall_correction", None) is not None:
            raise ValueError("Reference calibration v1 requires unmodified backbone CARH inputs (no recall correction)")


def validate_predictive_resume(saved_args, args):
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    defaults = dict(conv_source_semantic_alpha=0.0, carh_reference_calibration_alpha=0.0,
                    predictive_aux_warmup_ratio=0.10, predictive_aux_ramp_ratio=0.15,
                    source_semantic_codebook_seed=20261001)
    active = any(saved_args.get(key, 0) or getattr(args, key) for key in
                 ("conv_source_semantic_alpha", "carh_reference_calibration_alpha"))
    if active:
        for key, default in defaults.items():
            if saved_args.get(key, default) != getattr(args, key):
                raise ValueError(f"Predictive auxiliary resume setting changed: {key}")


def auxiliary_scale(step, total_steps, warmup, ramp):
    if total_steps <= 0:
        return 0.0
    progress = step / total_steps
    if progress < warmup:
        return 0.0
    return 1.0 if ramp == 0 else min(max((progress - warmup) / ramp, 0.0), 1.0)


def predecessor_source_weights(eval_mask, decay_weights):
    """Source slot j predicts label j, and is consumed by receiver slot j+1.

    No cross-block shift; omit the unused final source and invalid receivers.
    Weight by receiving depth. Slot zero's label is anchor+1, not the anchor.
    """
    weights = eval_mask[..., :-1] * eval_mask[..., 1:] * decay_weights[..., 1:]
    return F.pad(weights, (0, 1))


def reference_calibration_terms(head, hidden, target_ids, source_weights):
    """Calibrate the same reference used by the next slot's innovation.

    Ground-truth predecessor embeddings are detached labels, never forward
    inputs to the backbone. Only receivers 1..K-1 have a reference.
    """
    prediction = head.predecessor_context_proj(head.hidden_proj(hidden[..., :-1, :])).float()
    target = head.get_prev_embeddings(target_ids[..., :-1]).detach().float()
    weight = source_weights[..., :-1].float()
    cosine_error = 1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-6)
    # Relative MSE calibrates scale as well as direction; floor avoids exploding
    # normalization for tiny embedding rows. No label-side gradient is allowed.
    energy = target.square().mean(-1).clamp_min(1e-4)
    relative_mse = (prediction - target).square().mean(-1) / energy
    cosine_num = (cosine_error * weight).sum()
    mse_num = (relative_mse * weight).sum()
    norm_ratio = prediction.norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-6)
    return 0.5 * (cosine_num + mse_num), cosine_num, mse_num, (norm_ratio * weight).sum()


@torch.no_grad()
def build_semantic_codebook(weight, rank, seed, chunk_size=2048):
    """Bounded CPU working set; never allocate [vocab, hidden] in FP32.

    Two subspace iterations with 16 oversampling dimensions. Dedicated CPU
    generator preserves the training RNG stream. Codes are fixed unit vectors.
    """
    vocab, hidden = weight.shape
    if not 0 < rank < min(vocab, hidden):
        raise ValueError("Codebook rank must be smaller than vocabulary and hidden width")
    width = min(hidden, rank + 16)

    def rows():
        for start in range(0, vocab, chunk_size):
            row = weight[start:start + chunk_size].detach().to(device="cpu", dtype=torch.float32)
            yield start, F.normalize(row, dim=-1, eps=1e-8)

    mean = torch.zeros(hidden)
    for _, row in rows():
        mean += row.sum(0) / vocab
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.linalg.qr(torch.randn(hidden, width, generator=generator), mode="reduced").Q
    for _ in range(2):
        sketch = torch.zeros_like(q)
        for _, row in rows():
            centered = row - mean
            sketch.add_(centered.T @ (centered @ q) / vocab)
        q = torch.linalg.qr(sketch, mode="reduced").Q
    covariance = torch.zeros(width, width)
    for _, row in rows():
        projected = (row - mean) @ q
        covariance.add_(projected.T @ projected / vocab)
    values, vectors = torch.linalg.eigh(covariance)
    values, vectors = values[-rank:].flip(0), vectors[:, -rank:].flip(1)
    basis = q @ vectors
    # Fix the arbitrary eigenvector signs for reproducibility on one backend.
    pivot = basis.abs().argmax(0)
    sign = basis[pivot, torch.arange(rank)].sign()
    basis *= torch.where(sign == 0, torch.ones_like(sign), sign)
    scale = values.clamp_min(max(float(values.max()) * 1e-5, 1e-8)).rsqrt()
    codes = torch.empty(vocab, rank, dtype=torch.float32)
    for start, row in rows():
        codes[start:start + len(row)] = F.normalize(((row - mean) @ basis) * scale, dim=-1)
    if not torch.isfinite(codes).all():
        raise ValueError("Non-finite semantic codebook")
    return codes


def prepare_semantic_codebook(args, weight, rank):
    """Build once on rank zero, save a training artifact, broadcast fixed codes.

    The codebook is not exported in draft weights and is not used by SGLang.
    Resume requires the original artifact rather than silently rebuilding it.
    """
    distributed = dist.is_available() and dist.is_initialized()
    rank_zero = not distributed or dist.get_rank() == 0
    path = Path(args.output_dir) / "source_semantic_codebook.pt"
    metadata = dict(algorithm="normalized-centered-randomized-pca-whiten-v1", rank=rank,
                    shape=list(weight.shape), seed=args.source_semantic_codebook_seed,
                    target_model_path=str(args.target_model_path))
    status, codes = [None], None
    if rank_zero:
        try:
            if path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=True)
                if artifact["metadata"] != metadata:
                    raise ValueError(f"Codebook provenance differs: {path}")
                codes = artifact["codes"]
            else:
                if args.resume:
                    raise ValueError(f"Resume needs the original codebook: {path}")
                codes = build_semantic_codebook(weight, rank, args.source_semantic_codebook_seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = path.with_suffix(".tmp")
                torch.save(dict(codes=codes, metadata=metadata), temporary_path)
                temporary_path.replace(path)
            if codes.shape != (weight.shape[0], rank) or not torch.isfinite(codes).all():
                raise ValueError("Invalid saved semantic codebook")
        except Exception as error:
            status[0] = str(error)
    if distributed:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise ValueError(status[0])
    codes = codes.to(device=weight.device, dtype=torch.float32) if rank_zero else torch.empty(
        weight.shape[0], rank, device=weight.device, dtype=torch.float32)
    if distributed:
        dist.broadcast(codes, src=0)
    return codes.detach()
