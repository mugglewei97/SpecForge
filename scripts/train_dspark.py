#!/usr/bin/env python3
# coding=utf-8
"""DSpark Training Script.

DSpark = DFlash block-diffusion drafter + EAGLE-style Markov & confidence heads,
trained with cross-entropy + L1 distribution distillation + confidence BCE, or
with an optional acceptance objective. ``--dspark-loss-type tv-acceptance``
samples candidate tokens, re-verifies their actual prefixes, and differentiates
the normalized TV accepted-length proxy with those samples held fixed. The
distributional objectives need the target model's FINAL hidden state, so the
target backend must surface it (HF always does; sglang does when it returns both
the captured aux stream and the final hidden state).

Cloned from ``scripts/train_dflash.py`` and adapted: builds a DSparkDraftModel +
OnlineDSparkModel, plumbs ``last_hidden_states`` into the forward, and logs the
per-component (ce / l1 / confidence) losses.
"""

import argparse
import contextlib
import functools
import json
import logging
import math
import os
import shutil
import time
import unicodedata
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import concatenate_datasets, load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dspark import OnlineDSparkModel
from specforge.predictive_auxiliary import (
    add_predictive_auxiliary_args, validate_predictive_auxiliary,
    validate_predictive_resume, prepare_semantic_codebook,
)
from specforge.bv_loss import add_bv_args, validate_bv_args, validate_bv_resume
from specforge.core.multi_teacher_oracle import (
    MultiTeacherOracleCache,
    MultiTeacherTrajectoryWriter,
)
from specforge.data.preprocessing import build_eagle3_dataset
from specforge.legacy.data_utils import (
    build_source_mixture_weights,
    prepare_dp_dataloaders,
)
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.inference.target_engine.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.legacy.dspark import DSparkDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.legacy.optimizer import (
    build_bf16_optimizer, save_fsdp_optimizer_state, load_fsdp_optimizer_state,
    validate_optimizer_resume,
)
from specforge.muon import (
    add_muon_optimizer_args, validate_muon_options, configure_fsdp_optimizer_precision,
)
from specforge.microbatch import forward_backward_microbatches, validate_microbatch_resume
from specforge.netprefix import (
    add_netprefix_args, validate_netprefix_args, GreedyPrefixHook, NetPrefixRuntime,
    rng_state, restore_rng,
)
from specforge.netprefix_stream import TrainingStreamSource
from specforge.fixed_prefix_reference import (
    add_reference_args, validate_reference_args, FixedPrefixReference,
)
from specforge.tracker import create_tracker
from specforge.legacy.checkpoint import get_last_checkpoint
from specforge.utils import (
    get_local_device,
    print_on_rank0,
    print_with_rank,
)


def _comma_separated_floats(value: str) -> Tuple[float, ...]:
    """Parse a non-empty comma-separated float schedule for argparse."""
    try:
        parsed = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated floats, got {value!r}"
        ) from error
    if not parsed:
        raise argparse.ArgumentTypeError("temperature schedule must not be empty")
    return parsed


def _comma_separated_strings(value: str) -> Tuple[str, ...]:
    """Parse a non-empty comma-separated label list for argparse."""
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("label list must not be empty")
    return parsed


def _build_dialogue_token_class_lookup(tokenizer) -> torch.Tensor:
    """Classify vocabulary entries for cheap Chinese/English boundary routing.

    Classes are other/Han/Latin/number/punctuation/special.  The lookup is a
    non-persistent uint8 buffer, so it adds neither trainable parameters nor
    checkpoint size.
    """
    vocab_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)
    classes = torch.zeros(vocab_size, dtype=torch.uint8)
    # Decoding avoids misclassifying byte-level BPE spellings of Han tokens as
    # Latin. Chunking keeps the temporary singleton lists bounded.
    for start in range(0, vocab_size, 4096):
        stop = min(start + 4096, vocab_size)
        texts = tokenizer.batch_decode(
            [[token_id] for token_id in range(start, stop)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id, decoded in zip(range(start, stop), texts):
            if token_id in special_ids:
                classes[token_id] = 5
                continue
            text = decoded.strip()
            if not text:
                classes[token_id] = 4
                continue
            if any("\u4e00" <= char <= "\u9fff" for char in text):
                classes[token_id] = 1
            elif any(
                ("a" <= char.lower() <= "z")
                or "LATIN" in unicodedata.name(char, "")
                for char in text
            ):
                classes[token_id] = 2
            elif any(char.isdigit() for char in text):
                classes[token_id] = 3
            elif all(
                unicodedata.category(char)[0] in {"P", "S", "Z"}
                for char in text
            ):
                classes[token_id] = 4
    return classes


def _apply_trainable_parameter_scope(
    draft_model: DSparkDraftModel, scope: str
) -> Tuple[int, int]:
    """Apply a named training scope before FSDP and optimizer construction."""
    if scope == "all":
        trainable = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        return trainable, 0
    if scope == "selector-only":
        if getattr(draft_model, "candidate_selector", None) is None:
            raise ValueError(
                "selector-only scope requires --selector-rank and --selector-top-k"
            )
        prefix = "candidate_selector."
    elif scope == "carh-only":
        if getattr(draft_model, "markov_head", None) is None:
            raise ValueError(
                "carh-only scope requires an initialized Markov/CARH head"
            )
        prefix = "markov_head."
    else:
        raise ValueError(f"unsupported trainable parameter scope: {scope}")

    trainable = 0
    frozen = 0
    for name, parameter in draft_model.named_parameters():
        enabled = name.startswith(prefix)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if trainable == 0:
        raise RuntimeError(f"{scope} scope selected no trainable parameters")
    return trainable, frozen


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSpark Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
        help="Backend for target model: 'sglang' (service) or 'hf' (local). "
        "DSpark's L1/confidence losses need the target's final hidden state; "
        "the 'hf' backend always surfaces it.",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument(
        "--dspark-loss-type", choices=["ce-l1", "tv-acceptance"], default="ce-l1",
        help="tv-acceptance: pure normalized accepted-length proxy on draft-sampled prefixes.",
    )
    model_group.add_argument("--tv-sampling-temperature", type=float, default=1.0)
    model_group.add_argument("--tv-objective-chunk-blocks", type=int, default=8)
    model_group.add_argument("--tv-verification-batch-size", type=int, default=4)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, auto-detect from tokenizer.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model.",
    )
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=4.0,
        help="Gamma for exponential within-block loss decay (exp(-k/gamma), "
        "k = within-block slot index). None disables.",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Embedding weight key in the target model. "
        "Default: 'model.embed_tokens.weight' for standard models.",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default=None,
        help="LM head weight key in the target model. Default: 'lm_head.weight'.",
    )

    # DSpark-specific knobs
    dspark_group = parser.add_argument_group("dspark")
    dspark_group.add_argument(
        "--markov-rank",
        type=int,
        default=256,
        help="Rank of the low-rank Markov (bigram) bias head. 0 disables it.",
    )
    dspark_group.add_argument(
        "--markov-head-type",
        type=str,
        default=None,
        choices=["vanilla", "gated", "rnn", "carh"],
        help="Causal head architecture. Omit to preserve the draft config value.",
    )
    dspark_group.add_argument(
        "--carh-gate-bias",
        type=float,
        default=0.0,
        help="Initial scalar-gate bias for the context-aware causal residual head.",
    )
    dspark_group.add_argument(
        "--carh-predecessor-count", type=int, choices=[1, 2, 3], default=None,
        help="CARH token window; extra lag-2/lag-3 projections are zero-initialized.",
    )
    dspark_group.add_argument(
        "--carh-predecessor-context-mode",
        choices=[
            "none", "innovation", "context", "innovation-position",
            "innovation-residual", "innovation-residual-position",
            "innovation-residual-norm-gate",
        ],
        default=None,
        help=(
            "Use the previous draft slot's rank-space backbone state to gate "
            "single-predecessor CARH; innovation contrasts it with the actual "
            "predecessor embedding, while context is a matched control."
        ),
    )
    dspark_group.add_argument(
        "--carh-sampled-prefix-memory-rank", type=int, default=None,
        help="Block-local sampled-prefix memory width for pred1 CARH (0 disables).",
    )
    dspark_group.add_argument(
        "--carh-sampled-prefix-memory-start-ratio", type=float, default=0.25,
        help="Fraction of optimizer steps before the sampled-prefix branch activates.",
    )
    dspark_group.add_argument(
        "--carh-sampled-prefix-memory-ramp-ratio", type=float, default=0.10,
        help="Fraction of optimizer steps used to ramp memory output from 0 to 1.",
    )
    dspark_group.add_argument(
        "--carh-predecessor-diagnostics-interval", type=int, default=0,
        help="Optimizer-step interval for offline branch-on/off diagnostics (0 disables).",
    )
    dspark_group.add_argument(
        "--parallel-refiner-rank",
        type=int,
        default=None,
        help="Low-rank width of Hazard-Adaptive Parallel Refinement; 0 disables.",
    )
    dspark_group.add_argument(
        "--parallel-refiner-steps",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Fixed number of HAPR refinement rounds used in training/serving.",
    )
    dspark_group.add_argument(
        "--parallel-refiner-gate-bias", type=float, default=None
    )
    dspark_group.add_argument(
        "--parallel-refiner-second-step-bias", type=float, default=None
    )
    dspark_group.add_argument(
        "--parallel-refiner-residual-scale", type=float, default=None
    )
    dspark_group.add_argument(
        "--conv-kernel-size",
        type=int,
        default=None,
        help=(
            "Enable DFlash2 causal dynamic convolution with this many taps. "
            "Set together with --conv-group-size; omit both to preserve config."
        ),
    )
    dspark_group.add_argument(
        "--local-transition-rank",
        type=int,
        default=None,
        help="Low-rank width of verified-anchor Local Transition Attention; 0 disables.",
    )
    dspark_group.add_argument("--local-transition-heads", type=int, default=None)
    dspark_group.add_argument(
        "--local-transition-window",
        type=int,
        default=None,
        help="Number of causal predecessor slots in addition to anchor and self.",
    )
    dspark_group.add_argument(
        "--local-transition-residual-scale", type=float, default=None
    )
    dspark_group.add_argument("--local-transition-gate-bias", type=float, default=None)
    dspark_group.add_argument(
        "--local-transition-conv-decay-start-ratio", type=float, default=None
    )
    dspark_group.add_argument(
        "--local-transition-conv-decay-end-ratio", type=float, default=None
    )
    dspark_group.add_argument(
        "--local-transition-final-conv-scale", type=float, default=None
    )
    dspark_group.add_argument(
        "--freeze-dynamic-conv-during-transition",
        action="store_true",
        help="Keep the warm-start DynamicConv fixed while its contribution decays.",
    )
    dspark_group.add_argument(
        "--prefix-state-mixer-mode",
        choices=["none", "basic", "survival-conditioned"],
        default=None,
        help="Replace local DynamicConv with a strictly causal prefix-state mixer.",
    )
    dspark_group.add_argument("--prefix-state-rank", type=int, default=None)
    dspark_group.add_argument("--block-summary-rank", type=int, default=None,
                              help="Optional anchor-hidden residual width after the draft backbone; 0 disables.")
    dspark_group.add_argument(
        "--block-summary-source", choices=["anchor", "position"], default=None,
        help="Source for the matched low-rank residual: block anchor or each position's own hidden state.",
    )
    dspark_group.add_argument(
        "--block-summary-gate-mode", choices=["position", "compatibility"], default=None,
        help="Position-only gate or anchor/current compatibility gate for the block summary.",
    )
    dspark_group.add_argument(
        "--block-summary-counterfactual-alpha", type=float, default=0.0,
        help="Training-only first-error repair and surviving-prefix protection proxy weight.",
    )
    dspark_group.add_argument(
        "--prefix-state-retention-bias", type=float, default=None
    )
    dspark_group.add_argument("--prefix-state-update-bias", type=float, default=None)
    dspark_group.add_argument("--prefix-state-gate-bias", type=float, default=None)
    dspark_group.add_argument(
        "--prefix-state-residual-scale", type=float, default=None
    )
    dspark_group.add_argument(
        "--conv-group-size",
        type=int,
        default=None,
        help="Channels sharing each dynamic convolution coefficient.",
    )
    dspark_group.add_argument(
        "--no-conv-identity-init",
        dest="conv_identity_init",
        action="store_false",
        default=True,
        help="Do not zero dynamic-kernel deltas after model initialization.",
    )
    dspark_group.add_argument(
        "--conv-mode",
        choices=["legacy", "survival-gated", "neighbor-bounded", "prefix-risk-bounded"],
        default=None,
        help="Dynamic-convolution parameterization; omit to preserve config.",
    )
    dspark_group.add_argument(
        "--conv-kernel-conditioning",
        choices=["input", "output", "source-aware", "grouped16", "grouped64", "low-rank64", "static"],
        default=None,
        help="Legacy kernel generator: input/output/source-aware; grouped16/grouped64 "
        "use local channel groups, low-rank64 uses a rank-64 factorization, "
        "static removes dynamic deltas. Convolution group size is unchanged.",
    )
    dspark_group.add_argument(
        "--conv-source-rank",
        type=int,
        default=None,
        help="Rank of the source-aware kernel generator (default 32).",
    )
    dspark_group.add_argument(
        "--conv-apply-to",
        choices=["attention-output", "attention", "attention-mlp"],
        default=None,
        help="Place convolution after attention only, around attention, or around attention+MLP.",
    )
    dspark_group.add_argument(
        "--conv-last-n-layers",
        type=int,
        default=None,
        help="Apply convolution only to the last N draft layers; 0 means all layers.",
    )
    dspark_group.add_argument(
        "--conv-residual-scale",
        type=float,
        default=None,
        help="Maximum residual scale used by survival-gated convolution.",
    )
    dspark_group.add_argument(
        "--conv-gate-bias",
        type=float,
        default=None,
        help="Initial logit bias of the survival gate.",
    )
    dspark_group.add_argument(
        "--conv-freeze-identity",
        action="store_true",
        default=None,
        help="Keep the convolution self tap fixed to the exact identity.",
    )
    dspark_group.add_argument(
        "--conv-gate-loss-alpha",
        type=float,
        default=0.0,
        help="BCE weight supervising convolution gates with rollout prefix survival.",
    )
    dspark_group.add_argument(
        "--enable-confidence-head",
        action="store_true",
        default=True,
        help="Enable the per-position accept-rate (confidence) head.",
    )
    dspark_group.add_argument(
        "--no-confidence-head",
        dest="enable_confidence_head",
        action="store_false",
        help="Disable the confidence head.",
    )
    dspark_group.add_argument(
        "--confidence-head-with-markov",
        action="store_true",
        default=True,
        help="Fuse the Markov prev-token embedding into the confidence features.",
    )
    dspark_group.add_argument(
        "--ce-loss-alpha", type=float, default=0.1, help="Weight on cross-entropy."
    )
    dspark_group.add_argument(
        "--l1-loss-alpha",
        type=float,
        default=0.9,
        help="Weight on L1 distribution distillation (needs target last hidden).",
    )
    dspark_group.add_argument(
        "--offline-acceptance-objective",
        choices=["none", "angel-lk-e2e"],
        default="none",
        help=(
            "Teacher-forced acceptance-aligned objective. angel-lk-e2e uses "
            "D-PACE-weighted hybrid LK cold start, then switches to an "
            "end-to-end expected accepted-length surrogate without rollout."
        ),
    )
    dspark_group.add_argument(
        "--offline-acceptance-greedy-weight", type=float, default=0.35
    )
    dspark_group.add_argument(
        "--offline-acceptance-stochastic-weight", type=float, default=0.65
    )
    dspark_group.add_argument(
        "--offline-acceptance-temperature",
        type=float,
        default=1.0,
        help="Positive serving temperature used to compute exact p/q overlap.",
    )
    dspark_group.add_argument(
        "--offline-acceptance-lk-eta",
        type=float,
        default=1.0,
        help="Acceptance-dependent KL decay in the hybrid LK cold start.",
    )
    dspark_group.add_argument(
        "--offline-acceptance-dpace-rho",
        type=float,
        default=0.5,
        help="Confidence smoothing floor in AngelSpec D-PACE weights.",
    )
    dspark_group.add_argument(
        "--offline-acceptance-cold-start-ratio", type=float, default=0.20
    )
    dspark_group.add_argument(
        "--offline-acceptance-transition-ratio", type=float, default=0.10
    )
    dspark_group.add_argument(
        "--confidence-head-alpha",
        type=float,
        default=1.0,
        help="Weight on the confidence-head BCE (needs target last hidden).",
    )
    dspark_group.add_argument(
        "--confidence-target-mode",
        choices=["l1-overlap", "on-policy-survival"],
        default="l1-overlap",
        help="Train confidence from L1 overlap or actual verifier prefix survival.",
    )
    dspark_group.add_argument(
        "--confidence-detach-backbone",
        action="store_true",
        help="Detach confidence features so calibration cannot move the drafter.",
    )
    dspark_group.add_argument("--recall-correction-rank", type=int, default=None)
    dspark_group.add_argument(
        "--recall-correction-gate-bias", type=float, default=None
    )
    dspark_group.add_argument(
        "--recall-correction-alpha", type=float, default=0.0
    )
    dspark_group.add_argument("--recall-partition-top-k", type=int, default=16)
    dspark_group.add_argument(
        "--recall-correction-loss-budget", type=float, default=0.08
    )
    dspark_group.add_argument(
        "--target-layer-fusion-mode",
        choices=["none", "per-draft-layer", "shared-low-rank", "shared-low-rank-delta", "static-channel", "shared-low-rank-grouped", "shared-low-rank-token"],
        default=None,
    )
    dspark_group.add_argument(
        "--target-layer-fusion-residual-scale", type=float, default=None
    )
    dspark_group.add_argument("--target-layer-fusion-rank", type=int, default=None)
    dspark_group.add_argument("--target-layer-fusion-source-dropout", type=float, default=None)
    dspark_group.add_argument(
        "--selector-rank",
        type=int,
        default=None,
        help="Rank of the CARH-conditioned DFlash2 path selector.",
    )
    dspark_group.add_argument(
        "--selector-top-k",
        type=int,
        default=None,
        help="Number of corrected-logit candidates considered at each depth.",
    )
    dspark_group.add_argument(
        "--selector-loss-alpha",
        type=float,
        default=0.0,
        help="Weight of the candidate-path selector objective.",
    )
    dspark_group.add_argument(
        "--selector-rollout-ratio",
        type=float,
        default=0.0,
        help="Mixture weight on self-conditioned selector path credit in [0,1].",
    )
    dspark_group.add_argument(
        "--selector-training-mode",
        choices=["legacy", "recoverability-aware", "survival-distill"],
        default="legacy",
        help=(
            "legacy reproduces the DFlash2-style gold-injected objective; "
            "recoverability-aware uses the strict serving Top-k, trains only "
            "recoverable errors, and protects unary-correct decisions; "
            "survival-distill uses strict serving Top-k, trains only on "
            "prefix-reachable ranking errors, and distills teacher gains to CARH."
        ),
    )
    dspark_group.add_argument(
        "--selector-distill-alpha",
        type=float,
        default=0.0,
        help="Weight of advantage-gated selector-to-CARH rank distillation.",
    )
    dspark_group.add_argument(
        "--selector-preservation-alpha",
        type=float,
        default=0.0,
        help="Weight preventing the selector from flipping unary Top-1-correct states.",
    )
    dspark_group.add_argument(
        "--selector-distill-temperature",
        type=float,
        default=1.0,
        help="Candidate-set temperature shared by selector teacher and CARH student.",
    )
    dspark_group.add_argument(
        "--selector-advantage-margin",
        type=float,
        default=0.0,
        help="Minimum teacher-minus-student gold probability required for distillation.",
    )
    dspark_group.add_argument(
        "--selector-teacher-warmup-ratio",
        type=float,
        default=0.10,
        help="Training fraction before activating the strict selector teacher.",
    )
    dspark_group.add_argument(
        "--selector-distill-start-ratio",
        type=float,
        default=0.30,
        help="Training fraction before transferring selector rankings into CARH.",
    )
    dspark_group.add_argument(
        "--selector-consolidation-ratio",
        type=float,
        default=0.15,
        help="Final training fraction with selector losses disabled.",
    )
    dspark_group.add_argument(
        "--selector-loss-budget",
        type=float,
        default=0.10,
        help="Maximum combined selector-teacher loss mass relative to main CE.",
    )
    dspark_group.add_argument(
        "--selector-runtime-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether serving should execute the selector. Use "
            "--no-selector-runtime-enabled for a training-only distilled teacher."
        ),
    )
    dspark_group.add_argument(
        "--selector-margin-threshold",
        type=float,
        default=None,
        help=(
            "Only apply the conditional candidate selector when the corrected "
            "unary Top-1/Top-2 logit margin is below this value. Zero keeps the "
            "legacy always-on selector."
        ),
    )
    dspark_group.add_argument(
        "--selector-tree-loss-alpha",
        type=float,
        default=0.0,
        help=(
            "Weight of the MACT branch-coverage surrogate. At reachable "
            "low-margin states it ranks the target token above the B-th "
            "strongest non-target candidate at the branch-width frontier."
        ),
    )
    dspark_group.add_argument(
        "--selector-tree-branch-width",
        type=int,
        default=2,
        help="Number of children retained for each low-margin MACT parent.",
    )
    dspark_group.add_argument(
        "--selector-tree-margin",
        type=float,
        default=0.2,
        help="Required target-score margin over the MACT branch frontier.",
    )
    dspark_group.add_argument(
        "--selector-tree-temperature",
        type=float,
        default=0.25,
        help="Softplus temperature for the MACT branch-coverage surrogate.",
    )
    dspark_group.add_argument(
        "--pace-mode",
        choices=["none", "token", "overlap", "hybrid", "rollout-residual"],
        default="none",
        help=(
            "Dynamic continuation-credit source. token uses target-token "
            "probability, overlap uses 1-TV(draft,target), and hybrid uses "
            "their geometric interpolation. rollout-residual performs a "
            "self-conditioned Markov rollout and adds bounded CE-only credit."
        ),
    )
    dspark_group.add_argument(
        "--pace-alpha",
        type=float,
        default=0.5,
        help="D-PACE probability smoothing floor in [0,1].",
    )
    dspark_group.add_argument(
        "--pace-hybrid-beta",
        type=float,
        default=0.5,
        help="Overlap contribution to hybrid PACE quality in [0,1].",
    )
    dspark_group.add_argument(
        "--pace-apply-to",
        choices=["ce", "ce-l1"],
        default="ce-l1",
        help="Apply dynamic PACE weights to CE only or to both CE and L1.",
    )
    dspark_group.add_argument(
        "--pace-blend-max",
        type=float,
        default=1.0,
        help=(
            "Maximum convex mixture coefficient on PACE weights. 1.0 is "
            "full replacement; values below 1 retain the fixed depth prior."
        ),
    )
    dspark_group.add_argument(
        "--pace-warmup-ratio",
        type=float,
        default=0.0,
        help="Training fraction kept on pure fixed DSpark weights.",
    )
    dspark_group.add_argument(
        "--pace-ramp-ratio",
        type=float,
        default=0.0,
        help="Training fraction used to ramp to --pace-blend-max.",
    )
    dspark_group.add_argument(
        "--pace-decay-start-ratio",
        type=float,
        default=1.0,
        help="Training fraction at which PACE begins cosine decay.",
    )
    dspark_group.add_argument(
        "--pace-blend-final",
        type=float,
        default=None,
        help="Final PACE coefficient after cosine decay (default: blend max).",
    )
    dspark_group.add_argument(
        "--pace-residual-beta",
        type=float,
        default=0.5,
        help="Log-credit advantage scale for rollout-residual PACE.",
    )
    dspark_group.add_argument(
        "--pace-residual-min", type=float, default=0.9,
        help="Minimum residual multiplier on the auxiliary CE weights.",
    )
    dspark_group.add_argument(
        "--pace-residual-max", type=float, default=1.1,
        help="Maximum residual multiplier on the auxiliary CE weights.",
    )
    dspark_group.add_argument(
        "--prefix-credit-mode",
        choices=["none", "full", "residual-only", "partial"],
        default="none",
        help=(
            "First-rejection auxiliary gradient destination. full updates the "
            "whole draft model; residual-only detaches backbone logits/hidden; "
            "partial scales the auxiliary gradient entering the backbone."
        ),
    )
    dspark_group.add_argument(
        "--prefix-credit-alpha",
        type=float,
        default=0.0,
        help="Weight of inference-rollout first-rejection credit.",
    )
    dspark_group.add_argument(
        "--prefix-credit-warmup-ratio", type=float, default=0.10
    )
    dspark_group.add_argument(
        "--prefix-credit-ramp-ratio", type=float, default=0.15
    )
    dspark_group.add_argument(
        "--prefix-credit-backbone-grad-scale",
        type=float,
        default=0.2,
        help="Backbone gradient multiplier used by partial prefix credit.",
    )
    dspark_group.add_argument("--shallow-frc-alpha", type=float, default=0.0)
    dspark_group.add_argument("--shallow-frc-max-depth", type=int, default=3)
    dspark_group.add_argument("--shallow-frc-margin", type=float, default=0.0)
    dspark_group.add_argument("--shallow-frc-temperature", type=float, default=1.0)
    dspark_group.add_argument("--dfap-alpha", type=float, default=0.0)
    dspark_group.add_argument("--dfap-min-depth", type=int, default=4)
    dspark_group.add_argument("--dfap-margin", type=float, default=0.5)
    dspark_group.add_argument("--dfap-temperature", type=float, default=1.0)
    dspark_group.add_argument(
        "--transition-credit-alpha",
        type=float,
        default=0.0,
        help="Weight of transition-aligned prefix credit (TAPC).",
    )
    dspark_group.add_argument(
        "--transition-credit-max-depth", type=int, default=3
    )
    dspark_group.add_argument(
        "--transition-credit-margin", type=float, default=0.0
    )
    dspark_group.add_argument(
        "--transition-credit-temperature", type=float, default=1.0
    )
    dspark_group.add_argument(
        "--transition-credit-loss-budget",
        type=float,
        default=0.10,
        help="Maximum TAPC loss mass relative to the detached main CE numerator.",
    )
    dspark_group.add_argument(
        "--transition2-margin-alpha",
        type=float,
        default=0.0,
        help=(
            "Weight of Transition-2 Conditional Margin repair. It updates the "
            "second draft decision only when the first rollout token survives."
        ),
    )
    dspark_group.add_argument(
        "--transition2-margin-floor", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--transition2-margin-temperature", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--transition2-margin-loss-budget",
        type=float,
        default=0.10,
        help="Maximum T2CM loss mass relative to main CE; 0 disables the cap.",
    )
    dspark_group.add_argument(
        "--prefix-bottleneck-alpha",
        type=float,
        default=0.0,
        help="Weight of the shallow-prefix normalized soft-min margin loss.",
    )
    dspark_group.add_argument(
        "--prefix-bottleneck-max-depth", type=int, default=3
    )
    dspark_group.add_argument(
        "--prefix-bottleneck-margin-floor", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--prefix-bottleneck-softmin-temperature", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--prefix-bottleneck-loss-temperature", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--prefix-bottleneck-loss-budget",
        type=float,
        default=0.10,
        help="Maximum PBM loss mass relative to main CE; 0 disables the cap.",
    )
    dspark_group.add_argument(
        "--deep-survival-guard-alpha",
        type=float,
        default=0.0,
        help=(
            "Weight of the low-margin guard on accepted deep-prefix tokens. "
            "Unlike DFAP, the block need not be fully accepted."
        ),
    )
    dspark_group.add_argument(
        "--deep-survival-guard-min-prefix", type=int, default=5
    )
    dspark_group.add_argument(
        "--deep-survival-guard-start-depth", type=int, default=4
    )
    dspark_group.add_argument(
        "--deep-survival-guard-margin-floor", type=float, default=0.3
    )
    dspark_group.add_argument(
        "--deep-survival-guard-temperature", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--deep-survival-guard-loss-budget",
        type=float,
        default=0.10,
        help="Maximum DSG loss mass relative to main CE; 0 disables the cap.",
    )
    dspark_group.add_argument(
        "--carh-recoverability-top-k",
        type=int,
        default=0,
        help=(
            "Strict unary Top-k used to partition CARH training into already "
            "correct, recoverable ranking error, and recall-miss states. "
            "Zero disables coverage-aware gradient routing."
        ),
    )
    dspark_group.add_argument(
        "--carh-recovery-loss-alpha",
        type=float,
        default=0.0,
        help="Weight of CARH's strict Top-k recoverable ranking CE.",
    )
    dspark_group.add_argument(
        "--carh-preservation-loss-alpha",
        type=float,
        default=0.0,
        help="Weight protecting unary Top-1-correct tokens from CARH overcorrection.",
    )
    dspark_group.add_argument(
        "--carh-noop-loss-alpha",
        type=float,
        default=0.0,
        help="Residual-energy penalty on unary Top-k recall misses.",
    )
    dspark_group.add_argument(
        "--carh-preservation-margin", type=float, default=0.0
    )
    dspark_group.add_argument(
        "--carh-preservation-temperature", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--carh-recoverability-loss-budget",
        type=float,
        default=0.10,
        help="Maximum combined CARH recoverability loss relative to main CE.",
    )
    dspark_group.add_argument(
        "--carh-gate-calibration-alpha",
        type=float,
        default=0.0,
        help=(
            "Weight for on-policy supervision of the existing CARH safety "
            "gate using whether the ungated residual improves target utility."
        ),
    )
    dspark_group.add_argument(
        "--carh-gate-noop-alpha",
        type=float,
        default=0.0,
        help="Penalty encouraging a closed CARH gate when correction is not useful.",
    )
    dspark_group.add_argument(
        "--carh-gate-margin-threshold",
        type=float,
        default=0.0,
        help="Minimum ungated target-utility gain labeled as a useful correction.",
    )
    dspark_group.add_argument(
        "--carh-gate-calibration-warmup-ratio",
        type=float,
        default=0.10,
        help="Training fraction before recovery-calibrated gate supervision ramps in.",
    )
    dspark_group.add_argument(
        "--state-credit-partition",
        choices=["overlap", "strict"],
        default="overlap",
        help=(
            "overlap lets Prefix-Full and shallow FRC both optimize shallow "
            "frontiers; strict assigns depths 1..FRC-max exclusively to FRC "
            "and later rejected frontiers to Prefix-Full."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-survival-alpha", type=float, default=0.0,
        help="Weight for target-scored accepted-prefix and first-rejection credit.",
    )
    dspark_group.add_argument(
        "--on-policy-full-alpha", type=float, default=0.0,
        help="Margin-floor protection weight for fully survived on-policy paths.",
    )
    dspark_group.add_argument(
        "--on-policy-first-rejection-max-depth", type=int, default=2,
    )
    dspark_group.add_argument(
        "--on-policy-boundary-max-depth",
        type=int,
        default=0,
        help=(
            "Restrict accepted-prefix and first-rejection OPSC credit to "
            "depths <= this value; 0 preserves legacy all-depth behavior."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-deep-alpha",
        type=float,
        default=0.0,
        help="Weight of Hi-Coverage accepted/full-path credit at deep depths.",
    )
    dspark_group.add_argument(
        "--on-policy-deep-start-depth",
        type=int,
        default=4,
        help="First one-indexed draft depth receiving deep Hi-Coverage credit.",
    )
    dspark_group.add_argument("--on-policy-margin-floor", type=float, default=0.3)
    dspark_group.add_argument("--on-policy-temperature", type=float, default=0.25)
    dspark_group.add_argument(
        "--on-policy-rollout-temperatures",
        type=_comma_separated_floats,
        default=(0.0,),
        help=(
            "Comma-separated proposal rollout temperatures. Temperature 0 is "
            "greedy; positive values sample CARH's corrected distribution."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-rollout-temperature-probs",
        type=_comma_separated_floats,
        default=(1.0,),
        help=(
            "Comma-separated categorical probabilities paired with "
            "--on-policy-rollout-temperatures. Values are normalized."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-anchor-sampling",
        choices=[
            "first",
            "uniform",
            "hazard",
            "hazard-flatness",
            "frontier-value",
            "dialogue-frontier",
        ],
        default="first",
        help=(
            "Select which valid block receives the extra target-scored rollout. "
            "frontier-value favors likely rejection with a confident target; "
            "dialogue-frontier additionally balances beginning/middle/end "
            "response phases and upweights linguistic boundaries."
        ),
    )
    dspark_group.add_argument("--on-policy-hazard-power", type=float, default=1.0)
    dspark_group.add_argument(
        "--on-policy-flatness-power-max", type=float, default=1.0
    )
    dspark_group.add_argument(
        "--on-policy-flatness-warmup-ratio", type=float, default=0.3
    )
    dspark_group.add_argument(
        "--on-policy-uniform-exploration", type=float, default=0.1
    )
    dspark_group.add_argument(
        "--dialogue-phase-masses",
        type=_comma_separated_floats,
        default=(0.15, 0.20, 0.30, 0.20, 0.15),
        help=(
            "Five sampling masses for single-turn response phases: opening, "
            "early, middle, late and closing."
        ),
    )
    dspark_group.add_argument(
        "--dialogue-boundary-boost",
        type=float,
        default=1.5,
        help="Multiplier for punctuation/special-token response boundaries.",
    )
    dspark_group.add_argument(
        "--dialogue-script-transition-boost",
        type=float,
        default=1.5,
        help="Multiplier for Han-to-Latin or Latin-to-Han transitions.",
    )
    dspark_group.add_argument(
        "--dialogue-occupancy-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Log selected/candidate phase occupancy and verifier survival by "
            "single-turn response phase and data source."
        ),
    )
    dspark_group.add_argument(
        "--dialogue-prefix-phase-weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mass-preserving five-phase weighting for teacher-forced "
            "Prefix-Full frontier credit."
        ),
    )
    dspark_group.add_argument(
        "--dialogue-t2cm-boundary-weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mass-preserving punctuation and Han/Latin boundary weighting "
            "for T2CM."
        ),
    )
    dspark_group.add_argument(
        "--dialogue-dsg-adaptive-margin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise the DSG margin floor at late-response and boundary anchors.",
    )
    dspark_group.add_argument(
        "--dialogue-dsg-late-margin-bonus", type=float, default=0.10
    )
    dspark_group.add_argument(
        "--dialogue-dsg-boundary-margin-bonus", type=float, default=0.05
    )
    dspark_group.add_argument(
        "--on-policy-distributional-top-k",
        type=int,
        default=0,
        help=(
            "Maximum target Top-k scored for distributional proposal overlap; "
            "0 disables. It remains the fixed K when adaptive mass is disabled."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-distributional-min-top-k",
        type=int,
        default=0,
        help="Minimum candidates retained by adaptive-mass OPSC; 0 disables it.",
    )
    dspark_group.add_argument(
        "--on-policy-distributional-mass-threshold",
        type=float,
        default=0.0,
        help=(
            "Target probability mass adaptive-mass OPSC tries to retain before "
            "the configured maximum Top-k; 0 preserves fixed Top-k."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-distributional-alpha",
        type=float,
        default=0.0,
        help="Shallow distributional prefix-survival weight.",
    )
    dspark_group.add_argument(
        "--on-policy-distributional-deep-alpha",
        type=float,
        default=0.0,
        help="Deep distributional prefix-survival preservation weight.",
    )
    dspark_group.add_argument(
        "--on-policy-distributional-cold-start-ratio",
        type=float,
        default=0.15,
        help="Fraction of training used to blend per-depth overlap into survival.",
    )
    dspark_group.add_argument(
        "--on-policy-rejection-aligned-alpha",
        type=float,
        default=0.0,
        help=(
            "Weight of stochastic p/q rejection-aligned overlap. Temperature "
            "zero keeps the existing hard SR-OPSC objective."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-rejection-aligned-survival-blend",
        type=float,
        default=0.5,
        help="Blend between -log overlap and occupancy-weighted soft survival.",
    )
    dspark_group.add_argument(
        "--on-policy-temperature-exclusive-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Route hard SR credit only to T=0 trajectories and rejection-aligned "
            "p/q credit only to T>0 trajectories instead of adding both at T>0."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-mixed-kl-alpha",
        type=float,
        default=0.0,
        help=(
            "Weight of the Draft-OPD-style mixed KL on DSpark on-policy states: "
            "forward KL on the accepted prefix and reverse KL on the rejected suffix."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-mixed-kl-accepted-weight", type=float, default=1.0
    )
    dspark_group.add_argument(
        "--on-policy-mixed-kl-rejected-weight", type=float, default=1.0
    )
    dspark_group.add_argument(
        "--on-policy-mixed-kl-rejection-decay",
        type=float,
        default=0.8,
        help="Rejected-suffix position weight gamma^(k-1), matching Draft-OPD.",
    )
    dspark_group.add_argument(
        "--on-policy-clipped-rkl-alpha",
        type=float,
        default=0.0,
        help=(
            "AdaFlash-style reverse-KL on verifier-reachable proposal states. "
            "Each state is robustly capped and can be weighted by rollout temperature."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-clipped-rkl-clip", type=float, default=0.01
    )
    dspark_group.add_argument(
        "--on-policy-clipped-rkl-temperature-weights",
        type=_comma_separated_floats,
        default=(1.0,),
        help="Weights paired with --on-policy-rollout-temperatures.",
    )
    dspark_group.add_argument(
        "--on-policy-target-distribution-temperature-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum temperature used only when target Top-k probabilities are "
            "scored; use 1.0 to expose a finite teacher margin for greedy rows."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-target-margin-alpha", type=float, default=0.0
    )
    dspark_group.add_argument(
        "--on-policy-target-margin-scale", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--on-policy-target-margin-offset", type=float, default=0.05
    )
    dspark_group.add_argument(
        "--on-policy-target-margin-min", type=float, default=0.05
    )
    dspark_group.add_argument(
        "--on-policy-target-margin-max", type=float, default=1.0
    )
    dspark_group.add_argument(
        "--on-policy-target-margin-max-depth", type=int, default=3
    )
    dspark_group.add_argument(
        "--on-policy-deployment-top-k",
        type=int,
        default=0,
        help="Top-k processor shared by stochastic rollout, target p, and draft q.",
    )
    dspark_group.add_argument(
        "--on-policy-deployment-top-p",
        type=float,
        default=1.0,
        help="Top-p processor shared by stochastic rollout, target p, and draft q.",
    )
    dspark_group.add_argument(
        "--on-policy-credit-partition",
        choices=["legacy", "strict-v2"],
        default="legacy",
        help=(
            "strict-v2 assigns real verifier states exclusively to shallow "
            "first-rejection repair, middle Prefix-Full CE, or deep survival "
            "preservation. legacy preserves the existing overlapping OPSC."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-middle-max-depth",
        type=int,
        default=4,
        help="Last one-indexed frontier depth owned by strict-v2 Prefix-Full.",
    )
    dspark_group.add_argument(
        "--on-policy-middle-alpha",
        type=float,
        default=0.0,
        help="Full-CE weight for strict-v2 middle first-rejection frontiers.",
    )
    dspark_group.add_argument(
        "--on-policy-temperature-hazard-credit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reweight strict survival credit with synchronized T/depth hazard EMA.",
    )
    dspark_group.add_argument(
        "--on-policy-hazard-ema-decay", type=float, default=0.99
    )
    dspark_group.add_argument(
        "--on-policy-hazard-weight-min", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--on-policy-hazard-weight-max", type=float, default=2.0
    )
    dspark_group.add_argument(
        "--on-policy-marginal-value-credit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mass-preserving reachability-times-continuation weighting for "
            "Tri-Alignment credit. Positive-temperature rows use verifier "
            "p/q acceptance when target Top-k probabilities are available."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-marginal-value-temperature", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--on-policy-marginal-value-weight-min", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--on-policy-marginal-value-weight-max", type=float, default=2.0
    )
    dspark_group.add_argument(
        "--on-policy-language-hazard-credit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mass-preserving hazard credit conditioned on response Han ratio, "
            "realized proposal-transition class and draft depth."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-language-han-threshold", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--on-policy-language-hazard-ema-decay", type=float, default=0.98
    )
    dspark_group.add_argument(
        "--on-policy-language-hazard-weight-min", type=float, default=0.75
    )
    dspark_group.add_argument(
        "--on-policy-language-hazard-weight-max", type=float, default=1.5
    )
    dspark_group.add_argument(
        "--on-policy-mix-ratio-max", type=float, default=0.5,
        help="Maximum scheduled mixture assigned to real proposal-prefix credit.",
    )
    dspark_group.add_argument("--on-policy-start-step", type=int, default=None,
                             help="Optional absolute optimizer-step start; independent of Prefix-Full warmup.")
    dspark_group.add_argument("--on-policy-ramp-steps", type=int, default=2000)
    dspark_group.add_argument(
        "--on-policy-interval", type=int, default=4,
        help="Run the extra frozen-target proposal scoring pass every N steps.",
    )
    dspark_group.add_argument(
        "--on-policy-loss-budget", type=float, default=0.10,
        help="Cap on-policy credit relative to the detached main CE numerator.",
    )
    dspark_group.add_argument(
        "--multi-teacher-oracle-export-dir",
        type=str,
        default=None,
        help="Export compact teacher trajectories instead of updating weights.",
    )
    dspark_group.add_argument("--multi-teacher-name", type=str, default="teacher")
    dspark_group.add_argument("--multi-teacher-export-top-k", type=int, default=32)
    dspark_group.add_argument("--multi-teacher-export-flush-records", type=int, default=4096)
    dspark_group.add_argument("--multi-teacher-oracle-cache", type=str, default=None)
    dspark_group.add_argument(
        "--multi-teacher-oracle-mode",
        choices=["none", "diagnostics", "distill"],
        default="none",
    )
    dspark_group.add_argument("--multi-teacher-oracle-alpha", type=float, default=0.005)
    dspark_group.add_argument(
        "--multi-teacher-oracle-advantage-threshold", type=float, default=0.0
    )
    dspark_group.add_argument("--multi-teacher-oracle-value-clip", type=float, default=7.0)
    dspark_group.add_argument("--multi-teacher-oracle-loss-budget", type=float, default=0.05)
    dspark_group.add_argument(
        "--on-policy-preservation-alpha",
        type=float,
        default=0.0,
        help="Margin-floor protection for every target-verified survived state.",
    )
    dspark_group.add_argument(
        "--on-policy-regression-alpha",
        type=float,
        default=0.0,
        help=(
            "Penalty when CARH lowers verifier-target margin below the unary "
            "draft backbone on a still-alive proposal prefix."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-regression-margin", type=float, default=0.0
    )
    dspark_group.add_argument(
        "--on-policy-greedy-preservation-alpha", type=float, default=0.0
    )
    dspark_group.add_argument(
        "--on-policy-greedy-preservation-margin-floor", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--on-policy-greedy-preservation-temperature", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--on-policy-reset-replay-alpha", type=float, default=0.0
    )
    dspark_group.add_argument(
        "--on-policy-reset-replay-horizon", type=int, default=3
    )
    dspark_group.add_argument(
        "--on-policy-reset-replay-max-rejection-depth", type=int, default=4
    )
    dspark_group.add_argument(
        "--on-policy-reset-replay-start-ratio", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--on-policy-reset-replay-ramp-ratio", type=float, default=0.15
    )
    dspark_group.add_argument(
        "--on-policy-reset-replay-loss-budget", type=float, default=0.05
    )
    dspark_group.add_argument(
        "--on-policy-reset-replay-advantage-threshold", type=float, default=0.0
    )
    dspark_group.add_argument(
        "--on-policy-reset-replay-value-clip", type=float, default=3.0
    )
    dspark_group.add_argument(
        "--on-policy-preference-alpha", type=float, default=0.0
    )
    dspark_group.add_argument(
        "--on-policy-preference-value-gap", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--on-policy-preference-temperature", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--on-policy-preference-loss-budget", type=float, default=0.03
    )
    dspark_group.add_argument(
        "--on-policy-pareto-credit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Dynamically upweight rollout temperatures with the largest "
            "EMA prefix-survival shortfall."
        ),
    )
    dspark_group.add_argument(
        "--on-policy-pareto-ema-decay", type=float, default=0.95
    )
    dspark_group.add_argument(
        "--on-policy-pareto-temperature", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--on-policy-pareto-weight-min", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--on-policy-pareto-weight-max", type=float, default=2.0
    )
    dspark_group.add_argument(
        "--parallel-refiner-hazard-alpha",
        type=float,
        default=0.0,
        help="BCE weight for real verifier first-rejection hazard prediction.",
    )
    dspark_group.add_argument(
        "--parallel-refiner-recovery-alpha",
        type=float,
        default=0.0,
        help="Margin-floor weight on still-alive CARH ranking errors.",
    )
    dspark_group.add_argument(
        "--parallel-refiner-preservation-alpha",
        type=float,
        default=0.0,
        help="Prevent HAPR from reducing already-correct CARH margins.",
    )
    dspark_group.add_argument(
        "--parallel-refiner-margin-floor", type=float, default=0.3
    )
    dspark_group.add_argument(
        "--parallel-refiner-temperature", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--parallel-refiner-loss-budget", type=float, default=0.05
    )
    dspark_group.add_argument(
        "--parallel-refiner-runtime-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Execute HAPR at serving time; disable for a distillation-only teacher.",
    )
    dspark_group.add_argument(
        "--parallel-refiner-teacher-path",
        type=str,
        default=None,
        help="Optional checkpoint from which only parallel_refiner weights are loaded.",
    )
    dspark_group.add_argument(
        "--freeze-parallel-refiner-teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze HAPR correction weights while leaving advantage heads trainable.",
    )
    dspark_group.add_argument(
        "--refiner-advantage-mode",
        choices=["none", "diagnostics", "distill"],
        default="none",
    )
    dspark_group.add_argument("--refiner-advantage-threshold", type=float, default=0.0)
    dspark_group.add_argument("--refiner-advantage-gate-alpha", type=float, default=0.005)
    dspark_group.add_argument("--refiner-advantage-regression-alpha", type=float, default=0.0025)
    dspark_group.add_argument("--refiner-advantage-distill-alpha", type=float, default=0.005)
    dspark_group.add_argument("--refiner-advantage-preservation-alpha", type=float, default=0.0025)
    dspark_group.add_argument(
        "--refiner-advantage-temperature-conditioned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Condition training-only advantage heads on rollout temperature.",
    )
    dspark_group.add_argument(
        "--refiner-advantage-greedy-preservation-alpha",
        type=float,
        default=0.0,
    )
    dspark_group.add_argument(
        "--refiner-advantage-greedy-margin-floor", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--refiner-advantage-gate-distill-min-probability",
        type=float,
        default=0.0,
    )
    dspark_group.add_argument("--refiner-advantage-distill-top-k", type=int, default=32)
    dspark_group.add_argument("--refiner-advantage-distill-temperature", type=float, default=1.0)
    dspark_group.add_argument("--refiner-advantage-value-clip", type=float, default=7.0)
    dspark_group.add_argument("--refiner-advantage-distill-start-ratio", type=float, default=0.20)
    dspark_group.add_argument("--refiner-advantage-consolidation-ratio", type=float, default=0.15)
    dspark_group.add_argument("--refiner-advantage-loss-budget", type=float, default=0.05)
    dspark_group.add_argument(
        "--vat-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the Verification-Aware Training matched baseline: "
            "teacher-forced verification simulation, first-rejection-anchored "
            "weights, hard+soft token losses, and a training-only survival head."
        ),
    )
    dspark_group.add_argument("--vat-hard-loss-alpha", type=float, default=1.0)
    dspark_group.add_argument("--vat-soft-loss-alpha", type=float, default=1.0)
    dspark_group.add_argument(
        "--vat-verification-head-alpha", type=float, default=1.0
    )
    dspark_group.add_argument(
        "--vat-post-rejection-decay-gamma", type=float, default=4.0
    )
    dspark_group.add_argument(
        "--vat-simulation-temperature",
        type=float,
        default=0.0,
        help="Zero uses greedy agreement; positive values simulate ratio acceptance.",
    )
    dspark_group.add_argument(
        "--branch-value-mode",
        choices=["none", "diagnostics", "distill"],
        default="none",
    )
    dspark_group.add_argument("--branch-value-top-m", type=int, default=4)
    dspark_group.add_argument("--branch-value-horizon", type=int, default=3)
    dspark_group.add_argument("--branch-value-alpha", type=float, default=0.0)
    dspark_group.add_argument(
        "--branch-value-temperature", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--branch-value-loss-budget", type=float, default=0.05
    )
    dspark_group.add_argument(
        "--branch-value-warmup-ratio", type=float, default=0.10
    )
    dspark_group.add_argument(
        "--branch-value-consolidation-ratio", type=float, default=0.15
    )
    dspark_group.add_argument(
        "--elastic-horizon-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train one fixed K_max graph with scheduled active K=short/long views.",
    )
    dspark_group.add_argument("--elastic-short-horizon", type=int, default=7)
    dspark_group.add_argument("--elastic-long-horizon", type=int, default=10)
    dspark_group.add_argument("--elastic-warmup-ratio", type=float, default=0.10)
    dspark_group.add_argument("--elastic-late-ratio", type=float, default=0.70)
    dspark_group.add_argument(
        "--elastic-consolidation-ratio", type=float, default=0.15
    )
    dspark_group.add_argument(
        "--elastic-middle-long-prob", type=float, default=0.30
    )
    dspark_group.add_argument(
        "--elastic-late-long-prob", type=float, default=0.50
    )
    dspark_group.add_argument(
        "--elastic-pair-probability", type=float, default=0.25
    )
    dspark_group.add_argument(
        "--elastic-projective-num-anchors", type=int, default=32
    )
    dspark_group.add_argument(
        "--elastic-projective-alpha", type=float, default=0.0025
    )
    dspark_group.add_argument(
        "--elastic-projective-top-k", type=int, default=64
    )
    dspark_group.add_argument("--elastic-overlap-gap", type=float, default=0.005)
    dspark_group.add_argument("--elastic-margin-gap", type=float, default=0.02)
    dspark_group.add_argument("--elastic-loss-budget", type=float, default=0.05)

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument(
        "--additional-train-data-path",
        action="append",
        default=[],
        help=(
            "Additional JSON/JSONL source. Repeat this flag and use mixture "
            "weights to build a language/domain-balanced epoch."
        ),
    )
    dataset_group.add_argument(
        "--train-data-mixture-weights",
        type=_comma_separated_floats,
        default=None,
        help=(
            "Source-level sampling masses for the primary plus additional "
            "training files, for example 0.5,0.5 for Chinese/English."
        ),
    )
    dataset_group.add_argument(
        "--dialogue-source-languages",
        type=_comma_separated_strings,
        default=None,
        help=(
            "Optional labels paired with the primary and additional sources, "
            "for example zh,en. They are logged with stable source IDs."
        ),
    )
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument(
        "--micro-batch-size", type=int, default=0,
        help="Opt-in physical split of each logical batch; 0 disables. Requires accumulation-steps=1. "
             "Preserves data/update cursor but averages microbatch-normalized losses, not exact full-batch losses.",
    )
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    add_muon_optimizer_args(training_group)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument(
        "--step-seeded-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Re-key anchor/temperature/proposal randomness from seed, global "
            "step and rank so P0/P1 comparisons remain reproducible."
        ),
    )
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument("--resume-extension-start-step", type=int, default=None)
    training_group.add_argument("--resume-extension-lr", type=float, default=1e-4)
    training_group.add_argument(
        "--trainable-parameter-scope",
        choices=["all", "carh-only", "selector-only"],
        default="all",
        help=(
            "Optimize all parameters, only CARH/Markov-head parameters, or only "
            "the conditional candidate selector. Restricted scopes keep the "
            "remaining loaded drafter parameters bitwise frozen."
        ),
    )
    training_group.add_argument(
        "--allow-conv-position-pruning",
        action="store_true",
        help="Warm start: allow only removed attention_conv/mlp_conv weights to be dropped.",
    )
    training_group.add_argument(
        "--init-draft-model-path",
        type=str,
        default=None,
        help=(
            "Initialize draft weights/config from a checkpoint without restoring "
            "optimizer or training progress. Mutually exclusive with --resume."
        ),
    )
    training_group.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="If set, stop after this many optimizer steps.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    # SGLang specific args
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    add_netprefix_args(parser)
    add_reference_args(parser)
    add_predictive_auxiliary_args(parser)
    add_bv_args(parser)
    return parser.parse_args()


def _apply_dspark_config(draft_config, args) -> None:
    """Set DSpark head fields on the draft config, preferring values already in
    the config JSON and falling back to CLI args."""
    defaults = {
        "markov_rank": args.markov_rank,
        "enable_confidence_head": args.enable_confidence_head,
        "confidence_head_with_markov": args.confidence_head_with_markov,
        "carh_gate_bias": args.carh_gate_bias,
    }
    draft_config.elastic_horizon_enabled = bool(args.elastic_horizon_enabled)
    draft_config.elastic_short_horizon = int(args.elastic_short_horizon)
    draft_config.elastic_long_horizon = int(args.elastic_long_horizon)
    for key, value in defaults.items():
        if not hasattr(draft_config, key) or getattr(draft_config, key) is None:
            setattr(draft_config, key, value)
    if args.markov_head_type is not None:
        draft_config.markov_head_type = args.markov_head_type
    elif not hasattr(draft_config, "markov_head_type"):
        draft_config.markov_head_type = "vanilla"
    if getattr(args, "carh_predecessor_count", None) is not None:
        draft_config.carh_predecessor_count = args.carh_predecessor_count
    elif not hasattr(draft_config, "carh_predecessor_count"):
        draft_config.carh_predecessor_count = 1
    if args.carh_predecessor_context_mode is not None:
        draft_config.carh_predecessor_context_mode = (
            args.carh_predecessor_context_mode
        )
    elif not hasattr(draft_config, "carh_predecessor_context_mode"):
        draft_config.carh_predecessor_context_mode = "none"
    if args.carh_sampled_prefix_memory_rank is not None:
        draft_config.carh_sampled_prefix_memory_rank = (
            args.carh_sampled_prefix_memory_rank
        )
    elif not hasattr(draft_config, "carh_sampled_prefix_memory_rank"):
        draft_config.carh_sampled_prefix_memory_rank = 0
    if args.selector_rank is not None:
        draft_config.selector_rank = args.selector_rank
    elif not hasattr(draft_config, "selector_rank"):
        draft_config.selector_rank = 0
    if args.selector_top_k is not None:
        draft_config.selector_top_k = args.selector_top_k
    elif not hasattr(draft_config, "selector_top_k"):
        draft_config.selector_top_k = 0
    if args.selector_runtime_enabled is not None:
        draft_config.selector_runtime_enabled = args.selector_runtime_enabled
    elif not hasattr(draft_config, "selector_runtime_enabled"):
        draft_config.selector_runtime_enabled = True
    if args.selector_margin_threshold is not None:
        draft_config.selector_margin_threshold = args.selector_margin_threshold
    elif not hasattr(draft_config, "selector_margin_threshold"):
        draft_config.selector_margin_threshold = 0.0
    if args.recall_correction_rank is not None:
        draft_config.recall_correction_rank = args.recall_correction_rank
    elif not hasattr(draft_config, "recall_correction_rank"):
        draft_config.recall_correction_rank = 0
    if args.recall_correction_gate_bias is not None:
        draft_config.recall_correction_gate_bias = (
            args.recall_correction_gate_bias
        )
    elif not hasattr(draft_config, "recall_correction_gate_bias"):
        draft_config.recall_correction_gate_bias = -2.0
    for arg_name, default in (
        ("prefix_state_mixer_mode", "none"),
        ("prefix_state_rank", 128),
        ("prefix_state_retention_bias", 2.0),
        ("prefix_state_update_bias", -1.0),
        ("prefix_state_gate_bias", -2.0),
        ("prefix_state_residual_scale", 0.1),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(draft_config, arg_name, value)
        elif not hasattr(draft_config, arg_name):
            setattr(draft_config, arg_name, default)
    if args.block_summary_rank is not None:
        draft_config.block_summary_rank = args.block_summary_rank
    elif not hasattr(draft_config, "block_summary_rank"):
        draft_config.block_summary_rank = 0
    if args.block_summary_source is not None:
        draft_config.block_summary_source = args.block_summary_source
    elif not hasattr(draft_config, "block_summary_source"):
        draft_config.block_summary_source = "anchor"
    if args.block_summary_gate_mode is not None:
        draft_config.block_summary_gate_mode = args.block_summary_gate_mode
    elif not hasattr(draft_config, "block_summary_gate_mode"):
        draft_config.block_summary_gate_mode = "position"
    for arg_name, default in (
        ("parallel_refiner_rank", 0),
        ("parallel_refiner_steps", 0),
        ("parallel_refiner_gate_bias", -1.0),
        ("parallel_refiner_second_step_bias", -1.0),
        ("parallel_refiner_residual_scale", 0.1),
        ("parallel_refiner_runtime_enabled", True),
        ("refiner_advantage_temperature_conditioned", False),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(draft_config, arg_name, value)
        elif not hasattr(draft_config, arg_name):
            setattr(draft_config, arg_name, default)

    method_config = dict(getattr(draft_config, "dflash_config", None) or {})
    if args.target_layer_fusion_mode is not None:
        method_config["target_layer_fusion_mode"] = args.target_layer_fusion_mode
    if args.target_layer_fusion_rank is not None:
        method_config["target_layer_fusion_rank"] = args.target_layer_fusion_rank
    if args.target_layer_fusion_source_dropout is not None:
        method_config["target_layer_fusion_source_dropout"] = args.target_layer_fusion_source_dropout
    if args.target_layer_fusion_residual_scale is not None:
        method_config["target_layer_fusion_residual_scale"] = (
            args.target_layer_fusion_residual_scale
        )
    if args.conv_kernel_size is not None:
        method_config["conv_kernel_size"] = args.conv_kernel_size
    if args.conv_group_size is not None:
        method_config["conv_group_size"] = args.conv_group_size
    for arg_name, config_name in (
        ("local_transition_rank", "local_transition_rank"),
        ("local_transition_heads", "local_transition_heads"),
        ("local_transition_window", "local_transition_window"),
        ("local_transition_residual_scale", "local_transition_residual_scale"),
        ("local_transition_gate_bias", "local_transition_gate_bias"),
        (
            "local_transition_conv_decay_start_ratio",
            "local_transition_conv_decay_start_ratio",
        ),
        (
            "local_transition_conv_decay_end_ratio",
            "local_transition_conv_decay_end_ratio",
        ),
        ("local_transition_final_conv_scale", "local_transition_final_conv_scale"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            method_config[config_name] = value
    if int(method_config.get("local_transition_rank", 0) or 0) > 0:
        method_config.setdefault("local_transition_heads", 4)
        method_config.setdefault("local_transition_window", 2)
        method_config.setdefault("local_transition_residual_scale", 0.1)
        method_config.setdefault("local_transition_gate_bias", -1.0)
        method_config.setdefault("local_transition_conv_decay_start_ratio", 0.10)
        method_config.setdefault("local_transition_conv_decay_end_ratio", 0.70)
        method_config.setdefault("local_transition_final_conv_scale", 0.0)
        # A newly expanded checkpoint starts with the full convolution teacher.
        method_config["local_transition_conv_scale"] = 1.0
    for arg_name, config_name in (
        ("conv_mode", "conv_mode"),
        ("conv_kernel_conditioning", "conv_kernel_conditioning"),
        ("conv_source_rank", "conv_source_rank"),
        ("conv_apply_to", "conv_apply_to"),
        ("conv_last_n_layers", "conv_last_n_layers"),
        ("conv_residual_scale", "conv_residual_scale"),
        ("conv_gate_bias", "conv_gate_bias"),
        ("conv_freeze_identity", "conv_freeze_identity"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            method_config[config_name] = value
    if bool(method_config.get("conv_kernel_size", 0)) != bool(
        method_config.get("conv_group_size", 0)
    ):
        raise ValueError(
            "--conv-kernel-size and --conv-group-size must be enabled or "
            "disabled together"
        )
    method_config["conv_identity_init"] = bool(args.conv_identity_init)
    conditioning = method_config.get("conv_kernel_conditioning", "input")
    if conditioning not in {"input", "output", "source-aware", "grouped16", "grouped64", "low-rank64", "static"}:
        raise ValueError(f"Unsupported conv_kernel_conditioning={conditioning!r}")
    if conditioning != "input" and (
        method_config.get("conv_mode", "legacy") != "legacy"
        or not method_config.get("conv_kernel_size", 0)
    ):
        raise ValueError(
            "Alternative kernel conditioning requires enabled legacy convolution"
        )
    if int(method_config.get("conv_source_rank", 32)) < 1:
        raise ValueError("conv_source_rank must be positive")
    draft_config.dflash_config = method_config


def build_models(args, device) -> Tuple[DFlashTargetModel, DSparkDraftModel]:
    """Build target model (backend wrapper) and DSpark draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using "
        f"{args.target_model_backend} backend"
    )

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    device_type = device.type

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=device_type if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
        if (
            hasattr(draft_config, "block_size")
            and draft_config.block_size != args.block_size
        ):
            print_on_rank0(
                f"Warning: config block_size ({draft_config.block_size}) differs from "
                f"command-line arg ({args.block_size}). Using config value."
            )
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    if args.elastic_horizon_enabled:
        original_block_size = int(draft_config.block_size)
        draft_config.block_size = int(args.elastic_long_horizon)
        print_on_rank0(
            "Elastic horizon: expanded fixed model block size from "
            f"{original_block_size} to K_max={draft_config.block_size}; "
            f"active horizons={args.elastic_short_horizon},"
            f"{args.elastic_long_horizon}."
        )

    _apply_dspark_config(draft_config, args)
    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DSparkDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}, "
        f"markov_rank={getattr(draft_config, 'markov_rank', 0)}, "
        f"markov_head_type={getattr(draft_config, 'markov_head_type', None)}, "
        f"selector_rank={getattr(draft_config, 'selector_rank', 0)}, "
        f"selector_top_k={getattr(draft_config, 'selector_top_k', 0)}, "
        "selector_margin_threshold="
        f"{getattr(draft_config, 'selector_margin_threshold', 0.0)}, "
        "selector_runtime_enabled="
        f"{getattr(draft_config, 'selector_runtime_enabled', True)}, "
        "conv_kernel_size="
        f"{draft_config.dflash_config.get('conv_kernel_size', 0)}, "
        "conv_group_size="
        f"{draft_config.dflash_config.get('conv_group_size', 0)}, "
        "conv_identity_init="
        f"{draft_config.dflash_config.get('conv_identity_init', True)}, "
        "conv_mode="
        f"{draft_config.dflash_config.get('conv_mode', 'legacy')}, "
        "conv_kernel_conditioning="
        f"{draft_config.dflash_config.get('conv_kernel_conditioning', 'input')}, "
        "conv_source_rank="
        f"{draft_config.dflash_config.get('conv_source_rank', 32)}, "
        "conv_apply_to="
        f"{draft_config.dflash_config.get('conv_apply_to', 'attention-mlp')}, "
        "conv_last_n_layers="
        f"{draft_config.dflash_config.get('conv_last_n_layers', 0)}, "
        "prefix_state_mixer_mode="
        f"{getattr(draft_config, 'prefix_state_mixer_mode', 'none')}, "
        "prefix_state_rank="
        f"{getattr(draft_config, 'prefix_state_rank', 0)}, "
        "block_summary_rank="
        f"{getattr(draft_config, 'block_summary_rank', 0)}, "
        "parallel_refiner_rank="
        f"{getattr(draft_config, 'parallel_refiner_rank', 0)}, "
        "parallel_refiner_steps="
        f"{getattr(draft_config, 'parallel_refiner_steps', 0)}, "
        f"enable_confidence_head={getattr(draft_config, 'enable_confidence_head', False)}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return target_model, draft_model


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    train_paths = [args.train_data_path, *args.additional_train_data_path]
    mixture_weights = args.train_data_mixture_weights
    if mixture_weights is None:
        mixture_weights = tuple(1.0 / len(train_paths) for _ in train_paths)
    if len(mixture_weights) != len(train_paths):
        raise ValueError(
            "--train-data-mixture-weights must contain one value for the primary "
            "and every additional training source"
        )
    if any(weight < 0 for weight in mixture_weights) or sum(mixture_weights) <= 0:
        raise ValueError("training data mixture weights must have positive mass")
    mixture_weights = tuple(
        weight / sum(mixture_weights) for weight in mixture_weights
    )
    source_languages = args.dialogue_source_languages
    if source_languages is None:
        source_languages = tuple(
            f"source-{index}" for index in range(len(train_paths))
        )
    if len(source_languages) != len(train_paths):
        raise ValueError(
            "--dialogue-source-languages must contain one label for the primary "
            "and every additional training source"
        )
    cache_params_string = (
        f"{train_paths}-{mixture_weights}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    min_loss_tokens = 2 * args.block_size
    processed_sources = []
    source_sizes = []
    for source_index, path in enumerate(train_paths):
        train_dataset = load_dataset("json", data_files=path)["train"]
        source_cache_key = f"{cache_key}-source-{source_index}"
        processed = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=source_cache_key,
            num_proc=args.build_dataset_num_proc,
        )
        original_size = len(processed)
        processed = processed.filter(
            lambda x: x["loss_mask"].sum() >= min_loss_tokens
        )
        if len(processed) == 0:
            raise ValueError(f"training source is empty after filtering: {path}")
        if "__source_id" in processed.column_names:
            processed = processed.remove_columns("__source_id")
        processed = processed.add_column(
            "__source_id", [source_index] * len(processed)
        )
        processed_sources.append(processed)
        source_sizes.append(len(processed))
        print_on_rank0(
            f"Filtered train source {source_index}: {original_size} -> "
            f"{len(processed)} samples; mass={mixture_weights[source_index]:.4f}; "
            f"language={source_languages[source_index]}; path={path}"
        )
    train_eagle3_dataset = (
        processed_sources[0]
        if len(processed_sources) == 1
        else concatenate_datasets(processed_sources)
    )
    # Stable IDs survive distributed sampling and allow independently exported
    # teacher trajectories to be joined without relying on dataloader order.
    train_eagle3_dataset = train_eagle3_dataset.add_column(
        "__sample_id", list(range(len(train_eagle3_dataset)))
    )
    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        weighted_sampling=len(processed_sources) > 1,
        sampler_seed=args.seed,
    )
    if len(processed_sources) > 1:
        initial_weights = build_source_mixture_weights(
            source_sizes, mixture_weights
        )
        train_dataloader.sampler.update_weights(initial_weights)

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, dspark_model, draft_model, optimizer):
    """Save checkpoint."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    runtime = getattr(dspark_model, "netprefix_runtime", None)
    if runtime is not None:
        runtime.save(save_dir, step)
    reference_runtime = getattr(dspark_model, "fixed_prefix_reference_runtime", None)
    if reference_runtime is not None:
        reference_runtime.save(save_dir, step)

    with FSDP.state_dict_type(dspark_model, StateDictType.FULL_STATE_DICT):
        state_dict = dspark_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        training_state_path = os.path.join(save_dir, "training_state.pt")
        # Save optimizer state in rank-agnostic (full-param) format so that
        # all ranks can load the same file and re-shard to their local
        # FSDP param shapes.  Without this, each rank stores different
        # param shapes (FSDP-sharded), and loading rank 0's state on
        # other ranks causes size-mismatch crashes in Adam.
        save_fsdp_optimizer_state(
            optimizer,
            dspark_model,
            training_state_path,
        )
        if dist.get_rank() == 0:
            # Append non-optimizer fields (epoch, step, args) to the
            # file that save_fsdp_optimizer_state already created.
            saved = torch.load(training_state_path, map_location="cpu", weights_only=False)
            saved["epoch"] = epoch
            saved["global_step"] = step
            saved["args"] = args
            torch.save(saved, training_state_path)

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            # Patch config.json: HuggingFace save_pretrained writes
            # architectures=["DSparkDraftModel"] (the SpecForge class name),
            # but sglang registers the draft model as "Qwen3DSparkModel".
            # Rewrite the field so sglang's ModelRegistry can find it.
            config_path = os.path.join(save_dir, "config.json")
            with open(config_path) as f:
                saved_cfg = json.load(f)
            if saved_cfg.get("architectures") == ["DSparkDraftModel"]:
                saved_cfg["architectures"] = ["Qwen3DSparkModel"]
                with open(config_path, "w") as f:
                    json.dump(saved_cfg, f, indent=2)
                    f.write("\n")

            # Copy the modeling files next to the checkpoint so auto_map can
            # resolve DSparkDraftModel (which subclasses DFlashDraftModel) on
            # reload with trust_remote_code.
            modeling_dir = os.path.join(
                os.path.dirname(__file__), "..", "specforge", "legacy"
            )
            for fname in ("dspark.py", "dflash.py", "dflash2.py",
                          "neighbor_residual.py", "light_conv_kernel.py"):
                src = os.path.join(modeling_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(save_dir, fname))

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    components: dict,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()
        if hasattr(optimizer, "get_learning_rates"):
            for name, value in optimizer.get_learning_rates().items():
                logdict[f"train/lr_{name}"] = value

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    for key, value in components.items():
        logdict[f"{mode}/{key}"] = value

    comp_str = " ".join(f"{k}={v:.4f}" for k, v in components.items())
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}"
        f"[{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?],"
        f" Loss: {loss:.4f}, Acc: {accuracy:.4f}, {comp_str}"
    )

    tracker.log(logdict, step=global_step)


def _patch_flex_attention_triton_backend() -> None:
    """Force transformer :func:`flex_attention` onto the general Triton kernel
    instead of the Inductor "flex_decoding" path.

    Inductor's flex-decoding kernel derives its autotune block size from
    ``BLOCK_M = next_pow2(query_len * gqa_shared_heads)`` and rejects every
    candidate that does not divide the sparse mask block size
    (``SPARSE_Q_BLOCK_SIZE % BLOCK_M != 0``).  For some batches (query length
    roughly in (16, 128) with GQA) this filters out *all* candidates, and
    because the decode path has no ATen fallback, training crashes in
    ``autotune_select_algorithm`` with ``NoValidChoicesError``.  Forcing
    ``BACKEND="TRITON"`` routes attention to the general Triton kernel, which
    has no such divisibility requirement and always has a valid configuration.
    """
    import transformers.integrations.flex_attention as _tfa

    if getattr(_tfa, "_specforge_force_flex_triton", False):
        return
    _orig = _tfa.compile_friendly_flex_attention

    def _force_triton(query, key, value, training=False, **kwargs):
        # Transformers may pass kernel_options=None (the Qwen3 path does), in
        # which case torch's flex_attention defaults BACKEND to AUTO internally.
        # Always inject an explicit BACKEND so the general Triton kernel (which
        # has no divisibility requirement) is selected.
        ko = dict(kwargs["kernel_options"]) if kwargs.get("kernel_options") else {}
        ko["BACKEND"] = "TRITON"
        kwargs["kernel_options"] = ko
        return _orig(query, key, value, training=training, **kwargs)

    _tfa.compile_friendly_flex_attention = _force_triton
    _tfa._specforge_force_flex_triton = True
    print_on_rank0(
        "Flex attention: forcing BACKEND=TRITON (avoids Inductor "
        "flex-decoding NoValidChoicesError on dynamic short-query batches)."
    )


def _load_draft_state_allowing_selector_expansion(
    draft_model: DSparkDraftModel,
    loaded_model: DSparkDraftModel,
    allow_conv_position_pruning: bool = False,
) -> None:
    """Load a warm start that may predate optional research modules.

    All shared weights remain strict. Newly constructed research modules may be
    missing, and DynamicConv-only parameters may be dropped when an SPSM run
    explicitly replaces that backbone component.
    """
    loaded_state = loaded_model.state_dict()
    replacing_dynamic_conv = (
        getattr(draft_model, "prefix_state_mixer", None) is not None
        and getattr(loaded_model, "dynamic_conv_enabled", False)
        and not getattr(draft_model, "dynamic_conv_enabled", False)
    )
    dropped_dynamic_conv = []
    if replacing_dynamic_conv:
        dropped_dynamic_conv = [
            key
            for key in loaded_state
            if ".attention_conv." in key or ".mlp_conv." in key
        ]
        loaded_state = {
            key: value
            for key, value in loaded_state.items()
            if key not in dropped_dynamic_conv
        }
    target_state = draft_model.state_dict()
    if allow_conv_position_pruning:
        removed = [
            key for key in loaded_state
            if key not in target_state
            and (".attention_conv." in key or ".mlp_conv." in key)
        ]
        for key in removed:
            del loaded_state[key]
        print_on_rank0(f"Explicit conv-position pruning: removed {removed}")
    expanded_depth_embeddings = []
    for key, value in list(loaded_state.items()):
        target_value = target_state.get(key)
        if (
            target_value is not None
            and key.endswith("depth_embedding.weight")
            and value.ndim == target_value.ndim == 2
            and value.shape[1:] == target_value.shape[1:]
            and value.shape[0] < target_value.shape[0]
        ):
            expanded = value.new_empty(target_value.shape)
            expanded[: value.shape[0]].copy_(value)
            # New depths start close to the deepest learned K=7 state while
            # retaining a tiny symmetry-breaking perturbation.
            tail = value[-1:].expand(target_value.shape[0] - value.shape[0], -1)
            expanded[value.shape[0] :].copy_(tail)
            expanded[value.shape[0] :].add_(
                torch.randn_like(expanded[value.shape[0] :]) * 1e-4
            )
            loaded_state[key] = expanded
            expanded_depth_embeddings.append(key)
    incompatible = draft_model.load_state_dict(
        loaded_state, strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    adding_second_predecessor = (
        getattr(draft_model.markov_head, "predecessor_count", 1) >= 2
        and getattr(loaded_model.markov_head, "predecessor_count", 1) == 1
        and getattr(loaded_model.markov_head, "markov_head_type", "") == "carh"
    )
    adding_third_predecessor = (
        getattr(draft_model.markov_head, "predecessor_count", 1) == 3
        and getattr(loaded_model.markov_head, "predecessor_count", 1) < 3
        and getattr(loaded_model.markov_head, "markov_head_type", "") == "carh"
    )
    adding_sampled_prefix_memory = (
        int(getattr(draft_model.markov_head, "sampled_prefix_memory_rank", 0)) > 0
        and int(getattr(loaded_model.markov_head, "sampled_prefix_memory_rank", 0)) == 0
        and getattr(loaded_model.markov_head, "markov_head_type", "") == "carh"
    )
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(
            (
                "candidate_selector.",
                "recall_correction.",
                "target_layer_fusion_weights",
                "target_layer_fusion_proj.",
                "prefix_state_mixer.",
                "block_summary.",
                "parallel_refiner.",
            )
        )
        and ".local_transition_attention." not in key
        and not (adding_second_predecessor and key == "markov_head.second_prev_proj.weight")
        and not (adding_third_predecessor and key == "markov_head.third_prev_proj.weight")
        and not (adding_sampled_prefix_memory and key.startswith("markov_head.sampled_prefix_memory."))
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Warm-start state is incompatible: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    if incompatible.missing_keys:
        print_on_rank0(
            "Warm start predates optional research modules; initialized the "
            f"following parameters from scratch: {incompatible.missing_keys}"
        )
    if dropped_dynamic_conv:
        print_on_rank0(
            "SPSM replacement warm start: discarded DynamicConv-only "
            f"parameters: {dropped_dynamic_conv}"
        )
    if expanded_depth_embeddings:
        print_on_rank0(
            "Elastic warm start: expanded depth embeddings for K_max: "
            f"{expanded_depth_embeddings}"
        )


def _load_parallel_refiner_teacher(
    draft_model: DSparkDraftModel, checkpoint_path: str, freeze: bool
) -> None:
    """Overlay only HAPR weights, preserving the stronger CARH student."""
    if getattr(draft_model, "parallel_refiner", None) is None:
        raise ValueError("parallel refiner teacher path requires an enabled refiner")
    teacher_model = DSparkDraftModel.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16
    )
    teacher_refiner = getattr(teacher_model, "parallel_refiner", None)
    if teacher_refiner is None:
        raise ValueError(f"teacher checkpoint has no parallel refiner: {checkpoint_path}")
    incompatible = draft_model.parallel_refiner.load_state_dict(
        teacher_refiner.state_dict(), strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(
            (
                "advantage_class_proj.",
                "advantage_value_proj.",
                "advantage_temperature_proj.",
            )
        )
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Parallel refiner teacher is incompatible: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    del teacher_model
    if freeze:
        for name, parameter in draft_model.parallel_refiner.named_parameters():
            parameter.requires_grad = name.startswith(
                (
                    "advantage_class_proj.",
                    "advantage_value_proj.",
                    "advantage_temperature_proj.",
                )
            )
    print_on_rank0(
        "Loaded training-only parallel refiner teacher from "
        f"{checkpoint_path}; freeze={freeze}"
    )


def _forward_dspark_data_batch(
    data, args, device, target_model, dspark_model, needs_target_hidden,
    global_step, micro_index=0,
):
    # Slice on CPU before calling this function: neither target nor drafter
    # materializes the whole logical batch on GPU in microbatch mode.
    input_ids = data["input_ids"].to(device, non_blocking=True)
    attention_mask = data["attention_mask"].to(device, non_blocking=True)
    loss_mask = data["loss_mask"].to(device, non_blocking=True)
    sample_ids = data["sample_id"].to(device, non_blocking=True)
    source_ids = data["source_id"].to(device, non_blocking=True)
    rollout_seed = (
        args.seed * 1_000_003 + global_step + dist.get_rank() * 10_000_019
        + micro_index * 100_000_007
    )
    if args.multi_teacher_oracle_export_dir or args.multi_teacher_oracle_cache:
        torch.manual_seed(rollout_seed)
    target_output = target_model.generate_dflash_data(input_ids, attention_mask, loss_mask)
    hidden_states = target_output.hidden_states.to(device, non_blocking=True)
    last_hidden_states = target_output.last_hidden_states
    if last_hidden_states is not None:
        last_hidden_states = last_hidden_states.to(device, non_blocking=True)
    elif needs_target_hidden:
        raise RuntimeError(
            "DSpark L1/confidence losses are enabled but the target backend "
            f"({args.target_model_backend}) did not surface last_hidden_states. "
            "Use --target-model-backend hf, or run CE-only with "
            "--l1-loss-alpha 0 --no-confidence-head."
        )
    if args.step_seeded_rollouts:
        torch.manual_seed(rollout_seed)
    forward_context = torch.no_grad() if args.multi_teacher_oracle_export_dir else contextlib.nullcontext()
    with forward_context:
        return dspark_model(
            input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask,
            last_hidden_states=last_hidden_states, attention_mask=attention_mask,
            sample_ids=sample_ids, source_ids=source_ids,
        )


def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    if args.dspark_loss_type == "tv-acceptance":
        if not math.isfinite(args.tv_sampling_temperature) or args.tv_sampling_temperature <= 0:
            raise ValueError("--tv-sampling-temperature must be finite and positive")
        if min(args.tv_objective_chunk_blocks, args.tv_verification_batch_size) < 1:
            raise ValueError("TV objective and verification chunk sizes must be positive")
        if args.tp_size != 1:
            raise ValueError("tv-acceptance currently requires --tp-size 1")
        if args.accumulation_steps != 1 or args.micro_batch_size:
            raise ValueError("tv-acceptance currently requires accumulation-steps=1 and micro-batch-size=0 for exact block normalization")
        if (args.offline_acceptance_objective != "none" or args.netprefix_mode != "off"
                or args.fixed_prefix_reference_alpha or args.elastic_horizon_enabled
                or args.trainable_parameter_scope != "all" or args.vat_enabled
                or args.multi_teacher_oracle_export_dir or args.multi_teacher_oracle_cache):
            raise ValueError("tv-acceptance is a standalone objective; disable auxiliary training modes")
    validate_bv_args(args)
    validate_netprefix_args(args)
    validate_reference_args(args)
    if args.on_policy_start_step is not None and args.on_policy_start_step < 1:
        raise ValueError("--on-policy-start-step must be positive")
    if args.on_policy_ramp_steps < 0:
        raise ValueError("--on-policy-ramp-steps must be non-negative")
    if not 0 <= args.carh_sampled_prefix_memory_start_ratio <= 1:
        raise ValueError("--carh-sampled-prefix-memory-start-ratio must be in [0, 1]")
    if args.carh_sampled_prefix_memory_ramp_ratio < 0:
        raise ValueError("--carh-sampled-prefix-memory-ramp-ratio must be non-negative")
    if (args.carh_sampled_prefix_memory_start_ratio
            + args.carh_sampled_prefix_memory_ramp_ratio > 1):
        raise ValueError("sampled-prefix memory start+ramp ratios must be <= 1")
    if args.micro_batch_size < 0 or args.micro_batch_size > args.batch_size:
        raise ValueError("micro-batch-size must be 0 (disabled) or between 1 and batch-size")
    if args.micro_batch_size:
        if args.accumulation_steps != 1:
            raise ValueError("Physical microbatch splitting requires accumulation-steps=1")
        if args.multi_teacher_oracle_export_dir or args.multi_teacher_oracle_cache:
            raise ValueError("Physical microbatch mode does not support multi-teacher export/cache")
        logging.warning(
            "Physical microbatch=%d, logical batch=%d: data/update steps stay unchanged; "
            "loss normalization, random draws and batch-dependent statistics may change.",
            args.micro_batch_size, args.batch_size,
        )
    if args.optimizer == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("--optimizer muon requires torch.optim.Muon; use repository PyTorch 2.11")
        validate_muon_options(lr=args.learning_rate,
            muon_lr=10 * args.learning_rate if args.muon_lr is None else args.muon_lr,
            momentum=args.muon_momentum, weight_decay=args.weight_decay,
            muon_weight_decay=args.muon_weight_decay, ns_steps=args.muon_ns_steps,
            adjust_lr_fn=args.muon_adjust_lr_fn)
    if args.multi_teacher_oracle_export_dir and args.multi_teacher_oracle_cache:
        raise ValueError("multi-teacher export and distillation are separate phases")
    if args.multi_teacher_oracle_mode != "none" and not args.multi_teacher_oracle_cache:
        raise ValueError("multi-teacher oracle mode requires --multi-teacher-oracle-cache")
    if args.multi_teacher_oracle_export_dir and not args.init_draft_model_path:
        raise ValueError("multi-teacher export requires --init-draft-model-path")
    if args.multi_teacher_export_top_k < 2:
        raise ValueError("multi-teacher export top-k must be at least 2")
    if args.resume and args.init_draft_model_path:
        raise ValueError("--resume and --init-draft-model-path are mutually exclusive")
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    _patch_flex_attention_triton_backend()

    device = get_local_device()
    device_type = device.type

    needs_target_hidden = (
        args.offline_acceptance_objective != "none"
    ) or (args.bv_loss_alpha > 0) or (args.carh_predecessor_diagnostics_interval > 0) or (args.l1_loss_alpha > 0) or (
        args.enable_confidence_head and args.confidence_head_alpha > 0
    ) or args.pace_mode in {"overlap", "hybrid"} or (
        args.elastic_horizon_enabled and args.elastic_projective_alpha > 0
    )

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(
            draft_model_last_checkpoint, "config.json"
        )
        if os.path.exists(checkpoint_config_path):
            print(f"Loading draft config from checkpoint: {checkpoint_config_path}")
            args.draft_config_path = checkpoint_config_path
    elif args.init_draft_model_path:
        init_config_path = os.path.join(args.init_draft_model_path, "config.json")
        if not os.path.exists(init_config_path):
            raise FileNotFoundError(
                f"Initialization checkpoint has no config.json: {init_config_path}"
            )
        print(f"Loading draft config from initialization checkpoint: {init_config_path}")
        args.draft_config_path = init_config_path

    target_model, draft_model = build_models(args, device)
    validate_predictive_auxiliary(args, draft_model)

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DSparkDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            validate_optimizer_resume(resume_state, args.optimizer)
            validate_predictive_resume(resume_state.get("args", {}), args)
            validate_bv_resume(resume_state.get("args", {}), args)
            saved_tv_args = resume_state.get("args", {})
            if isinstance(saved_tv_args, argparse.Namespace):
                saved_tv_args = vars(saved_tv_args)
            for key, default in (("dspark_loss_type", "ce-l1"),
                                 ("tv_sampling_temperature", 1.0),
                                 ("tv_objective_chunk_blocks", 8),
                                 ("tv_verification_batch_size", 4)):
                if saved_tv_args.get(key, default) != getattr(args, key):
                    raise ValueError(f"Cannot resume with changed {key}; use --init-draft-model-path for a new objective")
            if int(getattr(draft_model.config, "carh_sampled_prefix_memory_rank", 0)):
                saved_args = resume_state.get("args", {})
                if isinstance(saved_args, argparse.Namespace):
                    saved_args = vars(saved_args)
                for key in (
                    "carh_sampled_prefix_memory_start_ratio",
                    "carh_sampled_prefix_memory_ramp_ratio",
                ):
                    if saved_args.get(key) != getattr(args, key):
                        raise ValueError(
                            f"Sampled-prefix memory scheduled resume setting changed: {key}"
                        )
            if args.on_policy_start_step is not None:
                saved_args = resume_state.get("args", {})
                if isinstance(saved_args, argparse.Namespace):
                    saved_args = vars(saved_args)
                for key in ("on_policy_start_step", "on_policy_ramp_steps", "max_steps", "num_epochs",
                            "batch_size", "accumulation_steps", "seed", "train_data_path",
                            "on_policy_repair_value", "on_policy_repair_value_horizon",
                            "fixed_prefix_reference_alpha", "fixed_prefix_reference_batches",
                            "fixed_prefix_reference_interval"):
                    # Pre-D/E checkpoints have no new fields; preserve C resumes.
                    if key not in saved_args and key in (
                        "on_policy_repair_value", "on_policy_repair_value_horizon",
                        "fixed_prefix_reference_alpha", "fixed_prefix_reference_batches",
                        "fixed_prefix_reference_interval",
                    ) and not args.on_policy_repair_value and not args.fixed_prefix_reference_alpha:
                        continue
                    if saved_args.get(key) != getattr(args, key):
                        raise ValueError(f"On-policy scheduled resume setting changed: {key}")
            if args.micro_batch_size:
                validate_microbatch_resume(resume_state.get("args"), args)
            print(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )
    elif args.init_draft_model_path:
        loaded_model = DSparkDraftModel.from_pretrained(
            args.init_draft_model_path, torch_dtype=torch.bfloat16
        )
        _load_draft_state_allowing_selector_expansion(
            draft_model, loaded_model,
            allow_conv_position_pruning=args.allow_conv_position_pruning,
        )
        del loaded_model
        print(
            "Loaded draft model weights without optimizer state from "
            f"{args.init_draft_model_path}"
        )

    if args.optimizer == "muon" and args.resume and resume_state is None:
        raise ValueError("Muon --resume requires a training_state.pt checkpoint; use weight-only init otherwise")

    if args.parallel_refiner_teacher_path:
        _load_parallel_refiner_teacher(
            draft_model,
            args.parallel_refiner_teacher_path,
            args.freeze_parallel_refiner_teacher,
        )

    if args.freeze_dynamic_conv_during_transition:
        frozen_conv_parameters = 0
        for layer in draft_model.layers:
            for conv in (
                getattr(layer, "attention_conv", None),
                getattr(layer, "mlp_conv", None),
            ):
                if conv is not None:
                    for parameter in conv.parameters():
                        parameter.requires_grad_(False)
                        frozen_conv_parameters += parameter.numel()
        print_on_rank0(
            "Local transition migration: froze "
            f"{frozen_conv_parameters:,} DynamicConv parameters."
        )

    trainable_parameters, scope_frozen_parameters = _apply_trainable_parameter_scope(
        draft_model, args.trainable_parameter_scope
    )
    print_on_rank0(
        f"Trainable parameter scope={args.trainable_parameter_scope}: "
        f"trainable={trainable_parameters:,}, frozen={scope_frozen_parameters:,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    dialogue_token_class_lookup = None
    if (
        args.on_policy_anchor_sampling == "dialogue-frontier"
        or args.dialogue_occupancy_diagnostics
        or args.dialogue_prefix_phase_weighting
        or args.dialogue_t2cm_boundary_weighting
        or args.dialogue_dsg_adaptive_margin
        or args.on_policy_language_hazard_credit
    ):
        dialogue_token_class_lookup = _build_dialogue_token_class_lookup(tokenizer)
        class_counts = torch.bincount(
            dialogue_token_class_lookup.long(), minlength=6
        ).tolist()
        print_on_rank0(
            "Dialogue token classes other/Han/Latin/number/punctuation/special="
            f"{class_counts}"
        )

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    print_on_rank0(f"Total training steps: {total_steps}")
    if args.bv_loss_alpha > 0:
        draft_model.config.bv_training = dict(
            objective="vocabulary-integrated-block-log-bv", alpha=args.bv_loss_alpha,
            temperature=args.bv_temperature, anneal_ratio=args.bv_anneal_ratio,
            block_chunk_size=args.bv_block_chunk_size, log_score_floor=-80.0,
            normalization="valid-block-mean", replaces="l1", corrected_proposal=True,
        )
        print_on_rank0(f"BV loss replacement (training only): {draft_model.config.bv_training}")
        print_on_rank0(
            f"BV beta 0->1 over first {round(total_steps * args.bv_anneal_ratio)} optimizer steps; "
            "CE/confidence/prefix auxiliaries retained. No D-PACE/depth weights inside BV. "
            "Target-path interpretation requires matching regen sampling settings; "
            "BV diagnostics are NOT measured SGLang acceptance lengths."
        )
    if args.offline_acceptance_objective != "none":
        cold_steps = round(
            total_steps * args.offline_acceptance_cold_start_ratio
        )
        transition_steps = round(
            total_steps * args.offline_acceptance_transition_ratio
        )
        print_on_rank0(
            "Offline acceptance objective: "
            f"{args.offline_acceptance_objective}; "
            f"LK steps=0..{cold_steps}, "
            f"transition steps={cold_steps}..{cold_steps + transition_steps}, "
            f"E2E steps={cold_steps + transition_steps}..{total_steps}; "
            f"T={args.offline_acceptance_temperature}, "
            "rollout=disabled by objective design"
        )

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device_type,
        trust_remote_code=args.trust_remote_code,
    )

    semantic_codebook = None
    if args.conv_source_semantic_alpha > 0:
        print_on_rank0("Preparing fixed semantic codebook (chunked CPU PCA, rank zero only)...")
        semantic_codebook = prepare_semantic_codebook(
            args, target_components.lm_head.weight, draft_model.layers[0].mlp_conv.source_rank,
        )
        print_on_rank0("Semantic codebook ready; artifact: " + os.path.join(args.output_dir, "source_semantic_codebook.pt"))
    draft_model.config.predictive_auxiliary = dict(
        conv_source_semantic_alpha=args.conv_source_semantic_alpha,
        carh_reference_calibration_alpha=args.carh_reference_calibration_alpha,
        warmup_ratio=args.predictive_aux_warmup_ratio,
        ramp_ratio=args.predictive_aux_ramp_ratio,
        source_semantic_codebook_seed=args.source_semantic_codebook_seed,
        semantic_scope="mlp-source-only", reference_scope="valid-in-block-predecessors",
    )
    if args.conv_source_semantic_alpha > 0 or args.carh_reference_calibration_alpha > 0:
        print_on_rank0(f"Training-only predictive auxiliaries: {draft_model.config.predictive_auxiliary}")

    dspark_model = OnlineDSparkModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        conv_source_semantic_alpha=args.conv_source_semantic_alpha,
        carh_reference_calibration_alpha=args.carh_reference_calibration_alpha,
        predictive_aux_warmup_ratio=args.predictive_aux_warmup_ratio,
        predictive_aux_ramp_ratio=args.predictive_aux_ramp_ratio,
        source_semantic_codebook=semantic_codebook,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        ce_loss_alpha=args.ce_loss_alpha,
        l1_loss_alpha=args.l1_loss_alpha,
        bv_loss_alpha=args.bv_loss_alpha,
        bv_temperature=args.bv_temperature,
        bv_anneal_ratio=args.bv_anneal_ratio,
        bv_block_chunk_size=args.bv_block_chunk_size,
        offline_acceptance_objective=args.offline_acceptance_objective,
        offline_acceptance_greedy_weight=(
            args.offline_acceptance_greedy_weight
        ),
        offline_acceptance_stochastic_weight=(
            args.offline_acceptance_stochastic_weight
        ),
        offline_acceptance_temperature=args.offline_acceptance_temperature,
        offline_acceptance_lk_eta=args.offline_acceptance_lk_eta,
        offline_acceptance_dpace_rho=args.offline_acceptance_dpace_rho,
        offline_acceptance_cold_start_ratio=(
            args.offline_acceptance_cold_start_ratio
        ),
        offline_acceptance_transition_ratio=(
            args.offline_acceptance_transition_ratio
        ),
        confidence_head_alpha=args.confidence_head_alpha,
        confidence_target_mode=args.confidence_target_mode,
        confidence_detach_backbone=args.confidence_detach_backbone,
        carh_predecessor_diagnostics_interval=args.carh_predecessor_diagnostics_interval,
        carh_sampled_prefix_memory_start_ratio=(
            args.carh_sampled_prefix_memory_start_ratio
        ),
        carh_sampled_prefix_memory_ramp_ratio=(
            args.carh_sampled_prefix_memory_ramp_ratio
        ),
        recall_correction_alpha=args.recall_correction_alpha,
        recall_partition_top_k=args.recall_partition_top_k,
        recall_correction_loss_budget=args.recall_correction_loss_budget,
        pace_mode=args.pace_mode,
        pace_alpha=args.pace_alpha,
        pace_hybrid_beta=args.pace_hybrid_beta,
        pace_apply_to=args.pace_apply_to,
        pace_blend_max=args.pace_blend_max,
        pace_warmup_ratio=args.pace_warmup_ratio,
        pace_ramp_ratio=args.pace_ramp_ratio,
        pace_decay_start_ratio=args.pace_decay_start_ratio,
        pace_blend_final=args.pace_blend_final,
        pace_residual_beta=args.pace_residual_beta,
        pace_residual_min=args.pace_residual_min,
        pace_residual_max=args.pace_residual_max,
        prefix_credit_mode=args.prefix_credit_mode,
        prefix_credit_alpha=args.prefix_credit_alpha,
        prefix_credit_warmup_ratio=args.prefix_credit_warmup_ratio,
        prefix_credit_ramp_ratio=args.prefix_credit_ramp_ratio,
        prefix_credit_backbone_grad_scale=(
            args.prefix_credit_backbone_grad_scale
        ),
        shallow_frc_alpha=args.shallow_frc_alpha,
        shallow_frc_max_depth=args.shallow_frc_max_depth,
        shallow_frc_margin=args.shallow_frc_margin,
        shallow_frc_temperature=args.shallow_frc_temperature,
        dfap_alpha=args.dfap_alpha,
        dfap_min_depth=args.dfap_min_depth,
        dfap_margin=args.dfap_margin,
        dfap_temperature=args.dfap_temperature,
        state_credit_partition=args.state_credit_partition,
        conv_gate_loss_alpha=args.conv_gate_loss_alpha,
        block_summary_counterfactual_alpha=args.block_summary_counterfactual_alpha,
        transition_credit_alpha=args.transition_credit_alpha,
        transition_credit_max_depth=args.transition_credit_max_depth,
        transition_credit_margin=args.transition_credit_margin,
        transition_credit_temperature=args.transition_credit_temperature,
        transition_credit_loss_budget=args.transition_credit_loss_budget,
        transition2_margin_alpha=args.transition2_margin_alpha,
        transition2_margin_floor=args.transition2_margin_floor,
        transition2_margin_temperature=args.transition2_margin_temperature,
        transition2_margin_loss_budget=args.transition2_margin_loss_budget,
        prefix_bottleneck_alpha=args.prefix_bottleneck_alpha,
        prefix_bottleneck_max_depth=args.prefix_bottleneck_max_depth,
        prefix_bottleneck_margin_floor=args.prefix_bottleneck_margin_floor,
        prefix_bottleneck_softmin_temperature=(
            args.prefix_bottleneck_softmin_temperature
        ),
        prefix_bottleneck_loss_temperature=(
            args.prefix_bottleneck_loss_temperature
        ),
        prefix_bottleneck_loss_budget=args.prefix_bottleneck_loss_budget,
        deep_survival_guard_alpha=args.deep_survival_guard_alpha,
        deep_survival_guard_min_prefix=args.deep_survival_guard_min_prefix,
        deep_survival_guard_start_depth=args.deep_survival_guard_start_depth,
        deep_survival_guard_margin_floor=(
            args.deep_survival_guard_margin_floor
        ),
        deep_survival_guard_temperature=args.deep_survival_guard_temperature,
        deep_survival_guard_loss_budget=(
            args.deep_survival_guard_loss_budget
        ),
        carh_recoverability_top_k=args.carh_recoverability_top_k,
        carh_recovery_loss_alpha=args.carh_recovery_loss_alpha,
        carh_preservation_loss_alpha=args.carh_preservation_loss_alpha,
        carh_noop_loss_alpha=args.carh_noop_loss_alpha,
        carh_preservation_margin=args.carh_preservation_margin,
        carh_preservation_temperature=args.carh_preservation_temperature,
        carh_recoverability_loss_budget=(
            args.carh_recoverability_loss_budget
        ),
        carh_gate_calibration_alpha=args.carh_gate_calibration_alpha,
        carh_gate_noop_alpha=args.carh_gate_noop_alpha,
        carh_gate_margin_threshold=args.carh_gate_margin_threshold,
        carh_gate_calibration_warmup_ratio=(
            args.carh_gate_calibration_warmup_ratio
        ),
        selector_loss_alpha=args.selector_loss_alpha,
        selector_rollout_ratio=args.selector_rollout_ratio,
        selector_training_mode=args.selector_training_mode,
        selector_distill_alpha=args.selector_distill_alpha,
        selector_preservation_alpha=args.selector_preservation_alpha,
        selector_distill_temperature=args.selector_distill_temperature,
        selector_advantage_margin=args.selector_advantage_margin,
        selector_teacher_warmup_ratio=args.selector_teacher_warmup_ratio,
        selector_distill_start_ratio=args.selector_distill_start_ratio,
        selector_consolidation_ratio=args.selector_consolidation_ratio,
        selector_loss_budget=args.selector_loss_budget,
        selector_margin_threshold=float(draft_model.selector_margin_threshold),
        selector_tree_loss_alpha=args.selector_tree_loss_alpha,
        selector_tree_branch_width=args.selector_tree_branch_width,
        selector_tree_margin=args.selector_tree_margin,
        selector_tree_temperature=args.selector_tree_temperature,
        on_policy_survival_alpha=args.on_policy_survival_alpha,
        on_policy_full_alpha=args.on_policy_full_alpha,
        on_policy_first_rejection_max_depth=(
            args.on_policy_first_rejection_max_depth
        ),
        on_policy_boundary_max_depth=args.on_policy_boundary_max_depth,
        on_policy_deep_alpha=args.on_policy_deep_alpha,
        on_policy_deep_start_depth=args.on_policy_deep_start_depth,
        on_policy_margin_floor=args.on_policy_margin_floor,
        on_policy_temperature=args.on_policy_temperature,
        on_policy_rollout_temperatures=args.on_policy_rollout_temperatures,
        on_policy_rollout_temperature_probs=(
            args.on_policy_rollout_temperature_probs
        ),
        on_policy_anchor_sampling=args.on_policy_anchor_sampling,
        on_policy_hazard_power=args.on_policy_hazard_power,
        on_policy_flatness_power_max=args.on_policy_flatness_power_max,
        on_policy_flatness_warmup_ratio=(
            args.on_policy_flatness_warmup_ratio
        ),
        on_policy_uniform_exploration=args.on_policy_uniform_exploration,
        dialogue_phase_masses=args.dialogue_phase_masses,
        dialogue_boundary_boost=args.dialogue_boundary_boost,
        dialogue_script_transition_boost=(
            args.dialogue_script_transition_boost
        ),
        dialogue_occupancy_diagnostics=args.dialogue_occupancy_diagnostics,
        dialogue_num_sources=1 + len(args.additional_train_data_path),
        dialogue_token_class_lookup=dialogue_token_class_lookup,
        dialogue_prefix_phase_weighting=(
            args.dialogue_prefix_phase_weighting
        ),
        dialogue_t2cm_boundary_weighting=(
            args.dialogue_t2cm_boundary_weighting
        ),
        dialogue_dsg_adaptive_margin=args.dialogue_dsg_adaptive_margin,
        dialogue_dsg_late_margin_bonus=(
            args.dialogue_dsg_late_margin_bonus
        ),
        dialogue_dsg_boundary_margin_bonus=(
            args.dialogue_dsg_boundary_margin_bonus
        ),
        on_policy_distributional_top_k=args.on_policy_distributional_top_k,
        on_policy_distributional_min_top_k=(
            args.on_policy_distributional_min_top_k
        ),
        on_policy_distributional_mass_threshold=(
            args.on_policy_distributional_mass_threshold
        ),
        on_policy_distributional_alpha=args.on_policy_distributional_alpha,
        on_policy_distributional_deep_alpha=(
            args.on_policy_distributional_deep_alpha
        ),
        on_policy_distributional_cold_start_ratio=(
            args.on_policy_distributional_cold_start_ratio
        ),
        on_policy_rejection_aligned_alpha=(
            args.on_policy_rejection_aligned_alpha
        ),
        on_policy_rejection_aligned_survival_blend=(
            args.on_policy_rejection_aligned_survival_blend
        ),
        on_policy_temperature_exclusive_routing=(
            args.on_policy_temperature_exclusive_routing
        ),
        on_policy_mixed_kl_alpha=args.on_policy_mixed_kl_alpha,
        on_policy_mixed_kl_accepted_weight=(
            args.on_policy_mixed_kl_accepted_weight
        ),
        on_policy_mixed_kl_rejected_weight=(
            args.on_policy_mixed_kl_rejected_weight
        ),
        on_policy_mixed_kl_rejection_decay=(
            args.on_policy_mixed_kl_rejection_decay
        ),
        on_policy_clipped_rkl_alpha=args.on_policy_clipped_rkl_alpha,
        on_policy_clipped_rkl_clip=args.on_policy_clipped_rkl_clip,
        on_policy_clipped_rkl_temperature_weights=(
            args.on_policy_clipped_rkl_temperature_weights
        ),
        on_policy_target_distribution_temperature_floor=(
            args.on_policy_target_distribution_temperature_floor
        ),
        on_policy_target_margin_alpha=args.on_policy_target_margin_alpha,
        on_policy_target_margin_scale=args.on_policy_target_margin_scale,
        on_policy_target_margin_offset=args.on_policy_target_margin_offset,
        on_policy_target_margin_min=args.on_policy_target_margin_min,
        on_policy_target_margin_max=args.on_policy_target_margin_max,
        on_policy_target_margin_max_depth=(
            args.on_policy_target_margin_max_depth
        ),
        on_policy_deployment_top_k=args.on_policy_deployment_top_k,
        on_policy_deployment_top_p=args.on_policy_deployment_top_p,
        on_policy_credit_partition=args.on_policy_credit_partition,
        on_policy_middle_max_depth=args.on_policy_middle_max_depth,
        on_policy_middle_alpha=args.on_policy_middle_alpha,
        on_policy_temperature_hazard_credit=(
            args.on_policy_temperature_hazard_credit
        ),
        on_policy_hazard_ema_decay=args.on_policy_hazard_ema_decay,
        on_policy_hazard_weight_min=args.on_policy_hazard_weight_min,
        on_policy_hazard_weight_max=args.on_policy_hazard_weight_max,
        on_policy_marginal_value_credit=(
            args.on_policy_marginal_value_credit
        ),
        on_policy_marginal_value_temperature=(
            args.on_policy_marginal_value_temperature
        ),
        on_policy_marginal_value_weight_min=(
            args.on_policy_marginal_value_weight_min
        ),
        on_policy_marginal_value_weight_max=(
            args.on_policy_marginal_value_weight_max
        ),
        on_policy_language_hazard_credit=(
            args.on_policy_language_hazard_credit
        ),
        on_policy_language_han_threshold=(
            args.on_policy_language_han_threshold
        ),
        on_policy_language_hazard_ema_decay=(
            args.on_policy_language_hazard_ema_decay
        ),
        on_policy_language_hazard_weight_min=(
            args.on_policy_language_hazard_weight_min
        ),
        on_policy_language_hazard_weight_max=(
            args.on_policy_language_hazard_weight_max
        ),
        on_policy_mix_ratio_max=args.on_policy_mix_ratio_max,
        on_policy_start_step=args.on_policy_start_step,
        on_policy_ramp_steps=args.on_policy_ramp_steps,
        on_policy_repair_value=args.on_policy_repair_value,
        on_policy_repair_value_horizon=args.on_policy_repair_value_horizon,
        on_policy_interval=args.on_policy_interval,
        on_policy_loss_budget=args.on_policy_loss_budget,
        on_policy_preservation_alpha=args.on_policy_preservation_alpha,
        on_policy_regression_alpha=args.on_policy_regression_alpha,
        on_policy_regression_margin=args.on_policy_regression_margin,
        on_policy_greedy_preservation_alpha=(
            args.on_policy_greedy_preservation_alpha
        ),
        on_policy_greedy_preservation_margin_floor=(
            args.on_policy_greedy_preservation_margin_floor
        ),
        on_policy_greedy_preservation_temperature=(
            args.on_policy_greedy_preservation_temperature
        ),
        on_policy_reset_replay_alpha=args.on_policy_reset_replay_alpha,
        on_policy_reset_replay_horizon=args.on_policy_reset_replay_horizon,
        on_policy_reset_replay_max_rejection_depth=(
            args.on_policy_reset_replay_max_rejection_depth
        ),
        on_policy_reset_replay_start_ratio=(
            args.on_policy_reset_replay_start_ratio
        ),
        on_policy_reset_replay_ramp_ratio=(
            args.on_policy_reset_replay_ramp_ratio
        ),
        on_policy_reset_replay_loss_budget=(
            args.on_policy_reset_replay_loss_budget
        ),
        on_policy_reset_replay_advantage_threshold=(
            args.on_policy_reset_replay_advantage_threshold
        ),
        on_policy_reset_replay_value_clip=(
            args.on_policy_reset_replay_value_clip
        ),
        on_policy_preference_alpha=args.on_policy_preference_alpha,
        on_policy_preference_value_gap=args.on_policy_preference_value_gap,
        on_policy_preference_temperature=args.on_policy_preference_temperature,
        on_policy_preference_loss_budget=args.on_policy_preference_loss_budget,
        on_policy_pareto_credit=args.on_policy_pareto_credit,
        on_policy_pareto_ema_decay=args.on_policy_pareto_ema_decay,
        on_policy_pareto_temperature=args.on_policy_pareto_temperature,
        on_policy_pareto_weight_min=args.on_policy_pareto_weight_min,
        on_policy_pareto_weight_max=args.on_policy_pareto_weight_max,
        parallel_refiner_hazard_alpha=args.parallel_refiner_hazard_alpha,
        parallel_refiner_recovery_alpha=args.parallel_refiner_recovery_alpha,
        parallel_refiner_preservation_alpha=(
            args.parallel_refiner_preservation_alpha
        ),
        parallel_refiner_margin_floor=args.parallel_refiner_margin_floor,
        parallel_refiner_temperature=args.parallel_refiner_temperature,
        parallel_refiner_loss_budget=args.parallel_refiner_loss_budget,
        refiner_advantage_mode=args.refiner_advantage_mode,
        refiner_advantage_threshold=args.refiner_advantage_threshold,
        refiner_advantage_gate_alpha=args.refiner_advantage_gate_alpha,
        refiner_advantage_regression_alpha=(
            args.refiner_advantage_regression_alpha
        ),
        refiner_advantage_distill_alpha=args.refiner_advantage_distill_alpha,
        refiner_advantage_preservation_alpha=(
            args.refiner_advantage_preservation_alpha
        ),
        refiner_advantage_greedy_preservation_alpha=(
            args.refiner_advantage_greedy_preservation_alpha
        ),
        refiner_advantage_greedy_margin_floor=(
            args.refiner_advantage_greedy_margin_floor
        ),
        refiner_advantage_gate_distill_min_probability=(
            args.refiner_advantage_gate_distill_min_probability
        ),
        refiner_advantage_distill_top_k=(
            args.refiner_advantage_distill_top_k
        ),
        refiner_advantage_distill_temperature=(
            args.refiner_advantage_distill_temperature
        ),
        refiner_advantage_value_clip=args.refiner_advantage_value_clip,
        refiner_advantage_distill_start_ratio=(
            args.refiner_advantage_distill_start_ratio
        ),
        refiner_advantage_consolidation_ratio=(
            args.refiner_advantage_consolidation_ratio
        ),
        refiner_advantage_loss_budget=args.refiner_advantage_loss_budget,
        vat_enabled=args.vat_enabled,
        vat_hard_loss_alpha=args.vat_hard_loss_alpha,
        vat_soft_loss_alpha=args.vat_soft_loss_alpha,
        vat_verification_head_alpha=args.vat_verification_head_alpha,
        vat_post_rejection_decay_gamma=(
            args.vat_post_rejection_decay_gamma
        ),
        vat_simulation_temperature=args.vat_simulation_temperature,
        branch_value_mode=args.branch_value_mode,
        branch_value_top_m=args.branch_value_top_m,
        branch_value_horizon=args.branch_value_horizon,
        branch_value_alpha=args.branch_value_alpha,
        branch_value_temperature=args.branch_value_temperature,
        branch_value_loss_budget=args.branch_value_loss_budget,
        branch_value_warmup_ratio=args.branch_value_warmup_ratio,
        branch_value_consolidation_ratio=(
            args.branch_value_consolidation_ratio
        ),
        elastic_horizon_enabled=args.elastic_horizon_enabled,
        elastic_short_horizon=args.elastic_short_horizon,
        elastic_long_horizon=args.elastic_long_horizon,
        elastic_warmup_ratio=args.elastic_warmup_ratio,
        elastic_late_ratio=args.elastic_late_ratio,
        elastic_consolidation_ratio=args.elastic_consolidation_ratio,
        elastic_middle_long_prob=args.elastic_middle_long_prob,
        elastic_late_long_prob=args.elastic_late_long_prob,
        elastic_pair_probability=args.elastic_pair_probability,
        elastic_projective_num_anchors=args.elastic_projective_num_anchors,
        elastic_projective_alpha=args.elastic_projective_alpha,
        elastic_projective_top_k=args.elastic_projective_top_k,
        elastic_overlap_gap=args.elastic_overlap_gap,
        elastic_margin_gap=args.elastic_margin_gap,
        elastic_loss_budget=args.elastic_loss_budget,
    )
    dspark_model.configure_multi_teacher_oracle(
        mode=args.multi_teacher_oracle_mode,
        alpha=args.multi_teacher_oracle_alpha,
        advantage_threshold=args.multi_teacher_oracle_advantage_threshold,
        value_clip=args.multi_teacher_oracle_value_clip,
        loss_budget=args.multi_teacher_oracle_loss_budget,
    )
    if args.dspark_loss_type == "tv-acceptance":
        from specforge.core.tv_acceptance import configure_tv_acceptance
        from specforge.inference.target_engine.candidate_verifier import CandidatePrefixVerifier

        # Read the same target config for HF and SGLang, including multi-EOS models.
        eos_ids = getattr(AutoConfig.from_pretrained(
            args.target_model_path, trust_remote_code=args.trust_remote_code,
        ), "eos_token_id", tokenizer.eos_token_id)
        eos_ids = [] if eos_ids is None else ([eos_ids] if isinstance(eos_ids, int) else list(eos_ids))
        configure_tv_acceptance(
            dspark_model, temperature=args.tv_sampling_temperature,
            chunk_blocks=args.tv_objective_chunk_blocks, eos_token_ids=eos_ids,
            verifier=CandidatePrefixVerifier(
                target_model, batch_size=args.tv_verification_batch_size,
                pad_token_id=tokenizer.pad_token_id or 0,
            ),
        )
        print_on_rank0(
            "TV acceptance: fixed sampled prefixes, full-vocabulary softmax, "
            "loss=mean_blocks(1-sum(cumprod(1-TV))/K). "
            "CE/L1/confidence weights, PACE weights and loss-decay-gamma are unused."
        )

    trajectory_writer = None
    if args.multi_teacher_oracle_export_dir:
        trajectory_writer = MultiTeacherTrajectoryWriter(
            output_dir=args.multi_teacher_oracle_export_dir,
            teacher_name=args.multi_teacher_name,
            rank=dist.get_rank(),
            top_k=args.multi_teacher_export_top_k,
            flush_records=args.multi_teacher_export_flush_records,
        )
        dspark_model.multi_teacher_export_sink = trajectory_writer
    if args.multi_teacher_oracle_cache:
        oracle_cache = MultiTeacherOracleCache(args.multi_teacher_oracle_cache)
        dspark_model.multi_teacher_oracle_provider = oracle_cache.lookup

    if (
        args.netprefix_mode in ("fixed-greedy", "h1-greedy")
        or args.branch_value_mode != "none"
        or args.refiner_advantage_mode != "none"
        or args.multi_teacher_oracle_export_dir is not None
        or args.multi_teacher_oracle_cache is not None
        or max(
            args.on_policy_survival_alpha,
            args.on_policy_full_alpha,
            args.on_policy_deep_alpha,
            args.on_policy_distributional_alpha,
            args.on_policy_distributional_deep_alpha,
            args.on_policy_rejection_aligned_alpha,
            args.on_policy_mixed_kl_alpha,
            args.on_policy_clipped_rkl_alpha,
            args.on_policy_target_margin_alpha,
            args.on_policy_preservation_alpha,
            args.on_policy_regression_alpha,
            args.on_policy_greedy_preservation_alpha,
            args.on_policy_reset_replay_alpha,
            args.on_policy_preference_alpha,
            args.carh_gate_calibration_alpha,
            args.carh_gate_noop_alpha,
            args.branch_value_alpha,
            args.parallel_refiner_hazard_alpha,
            args.parallel_refiner_recovery_alpha,
            args.parallel_refiner_preservation_alpha,
        ) > 0
    ):
        @torch.no_grad()
        def score_on_policy_proposals(
            input_ids,
            attention_mask,
            anchors,
            proposals,
            active_rows,
            rollout_temperatures=None,
        ):
            """Write proposal paths into causal prefixes and score them.

            Multi-branch objectives (VAGRD and Branch Value) are deliberately
            evaluated one branch at a time. Flattening branches into the batch
            makes SGLang materialize a correspondingly larger full-vocabulary
            FP32 logits tensor and can add more than 10 GiB of peak memory.
            Serial scoring leaves labels unchanged while keeping the target
            forward at the same batch size as ordinary SR-OPSC.
            """
            if proposals.dim() == 3:
                branch_results = [
                    score_on_policy_proposals(
                        input_ids,
                        attention_mask,
                        anchors,
                        proposals[:, branch_index],
                        active_rows,
                        rollout_temperatures,
                    )
                    for branch_index in range(proposals.size(1))
                ]
                if isinstance(branch_results[0], tuple):
                    return tuple(
                        torch.stack(
                            [result[item_index] for result in branch_results],
                            dim=1,
                        )
                        for item_index in range(len(branch_results[0]))
                    )
                return torch.stack(branch_results, dim=1)
            if proposals.dim() != 2:
                raise ValueError(
                    "on-policy proposals must have shape [B,K] or [B,N,K]"
                )
            scorer_ids = input_ids.clone()
            seq_len = scorer_ids.size(1)
            depth_count = proposals.size(-1)
            depth = torch.arange(
                depth_count, device=input_ids.device
            ).view(1, -1)
            write_pos = anchors.unsqueeze(-1) + depth + 1
            in_bounds = write_pos.lt(seq_len) & active_rows.unsqueeze(-1)
            rows = torch.arange(input_ids.size(0), device=input_ids.device).view(-1, 1)
            scorer_ids[rows.expand_as(write_pos)[in_bounds], write_pos[in_bounds]] = (
                proposals[in_bounds]
            )
            score_pos = (anchors.unsqueeze(-1) + depth).clamp(max=seq_len - 1)
            if args.on_policy_distributional_top_k > 0:
                target_next_ids, target_topk_ids, target_topk_probs = (
                    target_model.score_proposal_distribution(
                        scorer_ids,
                        attention_mask,
                        args.on_policy_distributional_top_k,
                        temperatures=rollout_temperatures,
                        sampling_top_k=args.on_policy_deployment_top_k,
                        sampling_top_p=args.on_policy_deployment_top_p,
                        score_positions=score_pos,
                    )
                )
                result = (
                    target_next_ids,
                    target_topk_ids,
                    target_topk_probs,
                )
                return result
            target_next_ids = target_model.score_proposal_tokens(
                scorer_ids, attention_mask
            ).to(input_ids.device)
            result = target_next_ids.gather(1, score_pos)
            return result

        dspark_model.on_policy_scorer = score_on_policy_proposals

    netprefix_hook = None
    if args.on_policy_repair_value or args.fixed_prefix_reference_alpha:
        if draft_model.markov_head is None:
            raise ValueError("D/E requires CARH recurrence")
        for name in ("recall_correction", "parallel_refiner", "candidate_selector", "prefix_state_mixer"):
            if getattr(draft_model, name, None) is not None:
                raise ValueError(f"D/E v1 does not support {name}")
    if args.netprefix_mode != "off":
        if draft_model.markov_head is None:
            raise ValueError("NetPrefix v1 requires the CARH/Markov serving path")
        for name in ("recall_correction", "parallel_refiner", "candidate_selector", "prefix_state_mixer"):
            if getattr(draft_model, name, None) is not None:
                raise ValueError(f"NetPrefix v1 does not support {name}")
        if args.on_policy_distributional_top_k > 0:
            raise ValueError("NetPrefix v1 requires the exact greedy ID scorer, not top-k distribution scoring")
        netprefix_hook = GreedyPrefixHook(dspark_model)
        dspark_model.netprefix_hook = netprefix_hook

    # Capture full (unsharded) parameter shapes BEFORE FSDP wrapping so
    # the optimizer can convert saved state dicts between rank-agnostic
    # and FSDP-sharded formats.
    draft_full_param_shapes = [
        p.shape for p in draft_model.parameters() if p.requires_grad
    ]

    # Wrap each transformer block as its own FSDP unit (compute/comm overlap).
    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=configure_fsdp_optimizer_precision(dspark_model, args.optimizer),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in dspark_model.modules() if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_classes,
        )
    else:
        print_with_rank(
            "No _no_split_modules on draft model; falling back to single-unit "
            "FSDP wrap (no compute-comm overlap)."
        )
    dspark_model = FSDP(dspark_model, **fsdp_kwargs)
    print_with_rank("Initialized FSDP")

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]

    optimizer = build_bf16_optimizer(
        draft_model,
        optimizer_name=args.optimizer,
        fsdp_model=dspark_model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )
    optimizer.set_full_param_shapes(draft_full_param_shapes)

    if args.resume_extension_start_step is not None:
        boundary = args.resume_extension_start_step
        if (not args.resume or resume_state is None or args.optimizer != "adamw"
                or not resume_state.get("_fsdp_full_param_format")
                or args.accumulation_steps != 1 or args.num_epochs != 10
                or boundary != 6 * len(train_dataloader)
                or not boundary <= resume_state["global_step"] < total_steps
                or not 0 < args.resume_extension_lr < float("inf")):
            raise ValueError("6-to-10 extension requires AdamW resume, unchanged batches/epoch, accumulation1, valid LR and step in [6 epochs, 10 epochs)")
        # Restore Adam below, but reconstruct the extension scheduler explicitly.
        # Do not load an old completed cosine schedule or a saved LambdaLR into it.
        resume_state["scheduler_state_dict"] = optimizer.scheduler.state_dict()

    if resume_state is not None:
        # Detect whether the saved optimizer state is in the new
        # rank-agnostic (full-param) format or the legacy FSDP-sharded
        # format.  The new format is identified by the presence of a
        # "_fsdp_full_param_format" marker; lacking that, we fall back
        # to the old load_state_dict for backwards compatibility and
        # then immediately re-save in the new format.
        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if resume_state.get("_fsdp_full_param_format"):
            load_fsdp_optimizer_state(
                optimizer, dspark_model, training_state_path,
                restore_scheduler=args.resume_extension_start_step is None,
            )
        else:
            # Legacy checkpoint: optimizer state was saved in
            # FSDP-sharded (rank-dependent) format.  We can only safely
            # load it on the rank that saved it (rank 0).  For other
            # ranks the param shapes differ, causing a crash in Adam.
            # Fall back to loading only the scheduler (which has no
            # per-param tensors) and let Adam re-initialise on other
            # ranks — the loss of a few steps of momentum is acceptable.
            if dist.get_rank() == 0:
                optimizer.load_state_dict(resume_state)
            else:
                # Only restore the scheduler (lr schedule) on non-zero
                # ranks.  The optimizer Adam state (exp_avg / exp_avg_sq)
                # will start from scratch on these ranks — mild warm-up
                # penalty, but avoids the crash.
                optimizer.scheduler.load_state_dict(
                    resume_state["scheduler_state_dict"]
                )
                print_on_rank0(
                    "Legacy FSDP-sharded optimizer checkpoint: only "
                    "scheduler restored on non-rank-0; Adam state "
                    "re-initialised."
                )
            # Re-save in the new rank-agnostic format so future
            # resumes work correctly on all ranks.
            save_fsdp_optimizer_state(
                optimizer,
                dspark_model,
                training_state_path,
            )
            if dist.get_rank() == 0:
                saved = torch.load(
                    training_state_path, map_location="cpu", weights_only=False
                )
                saved["epoch"] = resume_state["epoch"]
                saved["global_step"] = resume_state["global_step"]
                saved["args"] = resume_state["args"]
                torch.save(saved, training_state_path)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Restored optimizer/scheduler state: "
            f"epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}"
        )

    if args.resume_extension_start_step is not None:
        from specforge.resume_extension import extension_lr_factor
        boundary = args.resume_extension_start_step
        elapsed = global_step - boundary
        duration = total_steps - boundary
        if len(optimizer.optimizer.param_groups) != 1:
            raise ValueError("Extension currently supports one AdamW parameter group only")
        for group in optimizer.optimizer.param_groups:
            group["lr"] = args.resume_extension_lr
            group["initial_lr"] = args.resume_extension_lr
        optimizer.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer.optimizer,
            lambda step: extension_lr_factor(step, duration),
            last_epoch=elapsed - 1,
        )
        print_on_rank0(f"Extension schedule: start={boundary}, end={total_steps}, elapsed={elapsed}, peak_lr={args.resume_extension_lr}, current_lr={optimizer.get_learning_rate()}")

    # Derive the data cursor from consumed batches, including a mid-epoch pilot stop.
    if args.netprefix_mode != "off" or args.stop_after_steps is not None:
        start_epoch, skip_steps = divmod(global_step, len(train_dataloader))
    else:
        skip_steps = global_step - start_epoch * len(train_dataloader)

    reference_runtime = None
    if args.fixed_prefix_reference_alpha:
        reference_runtime = FixedPrefixReference(
            args, dspark_model, train_dataloader,
            lambda batch, seed: _forward_dspark_data_batch(
                batch, args, device, target_model, dspark_model, needs_target_hidden, seed),
        )
        dspark_model.fixed_prefix_reference_runtime = reference_runtime
        if args.resume:
            if not draft_model_last_checkpoint:
                raise ValueError("E resume requires a complete same-arm checkpoint")
            reference_runtime.load(draft_model_last_checkpoint, global_step)

    netprefix_runtime = None
    if args.netprefix_mode != "off":
        setup_rng = rng_state()
        control_batches, audit_loader = [], None
        if args.netprefix_mode in ("fixed-greedy", "h1-greedy"):
            def heldout_loader(path):
                dataset = load_dataset("json", data_files=path)["train"]
                processed = build_eagle3_dataset(
                    dataset=dataset, tokenizer=tokenizer, chat_template=args.chat_template,
                    max_length=args.max_length, is_preformatted=args.is_preformatted,
                )
                processed = processed.add_column("__sample_id", list(range(len(processed))))
                # Avoid DistributedSampler padding duplicated audit examples.
                multiple = dist.get_world_size(get_dp_group()) * args.batch_size
                usable = len(processed) // multiple * multiple
                if usable == 0:
                    raise ValueError(f"NetPrefix pool {path} has no complete global batch")
                processed = processed.select(range(usable))
                return prepare_dp_dataloaders(
                    processed, args.batch_size, num_workers=0, shuffle=False,
                    process_group=get_dp_group(),
                )
            if args.netprefix_data_source == "train-stream":
                audit_loader = TrainingStreamSource(
                    train_dataloader.dataset, train_dataloader.collate_fn,
                    args.batch_size, dist.get_rank(), dist.get_world_size(),
                    args.netprefix_control_batches, args.seed + 1700000,
                )
                control_batches = audit_loader.control_batches()
            else:
                control_loader = heldout_loader(args.netprefix_control_data_path)
                audit_loader = heldout_loader(args.netprefix_audit_data_path)
                import itertools
                control_batches = list(itertools.islice(control_loader, args.netprefix_control_batches))
                if len(control_batches) != args.netprefix_control_batches or len(audit_loader) == 0:
                    raise ValueError("NetPrefix held-out datasets are too small")
        netprefix_runtime = NetPrefixRuntime(
            args, dspark_model, optimizer, netprefix_hook,
            lambda batch, seed: _forward_dspark_data_batch(
                batch, args, device, target_model, dspark_model, needs_target_hidden, seed,
            ), control_batches, audit_loader,
        )
        dspark_model.netprefix_runtime = netprefix_runtime
        restore_rng(setup_rng)
        if args.resume:
            if not draft_model_last_checkpoint:
                raise ValueError("NetPrefix --resume did not find a checkpoint")
            netprefix_runtime.load(draft_model_last_checkpoint, global_step)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")
    stop = ((args.stop_after_steps is not None and global_step >= args.stop_after_steps)
            or (args.max_steps is not None and global_step >= args.max_steps))

    for epoch in range(start_epoch, args.num_epochs):
        if stop:
            break
        train_dataloader.sampler.set_epoch(epoch)
        if args.multi_teacher_oracle_export_dir:
            draft_model.eval()
        else:
            draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            # PACE mixture scheduling uses optimizer-step units, matching the
            # optimizer warmup and DFlash auxiliary schedules.
            optimizer_step = (
                global_step + args.accumulation_steps - 1
            ) // args.accumulation_steps
            dspark_model.set_training_progress(optimizer_step, total_steps)
            transition_conv_scale = draft_model.set_local_transition_progress(
                optimizer_step, total_steps
            )

            if netprefix_runtime is not None:
                (loss, accuracy, loss_per_position, acc_per_position,
                 count_per_position, loss_components) = netprefix_runtime.step(data, global_step)
                loss_components.update(netprefix_runtime.metrics(loss.device))
            elif args.micro_batch_size:
                loss, accuracy, loss_components = forward_backward_microbatches(
                    data,
                    args.micro_batch_size,
                    lambda micro, index: _forward_dspark_data_batch(
                        micro, args, device, target_model, dspark_model,
                        needs_target_hidden, global_step, micro_index=index,
                    ),
                )
            else:
                if reference_runtime is not None:
                    reference_runtime.before_step(global_step)
                (
                    loss,
                    accuracy,
                    loss_per_position,
                    acc_per_position,
                    count_per_position,
                    loss_components,
                ) = _forward_dspark_data_batch(
                    data, args, device, target_model, dspark_model,
                    needs_target_hidden, global_step,
                )

            if args.multi_teacher_oracle_export_dir:
                if global_step % args.log_interval == 0:
                    print_on_rank0(
                        f"Multi-teacher export step {global_step}: "
                        f"records/rank={trajectory_writer.count}"
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break
                continue

            if netprefix_runtime is None and not args.micro_batch_size:
                (loss / args.accumulation_steps).backward()
                if reference_runtime is not None:
                    loss_components.update(reference_runtime.backward(data, global_step))

            if netprefix_runtime is None and global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                # Average over the DP group (not the full world, which includes
                # TP ranks that hold redundant copies).
                dp_group = get_dp_group()
                dp_size = dist.get_world_size(dp_group)

                loss_log = loss.clone()
                acc_log = accuracy.clone()
                if dp_group is not None:
                    dist.all_reduce(loss_log, group=dp_group)
                    dist.all_reduce(acc_log, group=dp_group)
                else:
                    dist.all_reduce(loss_log)
                    dist.all_reduce(acc_log)
                loss_log = loss_log / dp_size
                acc_log = acc_log / dp_size

                # Optional objectives can legitimately emit rank-local
                # diagnostics (for example, one rank samples a useful
                # multi-rollout preference while another does not).  Reducing
                # each rank's raw dict directly gives NCCL a different number
                # of collectives and deadlocks the following FSDP forward.
                # Agree on a sorted union first, then contribute zero for a
                # locally absent diagnostic so every rank executes the exact
                # same collective sequence.
                local_component_keys = sorted(loss_components)
                gathered_component_keys = [None] * dp_size
                dist.all_gather_object(
                    gathered_component_keys,
                    local_component_keys,
                    group=dp_group,
                )
                component_keys = sorted(
                    {
                        key
                        for rank_keys in gathered_component_keys
                        for key in rank_keys
                    }
                )
                comp_log = {}
                for key in component_keys:
                    value = loss_components.get(key)
                    v = (
                        value.clone().float()
                        if value is not None
                        else torch.zeros((), device=loss.device, dtype=torch.float32)
                    )
                    if dp_group is not None:
                        dist.all_reduce(v, group=dp_group)
                    else:
                        dist.all_reduce(v)
                    comp_log[key] = (v / dp_size).item()
                if draft_model.local_transition_enabled:
                    comp_log["local_transition_conv_scale"] = transition_conv_scale

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    comp_log,
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, dspark_model, draft_model, optimizer
                )

            if args.stop_after_steps is not None and global_step >= args.stop_after_steps:
                print_on_rank0(f"Reached pilot stop={args.stop_after_steps}; full schedule unchanged.")
                stop = True
                break
            if args.max_steps is not None and global_step >= args.max_steps:
                print_on_rank0(f"Reached max_steps={args.max_steps}; stopping.")
                stop = True
                break

    if trajectory_writer is not None:
        trajectory_writer.close()
        dist.barrier()
        print_on_rank0(
            "Multi-teacher trajectory export complete: "
            f"{args.multi_teacher_oracle_export_dir}"
        )
    else:
        save_checkpoint(
            args, (global_step // len(train_dataloader)
                   if args.netprefix_mode != "off" or args.stop_after_steps is not None else args.num_epochs),
            global_step, dspark_model, draft_model, optimizer
        )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
