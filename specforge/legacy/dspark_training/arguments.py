"""Command-line interface for legacy DSpark training."""

import argparse
import os
from typing import Tuple

from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.legacy.dspark_training.options import (
    add_bv_args,
    add_muon_optimizer_args,
    add_netprefix_args,
    add_predictive_auxiliary_args,
    add_reference_args,
)

from .objective import add_objective_args


def _comma_separated_floats(value: str) -> Tuple[float, ...]:
    """Parse a non-empty comma-separated float schedule for argparse."""
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
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


def build_parser() -> argparse.ArgumentParser:
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
    add_objective_args(model_group)
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
        "--carh-predecessor-count",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="CARH token window; extra lag-2/lag-3 projections are zero-initialized.",
    )
    dspark_group.add_argument(
        "--carh-predecessor-context-mode",
        choices=[
            "none",
            "innovation",
            "context",
            "innovation-position",
            "innovation-residual",
            "innovation-residual-position",
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
        "--carh-sampled-prefix-memory-rank",
        type=int,
        default=None,
        help="Block-local sampled-prefix memory width for pred1 CARH (0 disables).",
    )
    dspark_group.add_argument(
        "--carh-sampled-prefix-memory-start-ratio",
        type=float,
        default=0.25,
        help="Fraction of optimizer steps before the sampled-prefix branch activates.",
    )
    dspark_group.add_argument(
        "--carh-sampled-prefix-memory-ramp-ratio",
        type=float,
        default=0.10,
        help="Fraction of optimizer steps used to ramp memory output from 0 to 1.",
    )
    dspark_group.add_argument(
        "--carh-predecessor-diagnostics-interval",
        type=int,
        default=0,
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
    dspark_group.add_argument("--parallel-refiner-gate-bias", type=float, default=None)
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
    dspark_group.add_argument(
        "--block-summary-rank",
        type=int,
        default=None,
        help="Optional anchor-hidden residual width after the draft backbone; 0 disables.",
    )
    dspark_group.add_argument(
        "--block-summary-source",
        choices=["anchor", "position"],
        default=None,
        help="Source for the matched low-rank residual: block anchor or each position's own hidden state.",
    )
    dspark_group.add_argument(
        "--block-summary-gate-mode",
        choices=["position", "compatibility"],
        default=None,
        help="Position-only gate or anchor/current compatibility gate for the block summary.",
    )
    dspark_group.add_argument(
        "--block-summary-counterfactual-alpha",
        type=float,
        default=0.0,
        help="Training-only first-error repair and surviving-prefix protection proxy weight.",
    )
    dspark_group.add_argument("--prefix-state-retention-bias", type=float, default=None)
    dspark_group.add_argument("--prefix-state-update-bias", type=float, default=None)
    dspark_group.add_argument("--prefix-state-gate-bias", type=float, default=None)
    dspark_group.add_argument("--prefix-state-residual-scale", type=float, default=None)
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
        choices=[
            "input",
            "output",
            "source-aware",
            "grouped16",
            "grouped64",
            "low-rank64",
            "static",
        ],
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
    dspark_group.add_argument("--recall-correction-gate-bias", type=float, default=None)
    dspark_group.add_argument("--recall-correction-alpha", type=float, default=0.0)
    dspark_group.add_argument("--recall-partition-top-k", type=int, default=16)
    dspark_group.add_argument(
        "--recall-correction-loss-budget", type=float, default=0.08
    )
    dspark_group.add_argument(
        "--target-layer-fusion-mode",
        choices=[
            "none",
            "per-draft-layer",
            "shared-low-rank",
            "shared-low-rank-delta",
            "static-channel",
            "shared-low-rank-grouped",
            "shared-low-rank-token",
        ],
        default=None,
    )
    dspark_group.add_argument(
        "--target-layer-fusion-residual-scale", type=float, default=None
    )
    dspark_group.add_argument("--target-layer-fusion-rank", type=int, default=None)
    dspark_group.add_argument(
        "--target-layer-fusion-source-dropout", type=float, default=None
    )
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
        "--pace-residual-min",
        type=float,
        default=0.9,
        help="Minimum residual multiplier on the auxiliary CE weights.",
    )
    dspark_group.add_argument(
        "--pace-residual-max",
        type=float,
        default=1.1,
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
    dspark_group.add_argument("--prefix-credit-warmup-ratio", type=float, default=0.10)
    dspark_group.add_argument("--prefix-credit-ramp-ratio", type=float, default=0.15)
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
    dspark_group.add_argument("--transition-credit-max-depth", type=int, default=3)
    dspark_group.add_argument("--transition-credit-margin", type=float, default=0.0)
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
    dspark_group.add_argument("--transition2-margin-floor", type=float, default=0.5)
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
    dspark_group.add_argument("--prefix-bottleneck-max-depth", type=int, default=3)
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
    dspark_group.add_argument("--deep-survival-guard-min-prefix", type=int, default=5)
    dspark_group.add_argument("--deep-survival-guard-start-depth", type=int, default=4)
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
    dspark_group.add_argument("--carh-preservation-margin", type=float, default=0.0)
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
        "--on-policy-survival-alpha",
        type=float,
        default=0.0,
        help="Weight for target-scored accepted-prefix and first-rejection credit.",
    )
    dspark_group.add_argument(
        "--on-policy-full-alpha",
        type=float,
        default=0.0,
        help="Margin-floor protection weight for fully survived on-policy paths.",
    )
    dspark_group.add_argument(
        "--on-policy-first-rejection-max-depth",
        type=int,
        default=2,
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
    dspark_group.add_argument("--on-policy-flatness-power-max", type=float, default=1.0)
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
        help=("Mass-preserving punctuation and Han/Latin boundary weighting for T2CM."),
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
    dspark_group.add_argument("--on-policy-clipped-rkl-clip", type=float, default=0.01)
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
    dspark_group.add_argument("--on-policy-target-margin-min", type=float, default=0.05)
    dspark_group.add_argument("--on-policy-target-margin-max", type=float, default=1.0)
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
    dspark_group.add_argument("--on-policy-hazard-ema-decay", type=float, default=0.99)
    dspark_group.add_argument("--on-policy-hazard-weight-min", type=float, default=0.5)
    dspark_group.add_argument("--on-policy-hazard-weight-max", type=float, default=2.0)
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
        "--on-policy-mix-ratio-max",
        type=float,
        default=0.5,
        help="Maximum scheduled mixture assigned to real proposal-prefix credit.",
    )
    dspark_group.add_argument(
        "--on-policy-start-step",
        type=int,
        default=None,
        help="Optional absolute optimizer-step start; independent of Prefix-Full warmup.",
    )
    dspark_group.add_argument("--on-policy-ramp-steps", type=int, default=2000)
    dspark_group.add_argument(
        "--on-policy-interval",
        type=int,
        default=4,
        help="Run the extra frozen-target proposal scoring pass every N steps.",
    )
    dspark_group.add_argument(
        "--on-policy-loss-budget",
        type=float,
        default=0.10,
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
    dspark_group.add_argument(
        "--multi-teacher-export-flush-records", type=int, default=4096
    )
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
    dspark_group.add_argument(
        "--multi-teacher-oracle-value-clip", type=float, default=7.0
    )
    dspark_group.add_argument(
        "--multi-teacher-oracle-loss-budget", type=float, default=0.05
    )
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
    dspark_group.add_argument("--on-policy-regression-margin", type=float, default=0.0)
    dspark_group.add_argument(
        "--on-policy-greedy-preservation-alpha", type=float, default=0.0
    )
    dspark_group.add_argument(
        "--on-policy-greedy-preservation-margin-floor", type=float, default=0.5
    )
    dspark_group.add_argument(
        "--on-policy-greedy-preservation-temperature", type=float, default=0.25
    )
    dspark_group.add_argument("--on-policy-reset-replay-alpha", type=float, default=0.0)
    dspark_group.add_argument("--on-policy-reset-replay-horizon", type=int, default=3)
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
    dspark_group.add_argument("--on-policy-preference-alpha", type=float, default=0.0)
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
    dspark_group.add_argument("--on-policy-pareto-ema-decay", type=float, default=0.95)
    dspark_group.add_argument(
        "--on-policy-pareto-temperature", type=float, default=0.25
    )
    dspark_group.add_argument("--on-policy-pareto-weight-min", type=float, default=0.5)
    dspark_group.add_argument("--on-policy-pareto-weight-max", type=float, default=2.0)
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
    dspark_group.add_argument(
        "--refiner-advantage-gate-alpha", type=float, default=0.005
    )
    dspark_group.add_argument(
        "--refiner-advantage-regression-alpha", type=float, default=0.0025
    )
    dspark_group.add_argument(
        "--refiner-advantage-distill-alpha", type=float, default=0.005
    )
    dspark_group.add_argument(
        "--refiner-advantage-preservation-alpha", type=float, default=0.0025
    )
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
    dspark_group.add_argument(
        "--refiner-advantage-distill-temperature", type=float, default=1.0
    )
    dspark_group.add_argument("--refiner-advantage-value-clip", type=float, default=7.0)
    dspark_group.add_argument(
        "--refiner-advantage-distill-start-ratio", type=float, default=0.20
    )
    dspark_group.add_argument(
        "--refiner-advantage-consolidation-ratio", type=float, default=0.15
    )
    dspark_group.add_argument(
        "--refiner-advantage-loss-budget", type=float, default=0.05
    )
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
    dspark_group.add_argument("--vat-verification-head-alpha", type=float, default=1.0)
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
    dspark_group.add_argument("--branch-value-temperature", type=float, default=0.5)
    dspark_group.add_argument("--branch-value-loss-budget", type=float, default=0.05)
    dspark_group.add_argument("--branch-value-warmup-ratio", type=float, default=0.10)
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
    dspark_group.add_argument("--elastic-consolidation-ratio", type=float, default=0.15)
    dspark_group.add_argument("--elastic-middle-long-prob", type=float, default=0.30)
    dspark_group.add_argument("--elastic-late-long-prob", type=float, default=0.50)
    dspark_group.add_argument("--elastic-pair-probability", type=float, default=0.25)
    dspark_group.add_argument("--elastic-projective-num-anchors", type=int, default=32)
    dspark_group.add_argument("--elastic-projective-alpha", type=float, default=0.0025)
    dspark_group.add_argument("--elastic-projective-top-k", type=int, default=64)
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
        "--micro-batch-size",
        type=int,
        default=0,
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
            "Post-train from a local DSpark checkpoint directory (config.json and model weights). "
            "Uses its architecture and mask token, with a fresh optimizer/scheduler and step 0. "
            "Overrides --draft-config-path; mutually exclusive with --resume."
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
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)
