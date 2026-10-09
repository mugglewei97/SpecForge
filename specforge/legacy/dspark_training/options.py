"""Argument declarations for optional legacy training experiments."""


def add_muon_optimizer_args(group):
    group.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    group.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay, including the AdamW group in Muon mode.",
    )
    group.add_argument(
        "--muon-lr",
        type=float,
        default=None,
        help="Muon LR; defaults to 10 * --learning-rate, as in speculators CLI.",
    )
    group.add_argument("--muon-momentum", type=float, default=0.95)
    group.add_argument("--muon-weight-decay", type=float, default=0.1)
    group.add_argument("--muon-ns-steps", type=int, default=5)
    group.add_argument(
        "--muon-adjust-lr-fn",
        choices=["original", "match_rms_adamw"],
        default="match_rms_adamw",
    )


def add_predictive_auxiliary_args(parser):
    group = parser.add_argument_group("training-only predictive supervision")
    group.add_argument("--conv-source-semantic-alpha", type=float, default=0.0)
    group.add_argument("--carh-reference-calibration-alpha", type=float, default=0.0)
    group.add_argument("--predictive-aux-warmup-ratio", type=float, default=0.10)
    group.add_argument("--predictive-aux-ramp-ratio", type=float, default=0.15)
    group.add_argument("--source-semantic-codebook-seed", type=int, default=20261001)


def add_bv_args(parser):
    group = parser.add_argument_group("BV distribution-loss replacement")
    group.add_argument(
        "--bv-loss-alpha",
        type=float,
        default=0.0,
        help="BV weight; positive requires l1-loss-alpha=0.",
    )
    group.add_argument("--bv-temperature", type=float, default=1.0)
    group.add_argument(
        "--bv-anneal-ratio",
        type=float,
        default=0.5,
        help="Fraction of optimizer steps annealing beta from 0 to 1.",
    )
    group.add_argument(
        "--bv-block-chunk-size",
        type=int,
        default=8,
        help="Blocks per recomputed full-vocabulary BV chunk.",
    )


def add_reference_args(parser):
    group = parser.add_argument_group("Repair value D / fixed reference E")
    group.add_argument("--on-policy-repair-value", action="store_true")
    group.add_argument("--on-policy-repair-value-horizon", type=int, default=2)
    group.add_argument("--fixed-prefix-reference-alpha", type=float, default=0.0)
    group.add_argument("--fixed-prefix-reference-batches", type=int, default=4)
    group.add_argument("--fixed-prefix-reference-interval", type=int, default=32)


def add_netprefix_args(parser):
    g = parser.add_argument_group("NetPrefix H1 greedy pilot")
    g.add_argument(
        "--netprefix-mode",
        choices=["off", "baseline", "fixed-greedy", "h1-greedy"],
        default="off",
    )
    g.add_argument("--netprefix-ramp-steps", type=int, default=0)
    g.add_argument("--netprefix-start-step", type=int, default=20000)
    g.add_argument("--netprefix-interval", type=int, default=128)
    g.add_argument(
        "--netprefix-repair-weights", type=float, nargs="+", default=[0.05, 0.1]
    )
    g.add_argument("--netprefix-protect-weight", type=float, default=0.1)
    g.add_argument("--netprefix-control-data-path")
    g.add_argument("--netprefix-audit-data-path")
    g.add_argument(
        "--netprefix-data-source", choices=["files", "train-stream"], default="files"
    )
    g.add_argument("--netprefix-control-batches", type=int, default=4)
    g.add_argument("--netprefix-extra-time-ratio", type=float, default=0.2)
    g.add_argument("--netprefix-min-gain", type=float, default=0.0)
    g.add_argument("--netprefix-audit-tolerance", type=float, default=0.0)
    g.add_argument(
        "--stop-after-steps",
        type=int,
        help="Pilot stop; does not shorten LR/auxiliary schedules.",
    )
