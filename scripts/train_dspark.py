#!/usr/bin/env python3
# coding=utf-8
"""Train DSpark with the legacy CE/L1 or sampled-prefix TV acceptance objective."""

from specforge.legacy.dspark_training.arguments import parse_args


def main(argv=None):
    args = parse_args(argv)
    # Parse/help stays usable without importing the GPU training stack.
    from specforge.legacy.dspark_training.trainer import run_training

    run_training(args)


if __name__ == "__main__":
    main()
