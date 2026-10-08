#!/usr/bin/env bash
set -euo pipefail

# Run from this checkout, including /ossfs/workspace/SpecForge on the server.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"
CONFIG="$REPO_ROOT/examples/on_policy/qwen3-8b-dspark-tv.yaml"

export SPECFORGE_DATA_NUM_PROC=32
export FLASHINFER_DISABLE_VERSION_CHECK=1
# Physical GPU 0 runs frozen target + draft rollout. Seven GPUs run FSDP.
# The effective global batch remains 4 * 8 = 32, independent of this split.
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

if [[ "${1:-}" == "--plan" && "$#" == 1 ]]; then
    exec python -m specforge.on_policy --config "$CONFIG" --plan
fi
if [[ "$#" != 0 ]]; then
    echo "Usage: bash scripts/train_dspark_on_policy_8gpu.sh [--plan]" >&2
    exit 2
fi

# Requires sglang==0.5.18 with patches/sglang/v0.5.18/on-policy.patch applied.
# Do not run the old 8-rank trainer alongside this job on the same devices.
exec torchrun --standalone --nproc_per_node=7 \
    -m specforge.on_policy --config "$CONFIG"
