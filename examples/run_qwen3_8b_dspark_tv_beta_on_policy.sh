#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SPECFORGE_DATA_NUM_PROC="${SPECFORGE_DATA_NUM_PROC:-32}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH to the target matching the DSpark checkpoint}"
: "${INIT_DRAFT_MODEL_PATH:?Set INIT_DRAFT_MODEL_PATH to the original DSpark checkpoint}"
: "${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to the reference JSONL used to extract prompts}"

NUM_GPUS="${1:-8}"
ATTENTION_BACKEND="${2:-flex_attention}"
TARGET_BACKEND="${3:-sglang}"
shift "$(( $# < 3 ? $# : 3 ))"

# The learned DSpark checkpoint supplies its full config, including CARH/Markov
# and convolution settings. Start a new optimizer and use a fresh output folder.
exec torchrun --standalone --nproc_per_node "$NUM_GPUS" scripts/train_dspark.py \
    --target-model-path "$TARGET_MODEL_PATH" \
    --init-draft-model-path "$INIT_DRAFT_MODEL_PATH" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "${OUTPUT_DIR:-outputs/qwen3-8b-dspark-tv-beta-onpolicy}" \
    --num-epochs "${NUM_EPOCHS:-1}" \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen3-instruct \
    --attention-backend "$ATTENTION_BACKEND" \
    --target-model-backend "$TARGET_BACKEND" \
    --block-size 7 \
    --num-anchors 512 \
    --build-dataset-num-proc "$SPECFORGE_DATA_NUM_PROC" \
    --sglang-mem-fraction-static 0.3 \
    --dspark-loss-type tv-acceptance-beta \
    --tv-anchor-mode on-policy \
    --tv-rollout-max-tokens "${TV_ROLLOUT_MAX_TOKENS:-256}" \
    --tv-temperature "${TV_TEMPERATURE:-1.0}" \
    --tv-objective-chunk-blocks 8 \
    --report-to wandb \
    --wandb-offline \
    --log-interval 10 \
    --save-interval 1000 \
    "$@"
