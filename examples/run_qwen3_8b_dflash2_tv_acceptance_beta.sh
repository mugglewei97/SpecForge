#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SPECFORGE_DATA_NUM_PROC="${SPECFORGE_DATA_NUM_PROC:-32}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS="${1:-8}"
ATTENTION_BACKEND="${2:-flex_attention}"
TARGET_BACKEND="${3:-sglang}"
shift "$(( $# < 3 ? $# : 3 ))"

# A warm start must be a native Qwen3-8B DFlash2 checkpoint with block_size=8.
# Without --init-draft-model-path this recipe initializes a new DFlash2 draft.
exec torchrun --standalone --nproc_per_node "$NUM_GPUS" scripts/train_dspark.py \
    --target-model-path "${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}" \
    --draft-config-path ./configs/qwen3-8b-dflash2.json \
    --train-data-path "${TRAIN_DATA_PATH:-cache/dataset/qwen3-8b-regen.jsonl}" \
    --output-dir "${OUTPUT_DIR:-outputs/qwen3-8b-dflash2-tv-beta}" \
    --num-epochs 1 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen \
    --attention-backend "$ATTENTION_BACKEND" \
    --target-model-backend "$TARGET_BACKEND" \
    --block-size 8 \
    --num-anchors 512 \
    --build-dataset-num-proc "$SPECFORGE_DATA_NUM_PROC" \
    --sglang-mem-fraction-static 0.3 \
    --dspark-loss-type tv-acceptance-beta \
    --tv-temperature 1.0 \
    --tv-objective-chunk-blocks 8 \
    --log-interval 10 \
    --save-interval 1000 \
    "$@"
