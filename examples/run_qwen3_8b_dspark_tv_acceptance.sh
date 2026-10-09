#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SPECFORGE_DATA_NUM_PROC=32
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS="${1:-8}"
ATTENTION_BACKEND="${2:-flex_attention}"
TARGET_BACKEND="${3:-sglang}"
# Remaining arguments override recipe defaults or select a warm-start checkpoint.
shift "$(( $# < 3 ? $# : 3 ))"

exec torchrun --standalone --nproc_per_node "$NUM_GPUS" scripts/train_dspark.py \
    --target-model-path /mnt/amed-s1/common/ckpt/gaochang/Qwen3-8B \
    --draft-config-path ./configs/qwen3-8b-dspark.json \
    --train-data-path /mnt/amed-s1/common/data/gaochang/eagle-data/qwen3-8b-regen-mix-70w.jsonl \
    --output-dir /mnt/amed-s1/common/ckpt/gaochang/EagleModel/outputs/qwen3-8b-dspark-tv-acceptance/ \
    --num-epochs 6 \
    --batch-size 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen \
    --attention-backend "$ATTENTION_BACKEND" \
    --log-interval 50 \
    --save-interval 2000 \
    --target-model-backend "$TARGET_BACKEND" \
    --block-size 7 \
    --num-anchors 512 \
    --build-dataset-num-proc 32 \
    --sglang-mem-fraction-static 0.3 \
    --dspark-loss-type tv-acceptance \
    --tv-temperature 1.0 \
    --tv-objective-chunk-blocks 8 \
    "$@"
