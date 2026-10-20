#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SPECFORGE_DATA_NUM_PROC="${SPECFORGE_DATA_NUM_PROC:-100}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS="${1:-8}"
ATTENTION_BACKEND="${2:-flex_attention}"
TARGET_BACKEND="${3:-sglang}"
shift "$(( $# < 3 ? $# : 3 ))"

# Start a new optimizer from the epoch-5 native DFlash2 checkpoint. The new
# output directory keeps this run separate from random-anchor beta training.
exec torchrun --standalone --nproc_per_node "$NUM_GPUS" scripts/train_dspark.py \
    --target-model-path "${TARGET_MODEL_PATH:-/mnt/amed-s1/common/ckpt/gaochang/Qwen3-8B}" \
    --draft-config-path ./configs/qwen3-8b-dflash2.json \
    --init-draft-model-path "${INIT_DRAFT_MODEL_PATH:-/mnt/amed-s1/common/ckpt/gaochang/EagleModel/qwen3-8b-dflash2-perfectblend-4096-nothinking/epoch_5_step_322000}" \
    --train-data-path "${TRAIN_DATA_PATH:-/mnt/amed-s1/common/data/zhonglv.zl/datasets/qwen3_8b_regenerate_open-perfectblend_20260929_regenerate_nothink.jsonl}" \
    --output-dir "${OUTPUT_DIR:-/mnt/amed-s1/common/ckpt/yilei.wm/lab/tvbeta-dflash2-lr2e4-perfectblend-fromEpoch5-onpolicy}" \
    --num-epochs "${NUM_EPOCHS:-1}" \
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
    --tv-anchor-mode on-policy \
    --tv-rollout-max-tokens "${TV_ROLLOUT_MAX_TOKENS:-256}" \
    --tv-temperature "${TV_TEMPERATURE:-1.0}" \
    --tv-objective-chunk-blocks 8 \
    --report-to wandb \
    --wandb-offline \
    --log-interval 10 \
    --save-interval 1000 \
    "$@"
