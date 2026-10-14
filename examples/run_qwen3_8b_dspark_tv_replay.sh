#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SPECFORGE_DATA_NUM_PROC="${SPECFORGE_DATA_NUM_PROC:-100}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS="${1:-8}"
ATTENTION_BACKEND="${2:-sdpa}"
shift "$(( $# < 2 ? $# : 2 ))"

# Read-only check of the installed server sources and required trace patch.
# Install once with: python -m specforge.inference.dspark_replay_patch --apply
python -c 'from specforge.inference.dspark_replay_patch import inspect_patch, installed_root; inspect_patch(installed_root(), require=True)'

exec torchrun --standalone --nproc_per_node "$NUM_GPUS" scripts/train_dspark.py \
    --target-model-path /mnt/amed-s1/common/ckpt/gaochang/Qwen3-8B \
    --rollout-target-version Qwen3-8B-frozen-local-checkpoint \
    --draft-config-path ./configs/qwen3-8b-dspark.json \
    --train-data-path /mnt/amed-s1/common/data/gaochang/eagle-data/qwen3-8b-regen-mix-70w.jsonl \
    --output-dir /mnt/amed-s1/common/ckpt/yilei.wm/tv-replay-lr3e4 \
    --init-draft-model-path /mnt/amed-s1/common/ckpt/gaochang/EagleModel/outputs/qwen3-8b-dspark-120w-mix/epoch_6_step_228768 \
    --num-epochs 1 \
    --batch-size 4 \
    --learning-rate 3e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen \
    --attention-backend "$ATTENTION_BACKEND" \
    --log-interval 10 \
    --save-interval 1000 \
    --dspark-loss-type tv-acceptance \
    --tv-prefix-mode proposal \
    --replay-tensor-transport cuda-ipc \
    --anchor-mode replay \
    --target-model-backend sglang \
    --block-size 7 \
    --num-anchors 512 \
    --rollout-max-new-tokens 1024 \
    --rollout-temperature 1.0 \
    --tv-temperature 1.0 \
    --sglang-mem-fraction-static 0.3 \
    --build-dataset-num-proc "$SPECFORGE_DATA_NUM_PROC" \
    --report-to wandb --wandb-offline \
    "$@"
