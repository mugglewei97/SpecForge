#!/usr/bin/env bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

cd /ossfs/workspace/SpecForge || exit 1

export SPECFORGE_DATA_NUM_PROC=100
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}
TARGET_BACKEND=${3:-sglang}

# Warm start native DFlash2 with D-PACE + selector and a fresh optimizer.
# block_size=7 includes 1 anchor + 6 predictions. D-PACE ignores loss-decay-gamma.
torchrun --standalone --nproc_per_node 8 scripts/train_dspark.py \
    --target-model-path /mnt/amed-s1/common/ckpt/gaochang/Qwen3-8B \
    --draft-config-path ./configs/qwen3-8b-dflash2.json \
    --train-data-path /mnt/amed-s1/common/data/zhonglv.zl/datasets/qwen3_8b_regenerate_open-perfectblend_20260929_regenerate_nothink.jsonl \
    --output-dir /mnt/amed-s1/common/ckpt/yilei.wm/lab/dpace-dflash2-lr2e4-perfectblend-4epochs \
    --init-draft-model-path /mnt/amed-s1/common/ckpt/gaochang/EagleModel/qwen3-8b-dflash2-perfectblend-4096-nothinking/epoch_6_step_324000 \
    --num-epochs 4 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen \
    --attention-backend "$ATTENTION_BACKEND" \
    --loss-decay-gamma 4.0 \
    --log-interval 50 \
    --save-interval 2000 \
    --target-model-backend "$TARGET_BACKEND" \
    --block-size 7 \
    --num-anchors 512 \
    --sglang-mem-fraction-static 0.3 \
    --dspark-loss-type dpace \
    --dpace-alpha 0.5
