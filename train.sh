#!/bin/bash

eval "$(conda shell.bash hook)"
# export CUDA_VISIBLE_DEVICES=1
conda activate pytorch-from-scratch

export ACCELERATE_LOG_LEVEL="INFO"

# DEBUG
# rm -rf /media/bryan/ssd01/expr/detr_from_scratch/debug01 
# accelerate launch --gpu_ids 0, --num_processes 1 detr/train.py \
# --output-dir /media/bryan/ssd01/expr/detr_from_scratch/debug01 \
# --train-batch-size 4 --val-batch-size 16 \
# --epochs 100 --lr-warmup-epochs 5  --lr-hold-max-epochs 35 --limit-train-iters 32  --limit-val-iters 20

# rm -rf /media/bryan/ssd01/expr/detr_from_scratch/debug02
# accelerate launch --gpu_ids 0, --num_processes 1 detr/train.py \
# --output-dir /media/bryan/ssd01/expr/detr_from_scratch/debug02 \
# --resume-from-checkpoint /media/bryan/ssd01/expr/detr_from_scratch/debug01/checkpoints/checkpoint_99 \
# --train-batch-size 5 --val-batch-size 25 \
# --epochs 100 --lr-warmup-epochs 5  --lr-hold-max-epochs 35 --limit-train-iters 1000  --limit-val-iters 100

# rm -rf /media/bryan/ssd01/expr/detr_from_scratch/100-epochs
accelerate launch --gpu_ids 0, --num_processes 1 detr/train.py \
--output-dir /media/bryan/ssd01/expr/detr_from_scratch/100-epochs \
--epochs 100 --lr-warmup-epochs 5  --lr-hold-max-epochs 35