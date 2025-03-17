#!/bin/bash

BATCH_SIZE=128
PRETRAIN_CHKPT=checkpoints/mae_pretrain/checkpoint-199.pth
BLR=1e-3 # follow vit-base
EPOCHS=100
MODEL=vit_tiny_img32_patch16
INPUT_SIZE=32
NUM_CLASSES=10

DATA=data
LOG=tensorboard/mae_finetune_wopretrain
OUTPUT=checkpoints/mae_finetune_wopretrain

NUM_GPUS=1
MASTER_PORT=29501  # Change if needed
SMOOTHING=0.0

# Compute per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))

CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
    --batch_size $PER_GPU_BATCH_SIZE \
    --model $MODEL --cls_token \
    --input_size $INPUT_SIZE \
    --nb_classes $NUM_CLASSES \
    --data_path $DATA \
    --epochs $EPOCHS \
    --blr $BLR \
    --weight_decay 0.05 \
    --log_dir $LOG \
    --smoothing $SMOOTHING \
    --output_dir $OUTPUT 