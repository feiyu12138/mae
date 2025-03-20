#!/bin/bash

# Training parameters
BATCH_SIZE=4096
EPOCHS=200
ACCUM_ITER=2
MODEL=mae_vit_tiny_img32_patch4_dec512d8b
MASK_RATIO=0.75
BLR=1.5e-4
WEIGHT_DECAY=0.05
DATA=data
LOG=tensorboard/mae_pretrain
OUTPUT=checkpoints/bmae_pretrain
BASE_MODEL=checkpoints/mae_pretrain/checkpoint-199.pth
TIMES=3
LAYER=6
# Set the number of GPUs
NUM_GPUS=8
MASTER_PORT=29500  # Change if needed
NORM_PIX_LOSS=False
# Compute per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))

# Launch Distributed Data Parallel (DDP) training
CUDA_VISIBLE_DEVICES=0 python main_pretrain_bootstrap.py \
    --batch_size $PER_GPU_BATCH_SIZE \
    --epochs $EPOCHS \
    --accum_iter $ACCUM_ITER \
    --model $MODEL \
    --base_model $BASE_MODEL \
    --select_layer $LAYER \
    --bootstrap_times $TIMES \
    --mask_ratio $MASK_RATIO \
    --blr $BLR \
    --weight_decay $WEIGHT_DECAY \
    --data_path $DATA \
    --norm_pix_loss \
    --log_dir $LOG \
    --output_dir $OUTPUT \
    # 1> log/bmae_train/pretrain_bootstrap_LAYER_${LAYER}_TIMES_${TIMES}.log \
    # 2> log/bmae_train/pretrain_bootstrap_LAYER_${LAYER}_TIMES_${TIMES}.err
