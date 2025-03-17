#!/bin/bash

# Training parameters
BATCH_SIZE=64
EPOCHS=200
ACCUM_ITER=8
MODEL=mae_vit_tiny_img32_patch16_dec512d8b
MASK_RATIO=0.75
BLR=1.5e-4
WEIGHT_DECAY=0.05
DATA=data
LOG=log
OUTPUT=checkpoints

# Set the number of GPUs
NUM_GPUS=8
MASTER_PORT=29500  # Change if needed

# Compute per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))

# Launch Distributed Data Parallel (DDP) training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main_pretrain.py \
    --batch_size $PER_GPU_BATCH_SIZE \
    --epochs $EPOCHS \
    --accum_iter $ACCUM_ITER \
    --model $MODEL \
    --mask_ratio $MASK_RATIO \
    --blr $BLR \
    --weight_decay $WEIGHT_DECAY \
    --data $DATA \
    --log_dir $LOG \
    --output_dir $OUTPUT