#!/bin/bash

# Training parameters
BATCH_SIZE=4096
EPOCHS=200
ACCUM_ITER=1
MODEL=mae_vit_base_img32_patch16_dec512d8b
MASK_RATIO=0.75
BLR=1.5e-4
WEIGHT_DECAY=0.05
DATA=data
LOG=tensorboard/mae_pretrain_base
OUTPUT=checkpoints/mae_pretrain_base

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
    --data_path $DATA \
    --log_dir $LOG \
    --output_dir $OUTPUT \
    1> log/mae_pretrain_base/base_pretrain.log \
    2> log/mae_pretrain_base/base_pretrain.err
