#!/bin/bash

# Training parameters
BATCH_SIZE=4096
EPOCHS=200
ACCUM_ITER=1
MODEL=mae_vit_tiny_img32_patch4_dec512d8b
MASK_RATIO=0.75
BLR=1.5e-4
WEIGHT_DECAY=0.05
DATA=data
BASE_MODEL=checkpoints/mae_pretrain/checkpoint-199.pth
TIMES=3
LAYER=9
LOG=tensorboard/bmae_pretrain_layer_${LAYER}_time_${TIMES}
OUTPUT=checkpoints/bmae_pretrain_layer_${LAYER}_time_${TIMES}
# Set the number of GPUs
NUM_GPUS=8
MASTER_PORT=29502  # Change if needed

# Compute per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))
# Launch Distributed Data Parallel (DDP) training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main_pretrain_bootstrap.py \
    --batch_size $PER_GPU_BATCH_SIZE \
    --epochs $EPOCHS \
    --accum_iter $ACCUM_ITER \
    --model $MODEL \
    --base_model $BASE_MODEL \
    --select_layer $LAYER \
    --bootstrap_times $TIMES \
    --mask_ratio $MASK_RATIO \
    --blr $BLR \
    --norm_pix_loss \
    --weight_decay $WEIGHT_DECAY \
    --data_path $DATA \
    --log_dir $LOG \
    --output_dir $OUTPUT \
    1> log/bmae_train/pretrain_bootstrap_LAYER_${LAYER}_TIMES_${TIMES}.log \
    2> log/bmae_train/pretrain_bootstrap_LAYER_${LAYER}_TIMES_${TIMES}.err
