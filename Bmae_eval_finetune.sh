#!/bin/bash
LAYER=11
TIMES=5
EMA_WARMUP=1
EMA_DECAY=0.99
BATCH_SIZE=1024
PRETRAIN_CKPT=checkpoints/bmae_pretrain/checkpoint-199.pth
BLR=1e-3 # follow vit-base
EPOCHS=100
MODEL=vit_tiny_img32_patch4
INPUT_SIZE=32
NUM_CLASSES=10

DATA=data
LOG=tensorboard/bmae_finetune
OUTPUT=checkpoints/bmae_finetune

NUM_GPUS=8
MASTER_PORT=29526  # Change if needed
SMOOTHING=0.0

# Compute per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main_finetune.py \
    --batch_size $PER_GPU_BATCH_SIZE \
    --model $MODEL --cls_token \
    --input_size $INPUT_SIZE \
    --nb_classes $NUM_CLASSES \
    --finetune $PRETRAIN_CKPT \
    --data_path $DATA \
    --epochs $EPOCHS \
    --blr $BLR \
    --weight_decay 0.05 \
    --log_dir $LOG \
    --smoothing $SMOOTHING \
    --output_dir $OUTPUT \
    1> log/bmae_finetune/bmae_pretrain.log \
    2> log/bmae_finetune/bmae_pretrain.err