#!/bin/bash

BATCH_SIZE=16384
PRETRAIN_CHKPT=checkpoints/mae_pretrain/checkpoint-199.pth
BLR=0.1
EPOCHS=100
MODEL=vit_tiny_img32_patch16
NUM_CLASSES=10

DATA=data
LOG=tensorboard/mae_linprobe
OUTPUT=checkpoints/mae_linprobe

NUM_GPUS=8
MASTER_PORT=29502  # Change if needed

# Compute per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main_linprobe.py \
    --batch_size $PER_GPU_BATCH_SIZE \
    --model $MODEL --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --nb_classes $NUM_CLASSES \
    --data_path $DATA \
    --epochs $EPOCHS \
    --blr $BLR \
    --weight_decay 0.0 \
    --log_dir $LOG \
    --output_dir $OUTPUT \
    1> log/mae_linprobe/base_pretrain_linprobe.log \
    2> log/mae_linprobe/base_pretrain_linprobe.err
