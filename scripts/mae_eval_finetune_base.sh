#!/bin/bash

BATCH_SIZE=1024
PRETRAIN_CHKPT=checkpoints/mae_pretrain_base/checkpoint-199.pth
BLR=5e-4 # follow vit-base
EPOCHS=100
MODEL=vit_base_img32_patch16
INPUT_SIZE=32
NUM_CLASSES=10

DATA=data
LOG=tensorboard/mae_finetune_base
OUTPUT=checkpoints/mae_finetune_base

NUM_GPUS=8
MASTER_PORT=29501  # Change if needed

# Compute per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main_finetune.py \
    --batch_size $PER_GPU_BATCH_SIZE \
    --model $MODEL --cls_token \
    --input_size $INPUT_SIZE \
    --finetune ${PRETRAIN_CHKPT} \
    --nb_classes $NUM_CLASSES \
    --data_path $DATA \
    --epochs $EPOCHS \
    --blr $BLR \
    --weight_decay 0.0 \
    --log_dir $LOG \
    --output_dir $OUTPUT \
    --layer_decay 0.65 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    1> log/mae_finetune_base/base_pretrain_finetune.log \
    2> log/mae_finetune_base/base_pretrain_finetune.err