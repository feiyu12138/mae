#!/bin/bash
LAYER=11
TIMES=5
EMA_WARMUP=1
EMA_DECAY=0.99
BATCH_SIZE=16384
PRETRAIN_CKPT=checkpoints/bmae_pretrain_layer_${LAYER}_ema_warmup_${EMA_WARMUP}_decay_${EMA_DECAY}/checkpoint-199.pth
BLR=0.1
EPOCHS=100
MODEL=vit_tiny_img32_patch4
NUM_CLASSES=10

DATA=data
LOG=tensorboard/bmae_linprobe_layer_${LAYER}_ema_warmup_${EMA_WARMUP}_decay_${EMA_DECAY}
OUTPUT=checkpoints/bmae_linprobe_layer_${LAYER}_ema_warmup_${EMA_WARMUP}_decay_${EMA_DECAY}

NUM_GPUS=8
MASTER_PORT=29522  # Change if needed

# Compute per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main_linprobe.py \
    --batch_size $PER_GPU_BATCH_SIZE \
    --model $MODEL --cls_token \
    --finetune ${PRETRAIN_CKPT} \
    --nb_classes $NUM_CLASSES \
    --data_path $DATA \
    --epochs $EPOCHS \
    --blr $BLR \
    --weight_decay 0.0 \
    --log_dir $LOG \
    --output_dir $OUTPUT \
    1> log/bmae_linprobe/bmae_pretrain_layer_${LAYER}_ema_warmup_${EMA_WARMUP}_decay_${EMA_DECAY}_linprobe.log \
    2> log/bmae_linprobe/bmae_pretrain_layer_${LAYER}_ema_warmup_${EMA_WARMUP}_decay_${EMA_DECAY}_linprobe.err
