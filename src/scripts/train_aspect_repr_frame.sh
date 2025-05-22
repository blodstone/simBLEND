#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TRAIN_DATA_PATH=$DATASET_DIR/mind/MINDlarge_MFC_train
DEV_DATA_PATH=$DATASET_DIR/mind/MINDlarge_MFC_dev
CHECKPOINT_DIR=$BASE_DIR/src/checkpoints

# Optimizer/Scheduler Parameters (Defaults from script, modify as needed)
LEARNING_RATE=5e-4
WARMUP_EPOCHS=1

# Training Parameters (Defaults from script, modify as needed)
BATCH_SIZE=24
GRAD_ACCUM=2
MAX_EPOCHS=25
DEVICES=1
GPU_IDS=0 # Comma-separated list of GPU IDs
TB_NAME="aspect_frame"
SEED=42


uv run python ../train_aspect_repr.py \
    --train_path "$TRAIN_DATA_PATH" \
    --dev_path "$DEV_DATA_PATH" \
    --selected_aspect "frame_class" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --learning_rate $LEARNING_RATE \
    --warm_up_epochs $WARMUP_EPOCHS \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --grad_accum $GRAD_ACCUM \
    --devices $DEVICES \
    --gpu_ids "$GPU_IDS" \
    --tb_name "$TB_NAME" \
    --seed $SEED 