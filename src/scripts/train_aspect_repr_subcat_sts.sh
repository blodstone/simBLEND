#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TRAIN_DATA_PATH=$DATASET_DIR/mind/MINDlarge_train
DEV_DATA_PATH=$DATASET_DIR/mind/MINDlarge_dev
CHECKPOINT_DIR=$BASE_DIR/src/checkpoints
NUM_WORKERS=4
PROJECTION_SIZE=128
PLM_NAME=nickprock/ModernBERT-large-sts

# Optimizer/Scheduler Parameters (Defaults from script, modify as needed)
LEARNING_RATE=1e-4
WARMUP_EPOCHS=1

# Training Parameters (Defaults from script, modify as needed)
BATCH_SIZE=8
GRAD_ACCUM=1
MAX_EPOCHS=50
DEVICES=1
GPU_IDS=3 # Comma-separated list of GPU IDs
TB_NAME="aspect_subcat_sts"
SEED=55


uv run python ../train_aspect_repr.py \
    --plm_name $PLM_NAME \
    --train_path "$TRAIN_DATA_PATH" \
    --dev_path "$DEV_DATA_PATH" \
    --projection_size $PROJECTION_SIZE \
    --selected_aspect "subcategory_class" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --learning_rate $LEARNING_RATE \
    --warm_up_epochs $WARMUP_EPOCHS \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --grad_accum $GRAD_ACCUM \
    --devices $DEVICES \
    --gpu_ids "$GPU_IDS" \
    --tb_name "$TB_NAME" \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    
