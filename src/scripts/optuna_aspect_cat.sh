#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TRAIN_DATA_PATH=$DATASET_DIR/mind/MINDsmall_train
DEV_DATA_PATH=$DATASET_DIR/mind/MINDsmall_dev


# Training Parameters (Defaults from script, modify as needed)
MAX_EPOCHS=8
DEVICES=1
GPU_IDS=2 # Comma-separated list of GPU IDs
SEED=42
WARMUP_EPOCHS=1
NUM_WORKERS=4
# Optuna Parameters
STUDY_NAME="aspect_cat_study"
N_TRIALS=50
STORAGE="sqlite:///optuna_aspect_cat.db"

uv run python ../optuna_aspect.py \
    --train_path "$TRAIN_DATA_PATH" \
    --dev_path "$DEV_DATA_PATH" \
    --selected_aspect "category_class" \
    --warm_up_epochs $WARMUP_EPOCHS \
    --max_epochs $MAX_EPOCHS \
    --warm_up_epochs $WARMUP_EPOCHS \
    --num_workers $NUM_WORKERS \
    --devices $DEVICES \
    --gpu_ids "$GPU_IDS" \
    --seed $SEED \
    --n_trials $N_TRIALS \
    --study_name $STUDY_NAME \
    --storage $STORAGE \
    --cleanup_trial_dirs
