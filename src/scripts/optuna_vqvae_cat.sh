#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)


# Training Parameters (Defaults from script, modify as needed)

MAX_EPOCHS=15
DEVICES=1
# GPU_IDS=0 # Comma-separated list of GPU IDs
SEED=42
WARMUP_EPOCHS=1
NUM_WORKERS=4
# Optuna Parameters
STUDY_NAME="vqvae_cat"
N_TRIALS=50
MIN_CODEBOOK_SIZE=18
STORAGE="sqlite:///optuna_vqvae_cat.db"

for GPU_IDS in 0 1 2 4; do
  echo "Starting trial $i with GPU IDs: $GPU_IDS"
  sleep 30; uv run python ../optuna_vqvae.py \
    --train_path $BASE_DIR/outputs/mind/train_mind_category_aspect_vectors.txt \
    --dev_path $BASE_DIR/outputs/mind/dev_mind_category_aspect_vectors.txt \
    --input_size 1024 \
    --max_epochs_trial $MAX_EPOCHS \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --devices $DEVICES \
    --gpu_ids "$GPU_IDS" \
    --min_codebook_size $MIN_CODEBOOK_SIZE \
    --n_trials $N_TRIALS \
    --study_name $STUDY_NAME \
    --storage $STORAGE &
done
wait
echo "All trials completed."
# uv run python ../optuna_vqvae.py \
#   --train_path $BASE_DIR/outputs/mind/train_mind_std_aspect_vectors.txt \
#   --dev_path $BASE_DIR/outputs/mind/dev_mind_std_aspect_vectors.txt \
#   --input_size 1024 \
#   --max_epochs_trial $MAX_EPOCHS \
#   --num_workers $NUM_WORKERS \
#   --seed $SEED \
#   --devices $DEVICES \
#   --gpu_ids "$GPU_IDS" \
#   --n_trials $N_TRIALS \
#   --study_name $STUDY_NAME \
#   --storage $STORAGE