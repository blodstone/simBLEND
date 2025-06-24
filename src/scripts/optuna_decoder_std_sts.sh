#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)


# Training Parameters (Defaults from script, modify as needed)
CHECKPOINT_PATH=$BASE_DIR/checkpoints/rvqvae_std_sts-epoch=12-val_loss=0.72483.ckpt
MAX_EPOCHS=4
DEVICES=1
# GPU_IDS=0 # Comma-separated list of GPU IDs
SEED=42
WARMUP_EPOCHS=1
CODEBOOK_SIZE=414
NUM_WORKERS=4
# Optuna Parameters
STUDY_NAME="decoder_std_sts"
N_TRIALS=50
STORAGE="sqlite:///optuna_decoder_std_sts.db"

for i in {1}; do
  for GPU_IDS in 0 1 3 5; do
    echo "Starting trial $i with GPU IDs: $GPU_IDS"
    sleep 6; uv run python ../optuna_decoder.py \
      --train_path $BASE_DIR/outputs/mind/train_mind_small_std_sts_histories_indices.csv \
      --dev_path $BASE_DIR/outputs/mind/dev_mind_small_std_sts_histories_indices.csv \
      --checkpoint_path $CHECKPOINT_PATH \
      --codebook_size $CODEBOOK_SIZE \
      --max_epochs $MAX_EPOCHS \
      --num_workers $NUM_WORKERS \
      --seed $SEED \
      --devices $DEVICES \
      --gpu_ids "$GPU_IDS" \
      --n_trials $N_TRIALS \
      --study_name $STUDY_NAME \
      --storage $STORAGE &
  done
done
wait
echo "All trials completed."