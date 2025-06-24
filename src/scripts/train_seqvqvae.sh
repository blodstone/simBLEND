#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TRAIN_DATA_PATH=$BASE_DIR/outputs/mind/train_mind_category_histories_indices.csv
DEV_DATA_PATH=$BASE_DIR/outputs/mind/dev_mind_category_histories_indices.csv
CHECKPOINT_DIR=$BASE_DIR/src/checkpoints
# Model/Llama Parameters (Defaults from script, modify as needed)
CODEBOOK_SIZE=256
HIDDEN_SIZE=1248
INTERMEDIATE_SIZE=3256
NUM_HIDDEN_LAYERS=12
NUM_ATTENTION_HEADS=12
MAX_POSITION_EMBEDDINGS=4096

# Optimizer/Scheduler Parameters (Defaults from script, modify as needed)
LEARNING_RATE=5e-5
WARMUP_EPOCHS=1

# Training Parameters (Defaults from script, modify as needed)
BATCH_SIZE=64
GRAD_ACCUM=2
MAX_EPOCHS=20
DEVICES=1
GPU_IDS=4 # Comma-separated list of GPU IDs
TB_NAME="seqvqvae4"
SEED=42


uv run python ../train_seq_vqvae.py \
    --train_path "$TRAIN_DATA_PATH" \
    --dev_path "$DEV_DATA_PATH" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --learning_rate $LEARNING_RATE \
    --intermediate_size $INTERMEDIATE_SIZE \
    --num_hidden_layers $NUM_HIDDEN_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --max_position_embeddings $MAX_POSITION_EMBEDDINGS \
    --codebook_size $CODEBOOK_SIZE \
    --batch_size $BATCH_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --max_epochs $MAX_EPOCHS \
    --grad_accum $GRAD_ACCUM \
    --devices $DEVICES \
    --gpu_ids "$GPU_IDS" \
    --tb_name "$TB_NAME" \
    --seed $SEED