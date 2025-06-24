#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TEST_DATA_PATH=$BASE_DIR/outputs/mind/dev_mind_political_histories_indices.csv
CHECKPOINT_PATH=$BASE_DIR/src/checkpoints/seqvqvae_political-epoch=03-val_loss=3.1228.ckpt
# Model/Llama Parameters (Defaults from script, modify as needed)
# CODEBOOK_SIZE=1120
# HIDDEN_SIZE=768
# INTERMEDIATE_SIZE=2048
# NUM_HIDDEN_LAYERS=10
# NUM_ATTENTION_HEADS=12
# MAX_POSITION_EMBEDDINGS=4090
CODEBOOK_SIZE=32
HIDDEN_SIZE=768
INTERMEDIATE_SIZE=2048
NUM_HIDDEN_LAYERS=10
NUM_ATTENTION_HEADS=12
MAX_POSITION_EMBEDDINGS=4090

# Training Parameters (Defaults from script, modify as needed)
BATCH_SIZE=8
DEVICES=1
GPU_IDS=3 # Comma-separated list of GPU IDs
TB_NAME="eval_seqvqvae_political"

uv run python ../eval_seq_vqvae.py \
    --test_path "$TEST_DATA_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --batch_size $BATCH_SIZE \
    --intermediate_size $INTERMEDIATE_SIZE \
    --num_hidden_layers $NUM_HIDDEN_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --max_position_embeddings $MAX_POSITION_EMBEDDINGS \
    --codebook_size $CODEBOOK_SIZE \
    --batch_size $BATCH_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --devices $DEVICES \
    --gpu_ids "$GPU_IDS" \
    --tb_name "$TB_NAME" \