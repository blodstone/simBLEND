#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TEST_DATA_PATH=$BASE_DIR/outputs/mind/dev_mind_category_histories_indices.csv
CHECKPOINT_PATH=$BASE_DIR/checkpoints/seqvqvae-epoch=04-val_loss=3.5992.ckpt
# Model/Llama Parameters (Defaults from script, modify as needed)
CODEBOOK_SIZE=256
HIDDEN_SIZE=512
INTERMEDIATE_SIZE=2048
NUM_HIDDEN_LAYERS=8
NUM_ATTENTION_HEADS=8
MAX_POSITION_EMBEDDINGS=4096

# Training Parameters (Defaults from script, modify as needed)
BATCH_SIZE=128
DEVICES=1
GPU_IDS=1 # Comma-separated list of GPU IDs
TB_NAME="eval_seqvqvae2"



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