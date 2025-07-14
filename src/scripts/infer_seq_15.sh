#!/bin/bash

source ./secrets.sh
INPUT_FILE=$BASE_DIR/outputs/mind/dev_combined_history_indices.csv
CHECKPOINT_PATH=$BASE_DIR/src/checkpoints/seqvqvae_all_sts-epoch=29-val_loss=1.8966.ckpt
CODEBOOK_SIZE="414 69 106 69 117"
OUTPUT_PATH=$BASE_DIR/outputs/mind/seq_prediction_beam_15.txt
BEAM_SIZE=15
BATCH_SIZE=8
GPU_IDS="2"  # Set the GPU IDs to use, e.g., "0" for the first GPU
uv run python ../infer_sequence.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --codebook_size $CODEBOOK_SIZE \
    --output_path "$OUTPUT_PATH" \
    --beam_size $BEAM_SIZE \
    --input_file "$INPUT_FILE" \
    --batch_size $BATCH_SIZE \
    --gpu_ids "$GPU_IDS"