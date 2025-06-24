#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TRAIN_DATA_PATH=$DATASET_DIR/mind/MINDlarge_train
DEV_DATA_PATH=$DATASET_DIR/mind/MINDlarge_dev
CHECKPOINT_DIR=$BASE_DIR/src/checkpoints

# Model/Data Parameters (Update as needed)
GLOVE_PATH=$DATASET_DIR/glove/glove.6B.300d.txt # Set to path of GloVe embeddings file or leave empty
CAT_VOCAB_SIZE=18 # Set to actual category vocab size
SUBCAT_VOCAB_SIZE=285 # Set to actual subcategory vocab size
USER_ID_SIZE=711222 # Set to actual user id vocab size
WINDOW_SIZE=4
EMBEDDING_SIZE=300
NUM_NEGATIVE_SAMPLES_K=5
USER_HIDDEN_SIZE=300
FINAL_HIDDEN_SIZE=512

# Add new arguments to the command


# Optimizer/Scheduler Parameters (Defaults from script, modify as needed)
LEARNING_RATE=5e-5
WARMUP_EPOCHS=1

# Training Parameters (Defaults from script, modify as needed)
BATCH_SIZE=24
GRAD_ACCUM=2
MAX_EPOCHS=10
DEVICES=2
GPU_IDS=3,4 # Comma-separated list of GPU IDs
TB_NAME="lstur"
SEED=42


uv run python ../train_mind_recsys.py \
    --train_path "$TRAIN_DATA_PATH" \
    --dev_path "$DEV_DATA_PATH" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --learning_rate $LEARNING_RATE \
    --glove_path "$GLOVE_PATH" \
    --cat_vocab_size $CAT_VOCAB_SIZE \
    --subcat_vocab_size $SUBCAT_VOCAB_SIZE \
    --user_id_size $USER_ID_SIZE \
    --window_size $WINDOW_SIZE \
    --embedding_size $EMBEDDING_SIZE \
    --num_negative_samples_k $NUM_NEGATIVE_SAMPLES_K \
    --user_hidden_size $USER_HIDDEN_SIZE \
    --final_hidden_size $FINAL_HIDDEN_SIZE \
    --max_epochs $MAX_EPOCHS \
    --grad_accum $GRAD_ACCUM \
    --devices $DEVICES \
    --gpu_ids "$GPU_IDS" \
    --tb_name "$TB_NAME" \
    --seed $SEED 