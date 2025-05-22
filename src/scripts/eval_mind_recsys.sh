#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TRAIN_DATA_PATH=$DATASET_DIR/mind/MINDlarge_train
DEV_DATA_PATH=$DATASET_DIR/mind/MINDlarge_dev
CHECKPOINT_DIR=$BASE_DIR/checkpoints/lstur-epoch=07-val_loss=3.2441.ckpt
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


# Training Parameters (Defaults from script, modify as needed)
BATCH_SIZE=24
DEVICES=1
GPU_IDS=1 # Comma-separated list of GPU IDs
SEED=42


uv run python ../eval_mind_recsys.py \
    --train_path "$TRAIN_DATA_PATH" \
    --dev_path "$DEV_DATA_PATH" \
    --test_path "$DEV_DATA_PATH" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --glove_path "$GLOVE_PATH" \
    --cat_vocab_size $CAT_VOCAB_SIZE \
    --subcat_vocab_size $SUBCAT_VOCAB_SIZE \
    --user_id_size $USER_ID_SIZE \
    --window_size $WINDOW_SIZE \
    --embedding_size $EMBEDDING_SIZE \
    --num_negative_samples_k $NUM_NEGATIVE_SAMPLES_K \
    --user_hidden_size $USER_HIDDEN_SIZE \
    --final_hidden_size $FINAL_HIDDEN_SIZE \
    --devices $DEVICES \
    --gpu_ids "$GPU_IDS" \
    --tb_name "$TB_NAME" \
    --seed $SEED 