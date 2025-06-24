#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TEST_DATA_PATH=$DATASET_DIR/mind/MINDlarge_sentiment_dev
OUTPUT_DIR=$BASE_DIR/outputs/eval
uv run python ../eval_aspect.py \
  --checkpoint_path $BASE_DIR/checkpoints/aspect_sentiment-epoch=07-val_loss=1.1253.ckpt \
  --test_path $TEST_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --name sentiment_class \
  --batch_size 64 \
  --devices 1 \
  --gpu_ids 1 \
  --num_workers 4