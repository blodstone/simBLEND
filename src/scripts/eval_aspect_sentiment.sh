#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TEST_DATA_PATH=$DATASET_DIR/mind/MINDlarge_sentiment_dev
OUTPUT_DIR=$BASE_DIR/outputs/eval
uv run python ../eval_aspect.py \
  --checkpoint_path $BASE_DIR/checkpoints/aspect_sentiment_sts-epoch=15-val_loss=0.8642.ckpt\
  --test_path $TEST_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --selected_aspect sentiment_class \
  --name sentiment_sts\
  --batch_size 64 \
  --devices 1 \
  --gpu_ids 1 \
  --num_workers 4