#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TRAIN_DATA_PATH=$DATASET_DIR/mind/MINDlarge_train
DEV_DATA_PATH=$DATASET_DIR/mind/MINDlarge_dev
TEST_DATA_PATH=$DATASET_DIR/mind/MINDlarge_test

uv run python ../classify_sentiment.py \
  --train_path $TRAIN_DATA_PATH \
  --dev_path $DEV_DATA_PATH \
  --test_path $TEST_DATA_PATH \
  --output_name sentiment 