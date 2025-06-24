#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TEST_DATA_PATH=$DATASET_DIR/mind/MINDlarge_political_dev
OUTPUT_DIR=$BASE_DIR/outputs/eval
uv run python ../eval_aspect.py \
  --checkpoint_path $BASE_DIR/checkpoints/aspect_political-epoch=19-val_loss=2.9094.ckpt \
  --test_path $TEST_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --name political_class \
  --batch_size 64 \
  --devices 1 \
  --gpu_ids 0 \
  --num_workers 4