#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TEST_DATA_PATH=$DATASET_DIR/mind/MINDlarge_political_dev
OUTPUT_DIR=$BASE_DIR/outputs/eval
uv run python ../eval_aspect.py \
  --checkpoint_path $BASE_DIR/checkpoints/aspect_political_sts-epoch=17-val_loss=2.9533.ckpt \
  --test_path $TEST_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --selected_aspect political_class \
  --name political_sts\
  --batch_size 64 \
  --devices 1 \
  --gpu_ids 0 \
  --num_workers 4