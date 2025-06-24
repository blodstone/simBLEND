#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TEST_DATA_PATH=$DATASET_DIR/mind/MINDlarge_dev
OUTPUT_DIR=$BASE_DIR/outputs/eval
uv run python ../eval_aspect.py \
  --checkpoint_path $BASE_DIR/src/checkpoints/aspect_subcat_sts-epoch=23-val_loss=0.2110.ckpt\
  --test_path $TEST_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --selected_aspect subcategory_class \
  --name subcategory_sts \
  --batch_size 64 \
  --devices 1 \
  --gpu_ids 0 \
  --num_workers 4