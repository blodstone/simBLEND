#!/bin/bash

source ./secrets.sh
# Required Paths (Update these with your actual paths)
TEST_DATA_PATH=$DATASET_DIR/mind/MINDlarge_MFC_dev
OUTPUT_DIR=$BASE_DIR/outputs/eval
uv run python ../eval_aspect.py \
  --checkpoint_path $BASE_DIR/src/checkpoints/aspect_frame_sts-epoch=17-val_loss=0.6141.ckpt \
  --test_path $TEST_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --selected_aspect frame_class \
  --name frame_sts \
  --batch_size 64 \
  --devices 1 \
  --gpu_ids 1 \
  --num_workers 4