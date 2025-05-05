#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/amodule-epoch=00-train_loss=0.00.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_category --train_path $DATASET_DIR/mind/MINDlarge_train --dev_path $DATASET_DIR/mind/MINDlarge_dev --test_path $DATASET_DIR/mind/MINDlarge_test 
 