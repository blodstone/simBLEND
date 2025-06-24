#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py  --output_folder $BASE_DIR/outputs/mind/ --output_name mind_std --train_path $DATASET_DIR/mind/MINDlarge_train --dev_path $DATASET_DIR/mind/MINDlarge_dev --test_path $DATASET_DIR/mind/MINDlarge_test 
 