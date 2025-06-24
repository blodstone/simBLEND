#!/bin/bash
source ./secrets.sh
PLM_NAME=nickprock/ModernBERT-large-sts
uv run python ../encode_aspect_repr.py --plm_name $PLM_NAME --output_folder $BASE_DIR/outputs/mind/ --output_name mind_std_sts --train_path $DATASET_DIR/mind/MINDlarge_train --dev_path $DATASET_DIR/mind/MINDlarge_dev --test_path $DATASET_DIR/mind/MINDlarge_test
