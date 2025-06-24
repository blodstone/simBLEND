#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/aspect_cat-epoch=23-val_loss=0.1483.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_category --train_path $DATASET_DIR/mind/MINDlarge_train --dev_path $DATASET_DIR/mind/MINDlarge_dev --test_path $DATASET_DIR/mind/MINDlarge_test --selected_aspect "category_class" --gpu_ids 1
 