#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/aspect_political-epoch=19-val_loss=2.9094.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_political --train_path $DATASET_DIR/mind/MINDlarge_political_train --dev_path $DATASET_DIR/mind/MINDlarge_political_dev --test_path $DATASET_DIR/mind/MINDlarge_political_test --selected_aspect "political_class" --gpu_ids 1
 