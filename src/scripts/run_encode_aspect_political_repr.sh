#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/aspect_political_sts-epoch=17-val_loss=2.9533.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_political --train_path $DATASET_DIR/mind_resplit/MINDlarge_political_train --dev_path $DATASET_DIR/mind_resplit/MINDlarge_political_dev --test_path $DATASET_DIR/mind_resplit/MINDlarge_political_test --selected_aspect "political_class" --gpu_ids 1
 