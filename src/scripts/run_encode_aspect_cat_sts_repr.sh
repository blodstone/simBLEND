#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/aspect_cat_sts-epoch=27-val_loss=0.6584.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_category --train_path $DATASET_DIR/mind_resplit/MINDlarge_train --dev_path $DATASET_DIR/mind_resplit/MINDlarge_dev --test_path $DATASET_DIR/mind_resplit/MINDlarge_test --selected_aspect "category_class" --gpu_ids 1
 