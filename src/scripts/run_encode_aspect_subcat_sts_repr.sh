#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/aspect_subcat_sts-epoch=28-val_loss=0.1984.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_subcategory --train_path $DATASET_DIR/mind_resplit/MINDlarge_train --dev_path $DATASET_DIR/mind_resplit/MINDlarge_dev --test_path $DATASET_DIR/mind_resplit/MINDlarge_test --selected_aspect "subcategory_class" --gpu_ids 1
 