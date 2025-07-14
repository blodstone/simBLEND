#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/aspect_frame_sts-epoch=17-val_loss=0.6141.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_frame_sts --train_path $DATASET_DIR/mind_resplit/MINDlarge_MFC_train --dev_path $DATASET_DIR/mind_resplit/MINDlarge_MFC_dev --test_path $DATASET_DIR/mind_resplit/MINDlarge_MFC_test --selected_aspect "frame_class" --gpu_ids 1
 