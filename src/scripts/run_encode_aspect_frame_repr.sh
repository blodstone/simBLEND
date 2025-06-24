#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/aspect_frame-epoch=19-val_loss=0.1394.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_frame --train_path $DATASET_DIR/mind/MINDlarge_MFC_train --dev_path $DATASET_DIR/mind/MINDlarge_MFC_dev --test_path $DATASET_DIR/mind/MINDlarge_MFC_test --selected_aspect "frame_class" --gpu_ids 2
 