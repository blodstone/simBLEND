#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/aspect_sentiment_sts-epoch=15-val_loss=0.8642.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_sentiment --train_path $DATASET_DIR/mind_resplit/MINDlarge_sentiment_train --dev_path $DATASET_DIR/mind_resplit/MINDlarge_sentiment_dev --test_path $DATASET_DIR/mind_resplit/MINDlarge_sentiment_test --selected_aspect "sentiment_class" --gpu_ids 1
 
 