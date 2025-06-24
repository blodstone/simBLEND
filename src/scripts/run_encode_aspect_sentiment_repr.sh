#!/bin/bash
source ./secrets.sh
uv run python ../encode_aspect_repr.py --model_path $BASE_DIR/checkpoints/aspect_sentiment-epoch=07-val_loss=1.1253.ckpt --output_folder $BASE_DIR/outputs/mind/ --output_name mind_sentiment --train_path $DATASET_DIR/mind/MINDlarge_sentiment_train --dev_path $DATASET_DIR/mind/MINDlarge_sentiment_dev --test_path $DATASET_DIR/mind/MINDlarge_sentiment_test --selected_aspect "sentiment_class" --gpu_ids 2
 
 