#!/bin/bash

source ./secrets.sh
uv run python ../encode_aspect_vq.py \
--model_path $BASE_DIR/checkpoints/rvqvae_sentiment-epoch=17-val_loss=0.0002.ckpt \
--output_folder $BASE_DIR/outputs/mind \
--output_name mind_sentiment \
--train_path $DATASET_DIR/mind/MINDlarge_train \
--dev_path $DATASET_DIR/mind/MINDlarge_dev \
--test_path $DATASET_DIR/mind/MINDlarge_test \
--codebook_dim 256 \
--codebook_sizes 32 \
--hidden_size 256 \
--train_a_dict_path $BASE_DIR/outputs/mind/train_mind_sentiment_aspect_vectors.txt \
--dev_a_dict_path $BASE_DIR/outputs/mind/dev_mind_sentiment_aspect_vectors.txt \
--test_a_dict_path $BASE_DIR/outputs/mind/test_mind_sentiment_aspect_vectors.txt \

