#!/bin/bash

source ./secrets.sh
uv run python ../encode_vqvae.py \
--model_path $BASE_DIR/checkpoints/rvqvae_aspect_ema_reset_single-epoch=36-val_loss=0.1263.ckpt \
--output_folder $BASE_DIR/outputs/mind \
--output_name mind_category \
--train_path $DATASET_DIR/mind/MINDlarge_train \
--dev_path $DATASET_DIR/mind/MINDlarge_dev \
--test_path $DATASET_DIR/mind/MINDlarge_test \
--train_a_dict_path $BASE_DIR/outputs/mind/train_mind_category_aspect_vectors.txt \
--dev_a_dict_path $BASE_DIR/outputs/mind/dev_mind_category_aspect_vectors.txt \
--test_a_dict_path $BASE_DIR/outputs/mind/test_mind_category_aspect_vectors.txt \

