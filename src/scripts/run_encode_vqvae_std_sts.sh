#!/bin/bash

source ./secrets.sh
uv run python ../encode_aspect_vq.py \
--model_path $BASE_DIR/checkpoints/rvqvae_std_sts-epoch=12-val_loss=0.72483.ckpt \
--output_folder $BASE_DIR/outputs/mind \
--output_name mind_std_sts \
--train_path $DATASET_DIR/mind/MINDlarge_train \
--dev_path $DATASET_DIR/mind/MINDlarge_dev \
--test_path $DATASET_DIR/mind/MINDlarge_test \
--codebook_dim 512 \
--codebook_sizes 414 \
--hidden_size 128 \
--train_a_dict_path $BASE_DIR/outputs/mind/train_mind_std_sts_aspect_vectors.txt \
--dev_a_dict_path $BASE_DIR/outputs/mind/dev_mind_std_sts_aspect_vectors.txt \
--test_a_dict_path $BASE_DIR/outputs/mind/test_mind_std_sts_aspect_vectors.txt \

