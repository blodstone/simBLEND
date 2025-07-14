#!/bin/bash

source ./secrets.sh
uv run python ../encode_aspect_vq.py \
--model_path $BASE_DIR/checkpoints/rvqvae_political_sts-epoch=07-val_loss=0.27821.ckpt \
--output_folder $BASE_DIR/outputs/mind \
--output_name mind_political \
--train_path $DATASET_DIR/mind_resplit/MINDlarge_train \
--dev_path $DATASET_DIR/mind_resplit/MINDlarge_dev \
--test_path $DATASET_DIR/mind_resplit/MINDlarge_test \
--codebook_dim 128 \
--codebook_sizes 69 \
--hidden_size 128 \
--train_a_dict_path $BASE_DIR/outputs/mind/train_mind_political_aspect_vectors.txt \
--dev_a_dict_path $BASE_DIR/outputs/mind/dev_mind_political_aspect_vectors.txt \
--test_a_dict_path $BASE_DIR/outputs/mind/test_mind_political_aspect_vectors.txt \

