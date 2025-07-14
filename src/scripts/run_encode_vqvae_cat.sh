#!/bin/bash

source ./secrets.sh
uv run python ../encode_aspect_vq.py \
--model_path $BASE_DIR/src/checkpoints/rvqvae_cat_sts-epoch=12-val_loss=0.44667.ckpt \
--output_folder $BASE_DIR/outputs/mind \
--output_name mind_category \
--train_path $DATASET_DIR/mind_resplit/MINDlarge_train \
--dev_path $DATASET_DIR/mind_resplit/MINDlarge_dev \
--test_path $DATASET_DIR/mind_resplit/MINDlarge_test \
--codebook_dim 512 \
--codebook_sizes 69 \
--hidden_size 256 \
--train_a_dict_path $BASE_DIR/outputs/mind/train_mind_category_aspect_vectors.txt \
--dev_a_dict_path $BASE_DIR/outputs/mind/dev_mind_category_aspect_vectors.txt \
--test_a_dict_path $BASE_DIR/outputs/mind/test_mind_category_aspect_vectors.txt \

