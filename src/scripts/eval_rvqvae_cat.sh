#!/bin/bash

source ./secrets.sh
uv run python ../eval_rvqvae.py \
  --test_path $BASE_DIR/outputs/mind/dev_mind_category_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/checkpoints/rvqvae_cat_sts-epoch=12-val_loss=0.44667.ckpt \
  --codebook_dim 512 \
  --codebook_sizes 69 \
  --hidden_size 256 \
  --input_size 1024 \
  --num_quantizers 1 \
  --batch_size 2048 \
  --devices 1 \
  --gpu_ids 1\
