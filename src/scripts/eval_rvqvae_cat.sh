#!/bin/bash

source ./secrets.sh
uv run python ../eval_rvqvae.py \
  --test_path $BASE_DIR/outputs/mind/dev_mind_category_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/src/checkpoints/rvqvae_cat-epoch=02-val_loss=0.2482.ckpt \
  --codebook_dim 128 \
  --codebook_sizes 32 \
  --hidden_size 512 \
  --input_size 1024 \
  --num_quantizers 1 \
  --batch_size 2048 \
  --devices 1 \
  --gpu_ids 1\
