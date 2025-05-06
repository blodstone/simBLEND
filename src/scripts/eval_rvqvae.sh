#!/bin/bash

source ./secrets.sh
uv run python ../eval_rvqvae.py \
  --test_path $BASE_DIR/outputs/mind/dev_mind_category_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/src/checkpoints/rvqvae_aspect_ema_reset-epoch=01-val_loss=0.1478.ckpt\
  --codebook_dim 512 \
  --codebook_size 20 \
  --hidden_size 768 \
  --input_size 1024 \
  --num_quantizers 3 \
  --batch_size 2048 \
  --devices 1 \
  --gpu_ids 1\
