#!/bin/bash

source ./secrets.sh
uv run python ../train_rvqvae.py \
  --train_path $BASE_DIR/outputs/mind/train_mind_category_aspect_vectors.txt \
  --dev_path $BASE_DIR/outputs/mind/dev_mind_category_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/src/checkpoints \
  --codebook_dim 512 \
  --codebook_size 256 \
  --hidden_size 768 \
  --input_size 1024 \
  --num_quantizers 1 \
  --batch_size 512 \
  --max_epochs 100 \
  --devices 1 \
  --gpu_ids 3\
  --tb_name rvqvae_aspect_ema_reset_single
