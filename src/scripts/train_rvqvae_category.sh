#!/bin/bash

source ./secrets.sh
uv run python ../train_rvqvae.py \
  --train_path $BASE_DIR/outputs/mind/train_mind_category_aspect_vectors.txt \
  --dev_path $BASE_DIR/outputs/mind/dev_mind_category_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/src/checkpoints \
  --commitment_cost 0.38 \
  --reset_threshold 0 \
  --codebook_dim 512 \
  --codebook_size 69 \
  --decay 0.98 \
  --hidden_size 256 \
  --input_size 1024 \
  --num_quantizers 1\
  --batch_size 256 \
  --max_epochs 20 \
  --devices 1 \
  --warm_up_epochs 1 \
  --gpu_ids 1 \
  --seed 42 \
  --learning_rate 2e-5 \
  --num_workers 30 \
  --grad_accum 1 \
  --tb_name rvqvae_cat_sts
