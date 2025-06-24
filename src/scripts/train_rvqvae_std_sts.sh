#!/bin/bash

source ./secrets.sh
uv run python ../train_rvqvae.py \
  --train_path $BASE_DIR/outputs/mind/train_mind_std_sts_aspect_vectors.txt \
  --dev_path $BASE_DIR/outputs/mind/dev_mind_std_sts_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/src/checkpoints \
  --commitment_cost 0.266 \
  --reset_threshold 0 \
  --codebook_dim 512 \
  --decay 0.99 \
  --codebook_sizes 414 \
  --hidden_size 128 \
  --input_size 1024 \
  --num_quantizers 1 \
  --batch_size 256 \
  --max_epochs 50 \
  --devices 2 \
  --warm_up_epochs 1 \
  --gpu_ids '4,6' \
  --seed 42 \
  --learning_rate 1e-5 \
  --num_workers 4 \
  --grad_accum 1 \
  --tb_name rvqvae_std_sts
