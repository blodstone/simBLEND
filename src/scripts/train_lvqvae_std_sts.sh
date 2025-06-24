#!/bin/bash

source ./secrets.sh
uv run python ../train_lvqvae.py \
  --train_path $BASE_DIR/outputs/mind/train_mind_std_sts_aspect_vectors.txt \
  --dev_path $BASE_DIR/outputs/mind/dev_mind_std_sts_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/src/checkpoints \
  --commitment_cost 0.25 \
  --reset_threshold 0 \
  --codebook_dim 1024 \
  --codebook_sizes 1024 \
  --hidden_size 1024 \
  --input_size 1024 \
  --num_quantizers 1 \
  --batch_size 1024 \
  --max_epochs 50 \
  --devices 1 \
  --warm_up_epochs 1 \
  --gpu_ids 0 \
  --seed 42 \
  --learning_rate 5e-5 \
  --num_workers 4 \
  --grad_accum 1 \
  --tb_name lvqvae_std_sts
