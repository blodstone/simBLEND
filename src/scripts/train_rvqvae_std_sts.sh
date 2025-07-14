#!/bin/bash

source ./secrets.sh
uv run python ../train_rvqvae.py \
  --train_path $BASE_DIR/outputs/mind/train_mind_std_sts_aspect_vectors.txt \
  --dev_path $BASE_DIR/outputs/mind/dev_mind_std_sts_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/src/checkpoints \
  --commitment_cost 0.35124988627298076 \
  --reset_threshold 0 \
  --codebook_dim 128 \
  --decay 0.91 \
  --codebook_sizes 297 \
  --hidden_size 512 \
  --input_size 1024 \
  --num_quantizers 1 \
  --batch_size 512 \
  --max_epochs 50 \
  --devices 1 \
  --warm_up_epochs 1 \
  --gpu_ids 1 \
  --seed 42 \
  --learning_rate 3.429251075957754e-05 \
  --num_workers 4 \
  --grad_accum 1 \
  --tb_name rvqvae_std_sts
