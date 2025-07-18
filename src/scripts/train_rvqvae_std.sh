#!/bin/bash

source ./secrets.sh
uv run python ../train_rvqvae.py \
  --train_path $BASE_DIR/outputs/mind/train_mind_std_aspect_vectors.txt \
  --dev_path $BASE_DIR/outputs/mind/dev_mind_std_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/src/checkpoints \
  --codebook_dim 256 \
  --codebook_size 768 \
  --hidden_size 128 \
  --input_size 1024 \
  --num_quantizers 1\
  --batch_size 256 \
  --max_epochs 20 \
  --devices 1 \
  --warm_up_epochs 1 \
  --gpu_ids 4,6 \
  --seed 42 \
  --learning_rate 3e-5 \
  --num_workers 30 \
  --grad_accum 1 \
  --tb_name rvqvae_std
