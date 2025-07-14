#!/bin/bash

source ./secrets.sh
uv run python ../eval_rvqvae.py \
  --test_path $BASE_DIR/outputs/mind/dev_mind_frame_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/checkpoints/rvqvae_frame_sts-epoch=18-val_loss=0.17045.ckpt \
  --codebook_dim 512 \
  --codebook_sizes 106 \
  --hidden_size 1024 \
  --input_size 1024 \
  --num_quantizers 1 \
  --batch_size 2048 \
  --devices 1 \
  --gpu_ids 1\
