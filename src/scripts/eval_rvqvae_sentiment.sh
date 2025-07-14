#!/bin/bash

source ./secrets.sh
uv run python ../eval_rvqvae.py \
  --test_path $BASE_DIR/outputs/mind/dev_mind_sentiment_aspect_vectors.txt \
  --checkpoint_path $BASE_DIR/checkpoints/rvqvae_sentiment_sts-epoch=10-val_loss=0.40279.ckpt\
  --codebook_dim 512 \
  --codebook_sizes 117 \
  --hidden_size 256 \
  --input_size 1024 \
  --num_quantizers 1 \
  --batch_size 2048 \
  --devices 1 \
  --gpu_ids 1\
