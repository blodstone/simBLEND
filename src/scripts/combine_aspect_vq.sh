#!/bin/bash

source ./secrets.sh
uv run python ../combine_aspect_vq.py \
--output_folder $BASE_DIR/outputs/mind \
--output_name mind_std mind_category mind_frame mind_political mind_sentiment \
--codebook_sizes 768 32 256 32 32 