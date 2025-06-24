#!/bin/bash
source ./secrets.sh

uv run python ../cand_generation.py \
--aspects_config_path $BASE_DIR/configs/aspect_cat.yaml \
--decoder_config_path $BASE_DIR/configs/seqvqvae.yaml \
--vqvae_config_paths $BASE_DIR/configs/vqvae.yaml \
--recsys_config_path $BASE_DIR/configs/lstur.yaml \
--grouped_articles_path $DATASET_DIR/mind/MINDlarge_dev/grouped_behaviors.tsv \
--output_path $BASE_DIR/outputs/test/output.json \
--news_path $DATASET_DIR/mind/MINDlarge_train/news.tsv \
--behavior_path $DATASET_DIR/mind/MINDlarge_train/behaviors.tsv