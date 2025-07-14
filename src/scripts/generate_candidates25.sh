#!/bin/bash

source ./secrets.sh
PREDICTION_PATH=$BASE_DIR/outputs/mind/seq_prediction_beam_25.txt
declare -a VQAE_PATHS=(
    "/mount/arbeitsdaten66/projekte/multiview/hardy/project/vae/checkpoints/rvqvae_std_sts-epoch=12-val_loss=0.72483.ckpt"
    "/mount/arbeitsdaten66/projekte/multiview/hardy/project/vae/checkpoints/rvqvae_cat_sts-epoch=12-val_loss=0.44667.ckpt"
    "/mount/arbeitsdaten66/projekte/multiview/hardy/project/vae/checkpoints/rvqvae_frame_sts-epoch=18-val_loss=0.17045.ckpt"
    "/mount/arbeitsdaten66/projekte/multiview/hardy/project/vae/checkpoints/rvqvae_political_sts-epoch=07-val_loss=0.27821.ckpt"
    "/mount/arbeitsdaten66/projekte/multiview/hardy/project/vae/checkpoints/rvqvae_sentiment_sts-epoch=10-val_loss=0.40279.ckpt"
)

CODEBOOK_SIZES="414 69 106 69 117"
CODEBOOK_DIMS="512 512 512 128 512"
HIDDEN_SIZES="128 256 1024 128 256"
ASPECT_VECTORS_PATH=$BASE_DIR/outputs/mind/dev_mind_std_sts_aspect_vectors.txt
MILVUS_DATABASE_PATH=$BASE_DIR/outputs/mind/aspect_data_new.db
COLLECTION_NAME="dev_mind_2019_11_14"
OUTPUT_PATH=$BASE_DIR/outputs/mind/seq_prediction_beam_25_cands.pickle
N_CANDS=14
N_WORKERS=15
GPU_IDS="0" 
uv run python ../generate_candidates.py \
    --prediction_path "$PREDICTION_PATH" \
    --vqvae_paths ${VQAE_PATHS[@]} \
    --codebook_sizes $CODEBOOK_SIZES \
    --codebook_dims $CODEBOOK_DIMS \
    --hidden_sizes $HIDDEN_SIZES \
    --n_cands $N_CANDS \
    --n_workers $N_WORKERS \
    --aspect_vectors_path "$ASPECT_VECTORS_PATH" \
    --milvus_database_path "$MILVUS_DATABASE_PATH" \
    --collection_name "$COLLECTION_NAME" \
    --output_path "$OUTPUT_PATH" \
    --gpu_ids "$GPU_IDS"