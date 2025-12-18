#!/bin/bash

cd "$(dirname "$0")/.."

python risk_lin_probe/compute_embeddings.py \
    --csv_path /path/to/dataset.csv \
    --token_0 $TOKEN_0 \
    --token_1 $TOKEN_1 \
    --cohort healthy \
    --train \
    --val \
    --test \
    --pin_mem \
    --num_workers 16 \
    --model_name dbtdino \
    --output_dir /path/to/output_dir \
    --streaming \
    --all_views_bilat \
    --backbone_checkpoint /path/to/backbone_checkpoint.pth