#!/bin/bash

cd "$(dirname "$0")/.."

python risk_lin_probe/risk_lin_probe.py \
    --batch_size 64 \
    --accum_iter 1 \
    --epochs 100 \
    --model dinov2_vitb14 \
    --embedding_type patch \
    --stats mean std \
    --task risk \
    --max_followup 5 \
    --lr 1e-05 \
    --lr_schedule cosine \
    --weight_decay 0.0001 \
    --csv_path /path/to/dataset.csv \
    --embedding_dir_cancer /path/to/embeddings_cancer \
    --embedding_dir_healthy /path/to/embeddings_healthy \
    --output_dir /path/to/output_dir \
    --log_dir /path/to/log_dir \
    --seed 42 \
    --num_workers 3 \
    --pin_mem \
    --n_trials 24 \
    --study_name risk_linprobe \
    --cumulative_prob sum \
    --validation balanced \
    --mode test \
    --checkpoint_path /path/to/dbtdino_risk.ckpt
