#!/bin/bash

cd "$(dirname "$0")/.."

python density_lin_probe/density_lin_probe.py \
    --batch_size 64 \
    --accum_iter 1 \
    --epochs 75 \
    --model dino_dbt \
    --task density \
    --nb_classes 4 \
    --lr 0.001 \
    --csv_path /path/to/dataset.csv \
    --data_dir /path/to/data_dir \
    --embedding_dir /path/to/embeddings \
    --output_dir /path/to/output_dir \
    --log_dir /path/to/log_dir \
    --seed 42 \
    --num_workers 3 \
    --pin_mem \
    --n_trials 24 \
    --study_name dino_linprobe \
    --dataset_fraction 1 \
    --use_optuna \
    --embedding_type patch \
    --stats mean std \
    --backbone_checkpoint /path/to/backbone_checkpoint.pth
