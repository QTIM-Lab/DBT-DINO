#!/bin/bash

cd "$(dirname "$0")/.."

python detection_deit/detection_deit_eval.py \
    --batch_size 64 \
    --accum_iter 1 \
    --epochs 50 \
    --lr 0.0001 \
    --csv_path /path/to/detection_dataset.csv \
    --data_dir /path/to/detection_data_dir \
    --output_dir /path/to/output_dir \
    --log_dir /path/to/log_dir \
    --seed 42 \
    --num_workers 3 \
    --pin_mem \
    --study_name detection_deit_test \
    --dataset_fraction 1 \
    --use_optuna \
    --n_trials 100 \
    --checkpoint_folder /path/to/dbtdino_detection.ckpt \
    --all_slices