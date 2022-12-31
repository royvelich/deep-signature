#!/bin/bash
cd "$(dirname "$0")"
conda activate deep-signature
export PYTHONPATH=.

python ./applets/training/train_signature.py \
--seed 42 \
--results_base_dir_path "/data/training" \
--training_curves_file_path "/data/curves/train/2022-12-31-15-12-51/curves.npy" \
--validation_curves_file_path "/data/curves/validation/2022-12-31-13-06-53/curves.npy" \
--test_curves_file_path "/data/curves/test/2022-12-31-13-36-14/curves.npy" \
--evaluation_benchmark_dir_path "/data/curves/benchmark/2022-12-31-20-42-33" \
--wandb_dir "/data/sweeps/output_dir" \
--wandb_cache_dir "/data/sweeps/cache_dir" \
--wandb_config_dir "/data/sweeps/config_dir" \
--training_num_workers $1 \
--validation_num_workers $2 \
--evaluation_num_workers $3 \
--sweep_id $4
