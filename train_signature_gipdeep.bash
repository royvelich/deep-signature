#!/bin/bash
cd "$(dirname "$0")"
conda activate deep-signature
export PYTHONPATH=.

python ./applets/training/train_signature.py \
--seed 42 \
--wandb_dir "/home/royve/data/sweeps/output_dir" \
--wandb_cache_dir "/home/royve/data/sweeps/cache_dir" \
--wandb_config_dir "/home/royve/data/sweeps/config_dir" \
--results_base_dir_path "/home/royve/data/training" \
--training_curves_file_path "/home/royve/data/curves/train/2022-12-31-15-12-51/curves.npy" \
--validation_curves_file_path "/home/royve/data/curves/validation/2022-12-31-13-06-53/curves.npy" \
--test_curves_file_path "/home/royve/data/curves/test/2022-12-31-13-36-14/curves.npy" \
--group_name "euclidean" \
--supporting_points_count 7 \
--negative_examples_count 2 \
--min_sampling_ratio 0.2 \
--max_sampling_ratio 0.7 \
--min_multimodality 5 \
--max_multimodality 20 \
--min_negative_example_offset 20 \
--max_negative_example_offset 50 \
--training_dataset_size 1000 \
--validation_dataset_size 1000 \
--training_num_workers 2 \
--validation_num_workers 2 \
--validation_items_queue_maxsize 2000 \
--training_items_queue_maxsize 2000 \
--training_items_buffer_size 1000 \
--validation_items_buffer_size 1000 \
--evaluation_num_workers 35 \
--evaluation_benchmark_dir_path "/home/royve/data/curves/benchmark/2023-01-01-13-36-51" \
--evaluation_curves_count_per_collection 10 \
--evaluation_curve_collections_file_names "basketball" \
--evaluation_sampling_ratios 0.9 \
--evaluation_multimodality 5 \
--epochs 200 \
--training_batch_size 1000 \
--validation_batch_size 1000 \
--learning_rate 0.01 \
--checkpoint_rate 50 \
--history_size 800 \
--activation_fn "sine" \
--batch_norm_fn "BatchNorm1d" \
--in_features_size 64 \
--hidden_layer_repetitions 2 \
--training_min_det 1 \
--training_max_det 4 \
--training_min_cond 1 \
--training_max_cond 4 \
--validation_min_det 2 \
--validation_max_det 2 \
--validation_min_cond 2 \
--validation_max_cond 2