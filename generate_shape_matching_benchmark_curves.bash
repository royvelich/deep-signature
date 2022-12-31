#!/bin/bash
cd "$(dirname "$0")"
conda activate deep-signature
export PYTHONPATH=.

python ./applets/generation/generate_shape_matching_curves.py \
--seed 42 \
--curves_base_dir_path $1 \
--results_base_dir_path $2 \
--sampling_ratios 1 0.9 0.8 0.7 0.6 0.5 \
--multimodalities 5 10 \
--groups euclidean equiaffine similarity affine \
--min_det 1.5 \
--max_det 3 \
--min_cond 1.5 \
--max_cond 3 \
--fig_size 10 10 \
--point_size 0.5 \
--num_workers $3