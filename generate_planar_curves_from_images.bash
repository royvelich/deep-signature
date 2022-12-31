#!/bin/bash
cd "$(dirname "$0")"
conda activate deep-signature
export PYTHONPATH=.

python ./applets/generation/generate_planar_curves_from_images.py \
--seed 42 \
--max_tasks $3 \
--images_base_dir_path /data/images/$1 \
--results_base_dir_path /data/curves/$1 \
--min_points_count 300 \
--max_points_count 5000 \
--kernel_sizes 5 9 17 33 \
--contour_levels 0.3 0.5 0.7 \
--num_workers $2 \
--min_equiaffine_std 0.05 \
--smoothing_iterations 6 \
--smoothing_window_length 99 \
--smoothing_poly_order 2 \
--flat_point_threshold 0.0001 \
--max_flat_points_ratio 0.2