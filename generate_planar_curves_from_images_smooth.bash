#!/bin/bash
cd "$(dirname "$0")"
conda activate deep-signature
export PYTHONPATH=.

python ./applets/generation/generate_planar_curves_from_images.py \
--seed 42 \
--max_tasks $2 \
--images_base_dir_path /data/images \
--results_base_dir_path /data/curves/corr_exp \
--min_points_count 1500 \
--max_points_count 6000 \
--kernel_sizes 3 5 9 17 33 \
--contour_levels 0.1 0.3 0.5 0.7 0.9 \
--num_workers $1 \
--min_equiaffine_std 0.1 \
--smoothing_iterations 6 \
--smoothing_window_length 99 \
--smoothing_poly_order 2 \
--flat_point_threshold 0.001 \
--max_flat_points_ratio 0.2