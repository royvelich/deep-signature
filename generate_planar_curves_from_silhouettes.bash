#!/bin/bash
cd "$(dirname "$0")"
conda activate deep-signature
export PYTHONPATH=.

python ./applets/generation/generate_planar_curves_from_silhouettes.py \
--images_base_dir_path $1 \
--results_base_dir_path $2 \
--min_points_count 200 \
--max_points_count 7000 \
--contour_level 0.7 \
--num_workers $3 \
--seed 42