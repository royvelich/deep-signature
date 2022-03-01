#!/bin/bash

conda activate deep-signature

cat /proc/cpuinfo | grep 'core id'

cd /workspace/deep-signature

export PYTHONPATH=.

/opt/conda/envs/deep-signature/bin/python applets/train/train_arclength.py \
--group "affine" \
--level_curves_euclidean_arclength_tuplets_results_dir_path 'deep-signature/level-curves/results/tuplets/arclength/euclidean' \
--level_curves_similarity_arclength_tuplets_results_dir_path 'deep-signature/level-curves/results/tuplets/arclength/similarity' \
--level_curves_equiaffine_arclength_tuplets_results_dir_path 'deep-signature/level-curves/results/tuplets/arclength/equiaffine' \
--level_curves_affine_arclength_tuplets_results_dir_path 'deep-signature/level-curves/results/tuplets/arclength/affine' \
--level_curves_dir_path_train 'deep-signature/level-curves/curves/train' \
--level_curves_dir_path_validation 'deep-signature/level-curves/curves/validation' \
