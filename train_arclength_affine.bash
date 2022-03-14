#!/bin/bash

cd /workspace/deep-signature

export PYTHONPATH=.

/opt/conda/envs/deep-signature/bin/python applets/level_curves/train/train_arclength.py \
--group "affine" \
--level-curves-euclidean-arclength-tuplets-results-dir-path '/deep-signature-data/level-curves/results/tuplets/arclength/euclidean' \
--level-curves-similarity-arclength-tuplets-results-dir-path '/deep-signature-data/level-curves/results/tuplets/arclength/similarity' \
--level-curves-equiaffine-arclength-tuplets-results-dir-path '/deep-signature-data/level-curves/results/tuplets/arclength/equiaffine' \
--level-curves-affine-arclength-tuplets_results-dir-path '/deep-signature-data/level-curves/results/tuplets/arclength/affine' \
--level-curves-dir-path-train '/deep-signature-data/level-curves/curves/train' \
--level-curves-dir-path-validation '/deep-signature-data/level-curves/curves/validation'
