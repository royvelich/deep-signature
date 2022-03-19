#!/bin/bash

cd /deep-signature

export PYTHONPATH=.

/opt/conda/envs/deep-signature/bin/python applets/level_curves/train/train_arclength.py \
--group "affine" \
--data-dir "/deep-signature-data" \
--train-batch-size 50000 \
--validation-batch-size 50000 \
--supporting-points-count 5 \
--learning-rate 0.001
