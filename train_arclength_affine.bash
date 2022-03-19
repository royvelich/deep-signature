#!/bin/bash

cd /deep-signature

export PYTHONPATH=.

/opt/conda/envs/deep-signature/bin/python applets/level_curves/train/train_arclength.py \
--group "affine" \
--data-dir "/deep-signature-data" \
--train-batch-size 256000 \
--validation-batch-size 100000 \
--supporting-points-count 5 \
--learning-rate 0.001
