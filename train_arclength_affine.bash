#!/bin/bash

cd /deep-signature

export PYTHONPATH=.

/opt/conda/envs/deep-signature/bin/python applets/level_curves/train/train_arclength.py \
--group "affine" \
--data-dir "/deep-signature-data" \
--train-batch-size 400000 \
--validation-batch-size 200000
