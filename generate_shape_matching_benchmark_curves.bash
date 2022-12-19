#!/bin/bash
cd "$(dirname "$0")"
conda activate deep-signature
export PYTHONPATH=.

--seed
42
--curves_base_dir_path
C:/deep-signature-data-new/curves/silhouettes/2022-12-16-01-06-26
--benchmark_base_dir_path
C:/deep-signature-data-new/curves/benchmark
--sampling_ratios
1
0.9
0.8
0.7
0.6
0.5
--multimodalities
5
10
--groups
euclidean
equiaffine
similarity
affine
--min_det
1.5
--max_det
3
--min_cond
1.5
--max_cond
3
--fig_size
10
10
--point_size
0.5
--num_workers
15