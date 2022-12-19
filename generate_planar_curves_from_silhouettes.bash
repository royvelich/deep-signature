#!/bin/bash
cd "$(dirname "$0")"
conda activate deep-signature
export PYTHONPATH=.

--images_base_dir_path
C:/deep-signature-data-new/vector-images/export/1x
--curves_base_dir_path
C:/deep-signature-data-new/curves/silhouettes
--min_points_count
300
--max_points_count
7000
--contour_level
0.7
--num_workers
10
--seed
42