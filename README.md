# deep-signatures

## Run Instructions

1. Download and install miniconda form https://docs.conda.io/en/latest/miniconda.html.
2. Create a conda environment by running `conda env create -f environment.yml`.
3. Download the contents of the OneDrive folder https://we.tl/t-q3MsgOIyqu into the folder that is defined by the variable `data_dir` in the file `settings.py`.
4. Make sure that the values of `arclength_default_num_workers_train`, `arclength_default_num_workers_validation`, `curvature_default_num_workers_train`, and `arclength_default_num_workers_validation` in `settings.py` do not exceed the maximal number of processor cores on your machine.
5. To train a curvature approximation model, run `python.exe ./applets/level_curves/train/train_curvature.py --group GROUP`, where `GROUP` can take the values `euclidean`, `equiaffine`, or `affine`.
6. To train an arc-length approximation model, run `python.exe ./applets/level_curves/train/train_arclength.py --group GROUP`, where `GROUP` can take the values `euclidean`, `equiaffine`, or `affine`.
7. Evaluate results by running `evaluate_metrics.ipynb`.
