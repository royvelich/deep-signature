# deep-signatures

## Run Instructions

1. Download and install miniconda form https://docs.conda.io/en/latest/miniconda.html.
2. Create a conda environment by running `conda env create -f environment.yml`.
3. Activate the newly created conda environment by running `conda activate deep-signature`.
4. Download the contents of the file stored in https://we.tl/t-9jGcqYKhAD into the folder that is defined by the variable `data_dir` in the file `settings.py` (feel free to modify).
5. Make sure that the values of `arclength_default_num_workers_train`, `arclength_default_num_workers_validation`, `curvature_default_num_workers_train`, and `arclength_default_num_workers_validation` in `settings.py` do not exceed the maximal number of processor cores on your machine.
6. To train a curvature approximation model, run `python.exe ./applets/level_curves/train/train_curvature.py --group GROUP`, where `GROUP` can take the values `euclidean`, `equiaffine`, or `affine`.
7. To train an arc-length approximation model, run `python.exe ./applets/level_curves/train/train_arclength.py --group GROUP`, where `GROUP` can take the values `euclidean`, `equiaffine`, or `affine`.
8. Evaluate results by running `qualitative_evaluation.ipynb`.
