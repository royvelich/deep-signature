# python peripherals
import random
import os
import sys
import math
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

# numpy
import numpy

# pandas
import pandas

# ipython
from IPython.display import display, HTML

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines

# pytorch
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader

# deep signature
from deep_signature.utils import utils
from deep_signature.data_generation.curve_generation import LevelCurvesGenerator
from deep_signature.data_manipulation import curve_processing
from deep_signature.nn.datasets import DeepSignatureTupletsDataset
from deep_signature.nn.networks import DeepSignatureArcLengthNet
from deep_signature.nn.networks import DeepSignatureCurvatureNet
from deep_signature.nn.losses import ContrastiveLoss
from deep_signature.nn.trainers import ModelTrainer
from deep_signature.data_manipulation import curve_sampling
from deep_signature.data_manipulation import curve_processing
from deep_signature.linalg import euclidean_transform
from deep_signature.linalg import affine_transform

# common
from common import settings
from common import utils as common_utils

# notebooks
from notebooks.utils import utils as notebook_utils






# plt.style.use("dark_background")

transform_type = 'equiaffine'

if transform_type == 'euclidean':
    level_curves_arclength_tuplets_dir_path = settings.level_curves_euclidean_arclength_tuplets_dir_path
    level_curves_arclength_tuplets_results_dir_path = settings.level_curves_euclidean_arclength_tuplets_results_dir_path
elif transform_type == 'equiaffine':
    level_curves_arclength_tuplets_dir_path = settings.level_curves_equiaffine_arclength_tuplets_dir_path
    level_curves_arclength_tuplets_results_dir_path = settings.level_curves_equiaffine_arclength_tuplets_results_dir_path
elif transform_type == 'affine':
    level_curves_arclength_tuplets_dir_path = settings.level_curves_affine_arclength_tuplets_dir_path
    level_curves_arclength_tuplets_results_dir_path = settings.level_curves_affine_arclength_tuplets_results_dir_path

if transform_type == 'euclidean':
    level_curves_curvature_tuplets_dir_path = settings.level_curves_euclidean_curvature_tuplets_dir_path
    level_curves_curvature_tuplets_results_dir_path = settings.level_curves_euclidean_curvature_tuplets_results_dir_path
elif transform_type == 'equiaffine':
    level_curves_curvature_tuplets_dir_path = settings.level_curves_equiaffine_curvature_tuplets_dir_path
    level_curves_curvature_tuplets_results_dir_path = settings.level_curves_equiaffine_curvature_tuplets_results_dir_path
elif transform_type == 'affine':
    level_curves_curvature_tuplets_dir_path = settings.level_curves_affine_curvature_tuplets_dir_path
    level_curves_curvature_tuplets_results_dir_path = settings.level_curves_affine_curvature_tuplets_results_dir_path




import warnings
warnings.filterwarnings("ignore")

# constants
true_arclength_colors = ['#FF8C00', '#444444']
predicted_arclength_colors = ['#AA0000', '#00AA00']
sample_colors = ['#AA0000', '#00AA00']
curve_colors = ['#AA0000', '#00AA00']

limit = 5
step = 60
comparison_curves_count = 1

section_supporting_points_count = 20
neighborhood_supporting_points_count = 3

curvature_sample_points = 2*neighborhood_supporting_points_count + 1
arclength_sample_points = section_supporting_points_count

sampling_ratio = 0.2
anchors_ratio = 0.2

device = torch.device('cuda')

import warnings
warnings.filterwarnings("ignore")

# constants
true_arclength_colors = ['#FF8C00', '#444444']
predicted_arclength_colors = ['#AA0000', '#00AA00']
sample_colors = ['#AA0000', '#00AA00']
curve_colors = ['#AA0000', '#00AA00', '#0000AA']
limit = 1
factor_extraction_limit = -2
# step = settings.arclength_default_supporting_points_count - 1
step = 10
comparison_curves_count = 2
sampling_ratio = 0.2
anchors_ratio = 0.3

device = torch.device('cuda')

# if we're in the equiaffine case, snap 'step' to the closest mutiple of 3 (from above)
# if transform_type == "equiaffine":
#     step = int(3 * numpy.ceil(step / 3))

# package settings
torch.set_default_dtype(torch.float64)
numpy.random.seed(60)

# create models
arclength_model = DeepSignatureArcLengthNet(sample_points=settings.arclength_default_supporting_points_count).cuda()
curvature_model = DeepSignatureCurvatureNet(sample_points=settings.curvature_default_sample_points_count).cuda()

# load arclength model state
latest_subdir = common_utils.get_latest_subdirectory(level_curves_arclength_tuplets_results_dir_path)
results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
arclength_model.load_state_dict(torch.load(results['model_file_path'], map_location=device))
arclength_model.eval()

# load curvature model state
latest_subdir = common_utils.get_latest_subdirectory(level_curves_curvature_tuplets_results_dir_path)
results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
curvature_model.load_state_dict(torch.load(results['model_file_path'], map_location=device))
curvature_model.eval()

# load curves (+ shuffle)
# curves_train = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_train)
# curves_validation = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_validation)
curves = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_test)

# print(len(curves_train))
# print(len(curves_validation))
# print(len(curves_test))

numpy.random.shuffle(curves)
curves_limited = curves[:limit]
factor_extraction_curves = curves[factor_extraction_limit:]

# create color map
color_map = plt.get_cmap('rainbow', limit)

# generate curve records
curve_records = notebook_utils.generate_curve_records(
    arclength_model=arclength_model,
    curvature_model=curvature_model,
    curves=curves_limited,
    factor_extraction_curves=factor_extraction_curves,
    transform_type=transform_type,
    comparison_curves_count=comparison_curves_count,
    sampling_ratio=sampling_ratio,
    anchors_ratio=anchors_ratio,
    step=step,
    neighborhood_supporting_points_count=settings.curvature_default_supporting_points_count,
    section_supporting_points_count=settings.arclength_default_supporting_points_count)

# notebook_utils.plot_curve_signature_comparisons(
#     curve_records=curve_records,
#     curve_colors=curve_colors)
#
# notebook_utils.plot_curve_arclength_records(
#     curve_records=curve_records,
#     true_arclength_colors=true_arclength_colors,
#     predicted_arclength_colors=predicted_arclength_colors,
#     sample_colors=sample_colors)
