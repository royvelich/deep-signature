# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
#   # ** GLOBAL SETTINGS **

# %%
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

# %% [markdown]
#   # ** IMPORT PACKAGES **

# %%
# plt.style.use("dark_background")

transform_type = 'equiaffine'

if transform_type == 'euclidean':
    level_curves_arclength_tuplets_dir_path = settings.level_curves_euclidean_arclength_tuplets_dir_path
    level_curves_arclength_tuplets_results_dir_path = settings.level_curves_euclidean_arclength_tuplets_results_dir_path
elif transform_type == 'equiaffine':
    level_curves_arclength_tuplets_dir_path = settings.level_curves_equiaffine_arclength_tuplets_dir_path
    level_curves_arclength_tuplets_results_dir_path = settings.level_curves_equiaffine_arclength_tuplets_results_dir_path

if transform_type == 'euclidean':
    level_curves_curvature_tuplets_dir_path = settings.level_curves_euclidean_curvature_tuplets_dir_path
    level_curves_curvature_tuplets_results_dir_path = settings.level_curves_euclidean_curvature_tuplets_results_dir_path
elif transform_type == 'equiaffine':
    level_curves_curvature_tuplets_dir_path = settings.level_curves_equiaffine_curvature_tuplets_dir_path
    level_curves_curvature_tuplets_results_dir_path = settings.level_curves_equiaffine_curvature_tuplets_results_dir_path

# %% [markdown]
#   # ** CURVES **

# %%
# curves = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_train)
# limit = 10
# color_map = plt.get_cmap('rainbow', limit)
# for i, curve in enumerate(curves[:limit]): 
#     fig, ax = plt.subplots(1, 1, figsize=(80,40))
#     for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#         label.set_fontsize(30)
#     ax.axis('equal')
#     notebook_utils.plot_curve(ax=ax, curve=curve, linewidth=5)
#     plt.show()

# %% [markdown]
#   # ** ARC-LENGTH **
# %% [markdown]
# ## INITIALIZATION ##

# %%
# constants
limit = 10
arclength_sample_points = 40
step = 40
comparision_curves_count = 1
device = torch.device('cuda')

# if we're in the equiaffine case, snap 'step' to the closest mutiple of 3 (from above)
# if transform_type == "equiaffine":
#     step = int(3 * numpy.ceil(step / 3))

# package settings
torch.set_default_dtype(torch.float64)
numpy.random.seed(60)

# create model
arclength_model = DeepSignatureArcLengthNet(sample_points=arclength_sample_points).cuda()

# load model state
latest_subdir = common_utils.get_latest_subdirectory(level_curves_arclength_tuplets_results_dir_path)
results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
arclength_model.load_state_dict(torch.load(results['model_file_path'], map_location=device))
arclength_model.eval()

# load curves (+ shuffle)
curves = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_train)
numpy.random.shuffle(curves)
curves = curves[:limit]

# create color map
color_map = plt.get_cmap('rainbow', limit)

# %% [markdown]
#   ## ** EVALUATE ARC-LENGTH **

# %%
true_arclength_colors = ['#FF8C00', '#444444']
predicted_arclength_colors = ['#AA0000', '#00AA00']
sample_colors = ['#AA0000', '#00AA00']

curve_arclength_records = notebook_utils.generate_curve_arclength_records(
    model=arclength_model, 
    curves=curves,
    transform_type=transform_type,
    comparision_curves_count=comparision_curves_count,
    step=step,
    sample_points=arclength_sample_points)

notebook_utils.plot_curve_arclength_records(
    curve_arclength_records=curve_arclength_records, 
    true_arclength_colors=true_arclength_colors, 
    predicted_arclength_colors=predicted_arclength_colors, 
    sample_colors=sample_colors, 
    curve_color='#FF8C00', 
    anchor_color='#3333FF', 
    first_anchor_color='#FF0FF0')


# %%



