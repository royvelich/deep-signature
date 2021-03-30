# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # ** IMPORT PACKAGES **

# %%
# python peripherals
import random
import os
import sys
import math
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

# numpy
import numpy

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# common
from common import settings
from common import utils as common_utils

# notebooks
from notebooks.utils import utils as notebook_utils

# %% [markdown]
# # ** GLOBAL SETTINGS: **

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
# # ** SANITY CHECK - CURVES **

# %%
curves = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_train)
limit = 10
color_map = plt.get_cmap('rainbow', limit)
for i, curve in enumerate(curves[:limit]): 
    fig, ax = plt.subplots(1, 1, figsize=(80,40))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(30)
    ax.axis('equal')
    notebook_utils.plot_curve(ax=ax, curve=curve, linewidth=5)
    plt.show()

# %% [markdown]
# # ** EVALUATE ARC-LENGTH **

# %%
# constants
limit = 40
arclength_sample_points = 40
step = 40
device = torch.device('cuda')

# if we're in the equiaffine case, snap 'step' to the closest mutiple of 3 (from above)
if transform_type == "equiaffine":
    step = int(3 * numpy.ceil(step / 3))

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

# for each curve
for curve_index, curve in enumerate(curves):
    indices = list(range(curve.shape[0]))[::step]
    sampled_segments = []
    full_segments = []
    for index1, index2 in zip(indices, indices[1:]):
        sampled_indices = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=arclength_sample_points,
            start_point_index=index1,
            end_point_index=index2)

        sampled_segment = {
            'indices': sampled_indices,
            'sample': curve[sampled_indices]
        }

        sampled_segments.append(sampled_segment)

        full_indices = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=step,
            start_point_index=index1,
            end_point_index=index2)

        full_segment = {
            'indices': full_indices,
            'sample': curve[full_indices]
        }

        full_segments.append(full_segment)

    fig, axes = plt.subplots(2, 1, figsize=(20,20))
    axes[0].axis('equal')

    for axis in axes:
        for label in (axis.get_xticklabels() + axis.get_yticklabels()):
            label.set_fontsize(10)

    for i, sampled_segment in enumerate(sampled_segments):
        sample = sampled_segment['sample']
        # plot_curve(ax=ax, curve=curve, color=color_map(curve_index), linewidth=5)
        axes[0].set_xlabel('X Coordinate', fontsize=18)
        axes[0].set_ylabel('Y Coordinate', fontsize=18)
        notebook_utils.plot_curve(ax=axes[0], curve=curve, color='orange', linewidth=3)
        notebook_utils.plot_sample(ax=axes[0], sample=sample, point_size=10, color='red', zorder=150)
        notebook_utils.plot_sample(ax=axes[0], sample=numpy.array([[sample[0,0] ,sample[0, 1]], [sample[-1,0] ,sample[-1, 1]]]), point_size=70, alpha=1, color='blue', zorder=200)
        if i == 0:
            notebook_utils.plot_sample(ax=axes[0], sample=numpy.array([[sample[0,0] ,sample[0, 1]]]), point_size=70, alpha=1, color='black', zorder=300) 

    true_arclength = numpy.zeros([len(indices), 2])
    predicted_arclength = numpy.zeros([len(indices), 2])

    for i, full_segment in enumerate(full_segments):
        if transform_type == 'equiaffine':
            segment_indices = list(full_segment['indices'])
            left_index = numpy.mod(segment_indices[0] - 1, curve.shape[0])
            right_index = numpy.mod(segment_indices[-1] + 1, curve.shape[0])
            segment_indices = numpy.concatenate((numpy.array([left_index]), numpy.array(segment_indices), numpy.array([right_index])))
            sample = curve[segment_indices]
        else:
            sample = full_segment['sample']
        
        point_index = i+1
        true_arclength[point_index, 0] = point_index

        if transform_type == 'euclidean':
            true_arclength[point_index, 1] = curve_processing.calculate_euclidean_arclength(curve=sample)[-1] + true_arclength[i, 1]
        elif transform_type == 'equiaffine':
            true_arclength[point_index, 1] = curve_processing.calculate_equiaffine_arclength(curve=sample) + true_arclength[i, 1]

    for i, sampled_segment in enumerate(sampled_segments):
        sample = sampled_segment['sample']
        indices = list(sampled_segment['indices'])
        point_index = i+1
        sample = curve_processing.normalize_curve(curve=sample, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
        arclength_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample).double(), dim=0), dim=0).cuda()
        with torch.no_grad():
            predicted_arclength[point_index, 0] = point_index
            predicted_arclength[point_index, 1] = torch.squeeze(arclength_model(arclength_batch_data), dim=0).cpu().detach().numpy() + predicted_arclength[i, 1]

    axes[1].set_xlabel('Index', fontsize=18)
    axes[1].set_ylabel('Arc-Length', fontsize=18)
    notebook_utils.plot_sample(ax=axes[1], sample=true_arclength, point_size=40, color='orange', zorder=250)
    notebook_utils.plot_curve(ax=axes[1], curve=true_arclength, linewidth=2, color='orange', zorder=150)

    factor = numpy.mean(true_arclength[1:, 1] / predicted_arclength[1:, 1])
    if numpy.isnan(factor):
        factor = 1000
        
    predicted_arclength[:, 1] = predicted_arclength[:, 1] * factor
    notebook_utils.plot_sample(ax=axes[1], sample=predicted_arclength, point_size=40, color='green', zorder=250)
    notebook_utils.plot_curve(ax=axes[1], curve=predicted_arclength, linewidth=2, color='green', zorder=150)

    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.show()

# %% [markdown]
# # ** EVALUATE SIGNATURE **

# %%
limit = 2
curvature_supporting_points = 6
curvature_sample_points = 2 * curvature_supporting_points + 1
arclength_sample_points = 40
step = 120

torch.set_default_dtype(torch.float64)
device = torch.device('cuda')
numpy.random.seed(60)

curvature_model = DeepSignatureCurvatureNet(sample_points=curvature_sample_points).cuda()
arclength_model = DeepSignatureArcLengthNet(sample_points=arclength_sample_points).cuda()

latest_subdir = common_utils.get_latest_subdirectory(settings.level_curves_curvature_tuplets_results_dir_path)
results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
curvature_model.load_state_dict(torch.load(results['model_file_path'], map_location=device))
curvature_model.eval()

latest_subdir = common_utils.get_latest_subdirectory(settings.level_curves_arclength_tuplets_results_dir_path)
results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
arclength_model.load_state_dict(torch.load(results['model_file_path'], map_location=device))
arclength_model.eval()

curves = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_train)
numpy.random.shuffle(curves)
curves = curves[:limit]
color_map = plt.get_cmap('rainbow', limit)

for _ in range(3):
    for curve_index, curve in enumerate(curves):
        indices = list(range(curve.shape[0]))[::step]
        sampled_segments = []
        for index1, index2 in zip(indices, indices[1:]):
            sampled_indices = curve_sampling.sample_curve_section_indices(
                curve=curve,
                supporting_points_count=arclength_sample_points,
                start_point_index=index1,
                end_point_index=index2)

            sampled_segment = {
                'indices': sampled_indices,
                'sample': curve[sampled_indices]
            }

            sampled_segments.append(sampled_segment)
            # print(sampled_segment['sample'].shape[0])
            # print(sampled_segment['indices'])

        
        fig, axes = plt.subplots(2, 1, figsize=(20,20))
        axes[0].axis('equal')

        for axis in axes:
            for label in (axis.get_xticklabels() + axis.get_yticklabels()):
                label.set_fontsize(10)

        for i, sampled_segment in enumerate(sampled_segments):
            sample = sampled_segment['sample']
            # plot_curve(ax=ax, curve=curve, color=color_map(curve_index), linewidth=5)
            axes[0].set_xlabel('X Coordinate', fontsize=18)
            axes[0].set_ylabel('Y Coordinate', fontsize=18)
            notebook_utils.plot_curve(ax=axes[0], curve=curve, color='orange', linewidth=3)
            notebook_utils.plot_sample(ax=axes[0], sample=sample, point_size=10, color='red', zorder=150)
            notebook_utils.plot_sample(ax=axes[0], sample=numpy.array([[sample[0,0] ,sample[0, 1]], [sample[-1,0] ,sample[-1, 1]]]), point_size=70, alpha=1, color='blue', zorder=200)
            if i == 0:
                notebook_utils.plot_sample(ax=axes[0], sample=numpy.array([[sample[0,0] ,sample[0, 1]]]), point_size=70, alpha=1, color='black', zorder=300) 

        signature = numpy.zeros([len(indices) - 2, 2])
        for i, [sampled_segment1, sampled_segment2] in enumerate(zip(sampled_segments, sampled_segments[1:])):
            arclength_sample1 = sampled_segment1['sample']
            arclength_sample2 = sampled_segment2['sample']
            indices1 = list(sampled_segment1['indices'])
            indices2 = list(sampled_segment2['indices'])
            curvature_indices = indices1[(len(indices1) - curvature_supporting_points - 1):len(indices1)] + indices2[:curvature_supporting_points]
            curvature_sample = curve[curvature_indices]

            arclength_sample = curve_processing.normalize_curve(curve=arclength_sample1, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
            arclength_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(arclength_sample).double(), dim=0), dim=0).cuda()

            curvature_sample = curve_processing.normalize_curve(curve=curvature_sample)
            curvature_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(curvature_sample).double(), dim=0), dim=0).cuda()

            with torch.no_grad():
                signature[i, 0] = torch.squeeze(arclength_model(arclength_batch_data), dim=0).cpu().detach().numpy() + signature[i - 1, 0]
                signature[i, 1] = torch.squeeze(curvature_model(curvature_batch_data), dim=0).cpu().detach().numpy()

        # print(signature)
        axes[1].set_xlabel('Arc-Length', fontsize=18)
        axes[1].set_ylabel('Curvature', fontsize=18)
        notebook_utils.plot_sample(ax=axes[1], sample=signature, point_size=40, color='blue', zorder=250)
        notebook_utils.plot_curve(ax=axes[1], curve=signature, linewidth=2, color='orange', zorder=150)
        plt.show()


# %%



