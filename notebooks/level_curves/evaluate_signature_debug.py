# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
#  # ** GLOBAL SETTINGS: **

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
#  # ** IMPORT PACKAGES **

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
#  # ** SANITY CHECK - CURVES **

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
#  # ** EVALUATE ARC-LENGTH **

# %%
# constants
limit = 2
arclength_sample_points = 40
step = 40
comparision_curves_count = 1
device = torch.device('cuda')

# if we're in the equiaffine case, snap 'step' to the closest mutiple of 3 (from above)
if transform_type == "equiaffine":
    step = int(3 * numpy.ceil(step / 3))

# package settings
torch.set_default_dtype(torch.float64)
# numpy.random.seed(60)

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
#  ## ** EVALUATE ARC-LENGTH NUMERICALLY **

# %%
# # sample_points = 40
# # supporting_points_count = 40
# # max_offset = 4
# # limit = 40
# # numpy.random.seed(60)

# # torch.set_default_dtype(torch.float64)
# # device = torch.device('cuda')
# # model = DeepSignatureArcLengthNet(sample_points=sample_points).cuda()
# # model.load_state_dict(torch.load(results['model_file_path'], map_location=device))
# # # model.load_state_dict(torch.load("C:/deep-signature-data/level-curves/results/tuplets/arclength/2021-01-14-02-42-52/model_349.pt", map_location=device))
# # model.eval()

# # curves = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_train)
# # numpy.random.shuffle(curves)
# # curves = curves[:limit]
# # color_map = plt.get_cmap('rainbow', limit)

# # for curve_index, curve in enumerate(curves):
# #     if curve_index == 25:
# #         break
# #     fig, ax = plt.subplots(1, 1, figsize=(5,5))
# #     ax.axis('equal')
# #     for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# #         label.set_fontsize(10)

# #     # plot predicted curvature
# #     predicted_arclength = numpy.zeros(curve.shape[0])
# #     for i in range(curve.shape[0]):
# #         if i == 1:
# #             break

# #         sample1_org = curve_sampling.sample_curve_section(
# #             curve=curve,
# #             supporting_points_count=sample_points,
# #             start_point_index=i,
# #             end_point_index=i+supporting_points_count - 1)
# #         sample1 = curve_processing.normalize_curve(curve=sample1_org, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
# #         batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()

# #         sample2_org = curve_sampling.sample_curve_section(
# #             curve=curve,
# #             supporting_points_count=sample_points,
# #             start_point_index=i+supporting_points_count - 1,
# #             end_point_index=i+2*supporting_points_count - 2)
# #         sample2 = curve_processing.normalize_curve(curve=sample2_org, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
# #         batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2).double(), dim=0), dim=0).cuda()

# #         sample3_org = curve_sampling.sample_curve_section(
# #             curve=curve,
# #             supporting_points_count=sample_points,
# #             start_point_index=i,
# #             end_point_index=i+2*supporting_points_count - 2)
# #         sample3 = curve_processing.normalize_curve(curve=sample3_org, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
# #         batch_data3 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample3).double(), dim=0), dim=0).cuda()

# #         sample4_org = curve_sampling.sample_curve_section(
# #             curve=curve,
# #             supporting_points_count=sample_points,
# #             start_point_index=i,
# #             end_point_index=i+supporting_points_count)
# #         sample4 = curve_processing.normalize_curve(curve=sample4_org, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
# #         batch_data4 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample4).double(), dim=0), dim=0).cuda()

# #         sample5_org = curve_sampling.sample_curve_section(
# #             curve=curve,
# #             supporting_points_count=sample_points,
# #             start_point_index=i,
# #             end_point_index=i+2*supporting_points_count)
# #         sample5 = curve_processing.normalize_curve(curve=sample5_org, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
# #         batch_data5 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample5).double(), dim=0), dim=0).cuda()

# #         print('------------ 4 + 5 -----------')
# #         with torch.no_grad():
# #             s1 = torch.squeeze(model(batch_data4), dim=0).cpu().detach().numpy()
# #             s2 = torch.squeeze(model(batch_data5), dim=0).cpu().detach().numpy()
# #             print(s1)
# #             print(s2)
# #             print(2 * s1)
# #             print('-----------------------')

# #         plot_sample(
# #             ax=ax, 
# #             sample=sample1, 
# #             point_size=20,
# #             color='lightcoral',
# #             alpha=0.5,
# #             zorder=50)

# #         plot_sample(
# #             ax=ax, 
# #             sample=sample2, 
# #             point_size=20,
# #             color='skyblue',
# #             alpha=0.5,
# #             zorder=50)

# #         plot_sample(
# #             ax=ax, 
# #             sample=sample3, 
# #             point_size=20,
# #             color='springgreen',
# #             zorder=150)

# #         plot_sample(ax, numpy.array([[sample1[0,0] ,sample1[0, 1]]]), point_size=50, alpha=1, color='white', zorder=200)

# #         with torch.no_grad():
# #             s1 = torch.squeeze(model(batch_data1), dim=0).cpu().detach().numpy()
# #             s2 = torch.squeeze(model(batch_data2), dim=0).cpu().detach().numpy()
# #             s3 = torch.squeeze(model(batch_data3), dim=0).cpu().detach().numpy()
# #             print(s1)
# #             print(s2)
# #             print(s1+s2)
# #             print(s3)
# #             print('-----------------------')

# #         for j in range(60):
# #             sample1 = curve_sampling.sample_curve_section2(
# #                 curve=curve,
# #                 supporting_points_count=sample_points,
# #                 start_point_index=i,
# #                 end_point_index=i+supporting_points_count + j)
# #             sample1 = curve_processing.normalize_curve(curve=sample1, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
# #             batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()
# #             with torch.no_grad():
# #                 s1 = torch.squeeze(model(batch_data1), dim=0).cpu().detach().numpy()
# #                 print(s1)
            
# #         print('-----------------------')

# #     plt.show()

# colors = ['red', 'green']

# # for each curve
# for curve_index, curve in enumerate(curves):
#     comparision_curves = [curve_processing.center_curve(curve=curve)]
#     for i in range(comparision_curves_count):
#         if transform_type == 'euclidean':
#             transform = euclidean_transform.random_euclidean_transform_2d()
#         elif transform_type == 'equiaffine':
#             transform = affine_transform.random_equiaffine_transform_2d()
#         transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
#         comparision_curves.append(curve_processing.center_curve(curve=transformed_curve))

#     comparision_curves_data = []
#     for comparision_curve in comparision_curves:
#         sampled_sections, full_sections = notebook_utils.extract_curve_section_pairs(
#             curve=comparision_curve, 
#             step=step, 
#             sample_points=arclength_sample_points)

#         comparision_curves_data.append({
#             'curve': comparision_curve,
#             'sampled_sections': sampled_sections,
#             'full_sections': full_sections
#         })

#     fig, axes = plt.subplots(3, 1, figsize=(20,20))
#     fig.patch.set_facecolor('white')
#     axes[0].axis('equal')
#     for axis in axes:
#         for label in (axis.get_xticklabels() + axis.get_yticklabels()):
#             label.set_fontsize(10)

#     for i, comparision_curve_data in enumerate(comparision_curves_data):
#         notebook_utils.plot_sectioned_curve(
#             ax=axes[0], 
#             curve=comparision_curve_data['curve'], 
#             sampled_sections=comparision_curve_data['sampled_sections'],
#             sampled_section_color=colors[i],
#             curve_color='black',
#             anchor_color='black',
#             first_anchor_color='magenta')

#     true_arclengths = []
#     for comparision_curve_data in comparision_curves_data:
#         true_arclength = notebook_utils.calculate_arclength_by_index(
#             curve=comparision_curve_data['curve'], 
#             full_sections=comparision_curve_data['full_sections'], 
#             transform_type=transform_type,
#             modifier='calabi')
#         true_arclengths.append(true_arclength)

#     predicted_arclengths = []
#     for comparision_curve_data in comparision_curves_data:
#         predicted_arclength = notebook_utils.predict_arclength_by_index(
#             model=arclength_model, 
#             sampled_sections=comparision_curve_data['sampled_sections'])
#         predicted_arclengths.append(predicted_arclength)

#     notebook_utils.plot_arclength_comparision_by_index(
#         plt=plt, 
#         ax=axes[1], 
#         true_arclengths=true_arclengths, 
#         predicted_arclengths=predicted_arclengths, 
#         colors=['orange', 'black', 'red', 'green'])


#     # curve = comparision_curves_data[0]['curve']
#     # curvature = numpy.cbrt(numpy.abs(curve_processing.calculate_euclidean_curvature(curve=curve)))
#     # notebook_utils.plot_curvature(ax=axes[2], curvature=curvature)

#     plt.show()

# %% [markdown]
#  ## ** EVALUATE ARC-LENGTH VISUALLY **

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



    # axes_list=axes_list,
    # axis_index=0,
    # evaluated_curves_arclength=evaluated_curves_arclength,
    # sample_colors=sample_colors,
    # curve_color='black',
    # anchor_color='black',
    # first_anchor_color='magenta')

# axes_list = []
# for i in range(len(evaluated_curves_arclength)):
#     fig, axes = plt.subplots(2, 1, figsize=(20,20))
#     fig.patch.set_facecolor('white')
#     axes[0].axis('equal')
#     for axis in axes:
#         for label in (axis.get_xticklabels() + axis.get_yticklabels()):
#             label.set_fontsize(10)

#     axes_list.append(axes)

# notebook_utils.plot_sectioned_curve(
#     axes_list=axes_list,
#     axis_index=0,
#     evaluated_curves_arclength=evaluated_curves_arclength,
#     sample_colors=sample_colors,
#     curve_color='black',
#     anchor_color='black',
#     first_anchor_color='magenta')

# notebook_utils.plot_arclength_by_index(
#     axes_list=axes_list,
#     axis_index=1,
#     evaluated_curves_arclength=evaluated_curves_arclength,
#     true_arclength_colors=true_arclength_colors,
#     predicted_arclength_colors=sample_colors)

# plt.show()

# d = {
#     'True [i, i+1]': true_arclengths[0][1:, 1, 1],
#     'True [i+1, i+2]': true_arclengths[0][1:, 1, 2],
#     'True [i, i+2]': true_arclengths[0][1:, 1, 3],
#     'True [i, i+1] + [i+1, i+2]': true_arclengths[0][1:, 1, 1] + true_arclengths[0][1:, 1, 2],
#     'Pred [i, i+1]': predicted_arclengths[0][1:, 1, 1],
#     'Pred [i+1, i+2]': predicted_arclengths[0][1:, 1, 2],
#     'Pred [i, i+2]': predicted_arclengths[0][1:, 1, 3],
#     'Pred [i, i+1] + [i+1, i+2]': predicted_arclengths[0][1:, 1, 1] + predicted_arclengths[0][1:, 1, 2]
# }

# df = pandas.DataFrame(data=d)

# style = df.style.set_properties(**{'background-color': '#AA0000'}, subset=list(d.keys())[:4])
# style = style.set_properties(**{'background-color': '#0000AA'}, subset=list(d.keys())[4:8])
# style = style.set_properties(**{'color': 'white', 'border-color': 'black','border-style' :'solid' ,'border-width': '1px'})

# display(HTML(style.render()))              

    # curve = comparision_curves_data[0]['curve']
    # curvature = numpy.cbrt(numpy.abs(curve_processing.calculate_euclidean_curvature(curve=curve)))
    # notebook_utils.plot_curvature(ax=axes[2], curvature=curvature)

    



# %%



