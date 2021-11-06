# python peripherals
import random
import pathlib
import os

# scipy
import scipy.io
import scipy.stats as ss

# numpy
import numpy

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.ticker as ticker
import matplotlib.lines

# pytorch
import torch

# pandas
import pandas

# ipython
from IPython.display import display, HTML

# deep signature
from deep_signature.data_manipulation import curve_sampling
from deep_signature.data_manipulation import curve_processing
from deep_signature.linalg import euclidean_transform
from deep_signature.linalg import affine_transform
from deep_signature.utils import utils
from common import settings

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines


# plotly
from plotly.subplots import make_subplots
from plotly import graph_objects


# https://stackoverflow.com/questions/36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar
from deep_signature.stats import discrete_distribution


# ---------------
# PLOTLY ROUTINES
# ---------------
def plot_dist_plotly(fig, row, col, dist, line_width=2, line_color='black', point_size=10, cmap='hsv'):
    x = numpy.array(range(dist.shape[0]))
    y = dist

    fig.add_trace(
        trace=graph_objects.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line={
               'color': line_color,
               'width': line_width
            },
            marker={
                'color': x,
                'colorscale': cmap,
                'size': point_size
            },
            customdata=x,
            hovertemplate='%{customdata}'),
        row=row,
        col=col)


def plot_curve_sample_plotly(fig, row, col, name, curve, curve_sample, color, point_size=5, color_scale='hsv'):
    x = curve_sample[:, 0]
    y = curve_sample[:, 1]

    index_colors = isinstance(color, (list, numpy.ndarray))

    fig.add_trace(
        trace=graph_objects.Scatter(
            name=name,
            x=x,
            y=y,
            mode='markers',
            marker={
                'color': color,
                'cmin': 0,
                'cmax': curve.shape[0],
                'colorscale': color_scale,
                'size': point_size
            },
            customdata=color if index_colors else None,
            hovertemplate='%{customdata}' if index_colors else None,
            hoverinfo='skip' if not index_colors else None),
        row=row,
        col=col)


def plot_curve_plotly(fig, row, col, curve, name, line_width=2, line_color='green'):
    x = curve[:, 0]
    y = curve[:, 1]

    fig.add_trace(
        trace=graph_objects.Scatter(
            name=name,
            x=x,
            y=y,
            mode='lines+markers',
            line={
                'color': line_color,
                'width': line_width
            }),
        row=row,
        col=col)


def plot_curvature_plotly(fig, row, col, name, curvature, line_width=2, line_color='green'):
    x = numpy.array(range(curvature.shape[0]))
    y = curvature

    fig.add_trace(
        trace=graph_objects.Scatter(
            name=name,
            x=x,
            y=y,
            mode='lines+markers',
            line={
                'color': line_color,
                'width': line_width
            },
            marker={
                'color': line_color,
            }),
        row=row,
        col=col)


def plot_arclength_plotly(fig, row, col, name, arclength, line_width=2, line_color='green'):
    x = numpy.array(range(arclength.shape[0]))
    y = arclength

    fig.add_trace(
        trace=graph_objects.Scatter(
            name=name,
            x=x,
            y=y,
            mode='lines+markers',
            line={
                'color': line_color,
                'width': line_width
            },
            marker={
                'color': line_color,
            }),
        row=row,
        col=col)


def plot_curvature_with_cmap_plotly(fig, row, col, name, curvature, curve, indices, line_color='black', line_width=2, point_size=5, color_scale='hsv'):
    x = numpy.array(range(curvature.shape[0]))
    y = curvature

    fig.add_trace(
        trace=graph_objects.Scatter(
            name=name,
            x=x,
            y=y,
            mode='lines+markers',
            line={
                'color': line_color,
                'width': line_width
            },
            marker={
                'color': indices,
                'cmin': 0,
                'cmax': curve.shape[0],
                'colorscale': color_scale,
                'size': point_size
            },
            customdata=indices,
            hovertemplate='%{customdata}'),
        row=row,
        col=col)

# -------------------
# MATPLOTLIB ROUTINES
# -------------------
def colorline(ax, x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = numpy.array([z])

    z = numpy.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    # ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_dist(ax, dist):
    x = numpy.array(range(dist.shape[0]))
    y = dist
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    return colorline(ax=ax, x=x, y=y, cmap='hsv')


def plot_curve_sample(ax, curve, curve_sample, indices, zorder, point_size=10, alpha=1, cmap='hsv'):
    x = curve_sample[:, 0]
    y = curve_sample[:, 1]
    c = numpy.linspace(0.0, 1.0, curve.shape[0])

    return ax.scatter(
        x=x,
        y=y,
        c=c[indices],
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        norm=plt.Normalize(0.0, 1.0),
        zorder=zorder)


def plot_curve_section_center_point(ax, x, y, zorder, radius=1, color='white'):
    circle = plt.Circle((x, y), radius=radius, color=color, zorder=zorder)
    return ax.add_artist(circle)


def plot_graph(ax, x, y, linewidth=2, color='red', alpha=1, zorder=1):
    return ax.plot(x, y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder)


def plot_curve(ax, curve, linewidth=2, color='red', alpha=1, zorder=1):
    x = curve[:, 0]
    y = curve[:, 1]
    return plot_graph(ax=ax, x=x, y=y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder)


def plot_curvature(ax, curvature, color='red', linewidth=2, alpha=1):
    x = numpy.array(range(curvature.shape[0]))
    y = curvature

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)


def plot_curvature_with_cmap(ax, curvature, curve, indices, linewidth=2, alpha=1, cmap='hsv'):
    x = numpy.array(range(curvature.shape[0]))
    y = curvature

    c = numpy.linspace(0.0, 1.0, curve.shape[0])
    z = c[indices]

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return colorline(ax=ax, x=x, y=y, z=z, cmap='hsv')


def plot_sample(ax, sample, color, zorder, point_size=10, alpha=1, x=None, y=None):
    if sample is not None:
        x = sample[:, 0]
        y = sample[:, 1]

    return ax.scatter(
        x=x,
        y=y,
        s=point_size,
        color=color,
        alpha=alpha,
        zorder=zorder)


# ---------------------
# GROUND TRUTH ROUTINES
# ---------------------
# def calculate_arclength_by_index(curve_sections, transform_type, modifier=None):
#     curve = curve_sections['curve']
#     full_sections = curve_sections['full_sections']
#     true_arclength = numpy.zeros([len(full_sections) + 1, 2, 4])
#     for i, full_section in enumerate(full_sections):
#         point_index = i + 1
#         for j, (indices, sample, accumulate) in enumerate(zip(full_section['indices'], full_section['samples'], full_section['accumulate'])):
#             true_arclength[point_index, 0, j] = point_index
#             if transform_type == 'equiaffine':
#                 if modifier == 'calabi':
#                     left_indices = numpy.mod(numpy.array([indices[0] - 1]), curve.shape[0])
#                     right_indices = numpy.mod(numpy.array([indices[-1] + 1]), curve.shape[0])
#                     segment_indices = numpy.concatenate((left_indices, indices, right_indices))
#                     sample = curve[segment_indices]
#                 else:
#                     left_indices = numpy.mod(numpy.array([indices[0] - 2, indices[0] - 1]), curve.shape[0])
#                     right_indices = numpy.mod(numpy.array([indices[-1] + 1, indices[-1] + 2]), curve.shape[0])
#                     segment_indices = numpy.concatenate((left_indices, indices, right_indices))
#                     sample = curve[segment_indices]
#
#             if transform_type == 'euclidean':
#                 true_arclength[point_index, 1, j] = curve_processing.calculate_euclidean_arclength(curve=sample)[-1]
#             elif transform_type == 'equiaffine':
#                 if modifier == 'calabi':
#                     true_arclength[point_index, 1, j] = curve_processing.calculate_equiaffine_arclength(curve=sample)[-1]
#                 else:
#                     true_arclength[point_index, 1, j] = curve_processing.calculate_equiaffine_arclength_by_euclidean_metrics(curve=sample)[-1]
#
#             if accumulate is True:
#                 true_arclength[point_index, 1, j] = true_arclength[point_index, 1, j] + true_arclength[i, 1, j]
#
#     return true_arclength


def calculate_arclength_by_index(curve, anchor_indices, transform_type, modifier=None):
    true_arclength = None
    if transform_type == 'euclidean':
        true_arclength = curve_processing.calculate_euclidean_arclength(curve=curve)
    elif transform_type == 'equiaffine':
        if modifier == 'calabi':
            true_arclength = curve_processing.calculate_equiaffine_arclength(curve=curve)
        else:
            true_arclength = curve_processing.calculate_equiaffine_arclength_by_euclidean_metrics(curve=curve)

    indices = numpy.array(list(range(anchor_indices.shape[0])))
    values = true_arclength[anchor_indices]
    return numpy.vstack((indices, values)).transpose()


# def calculate_arclength_by_index2(curve_sections, transform_type, modifier=None):
#     curve = curve_sections['curve']
#     sampled_extended_sections = curve_sections['sampled_extended_sections']
#     true_arclength = numpy.zeros([len(sampled_extended_sections) + 1, 2])
#     for i, sampled_extended_section in enumerate(sampled_extended_sections):
#         point_index = i + 1
#
#         indices1 = sampled_extended_section['indices'][0]
#         indices2 = sampled_extended_section['indices'][1]
#         # sample1 = curve_processing.normalize_curve(curve=sampled_section['samples'][0])
#         # sample2 = curve_processing.normalize_curve(curve=sampled_section['samples'][1])
#
#         # arclength_batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()
#         # arclength_batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2).double(), dim=0), dim=0).cuda()
#
#         with torch.no_grad():
#             true_arclength[point_index, 0] = point_index
#             if transform_type == 'euclidean':
#                 sample1 = curve[indices1]
#                 sample2 = curve[indices2]
#                 true_arclength[point_index, 1] = numpy.abs(curve_processing.calculate_euclidean_arclength(curve=sample1)[-1] - curve_processing.calculate_euclidean_arclength(curve=sample2)[-1])
#             elif transform_type == 'equiaffine':
#                 left_indices1 = numpy.mod(numpy.array([indices1[0] - 2, indices1[0] - 1]), curve.shape[0])
#                 right_indices1 = numpy.mod(numpy.array([indices1[-1] + 1, indices1[-1] + 2]), curve.shape[0])
#
#                 left_indices2 = numpy.mod(numpy.array([indices2[0] - 2, indices2[0] - 1]), curve.shape[0])
#                 right_indices2 = numpy.mod(numpy.array([indices2[-1] + 1, indices2[-1] + 2]), curve.shape[0])
#
#                 segment_indices1 = numpy.concatenate((left_indices1, indices1, right_indices1))
#                 segment_indices2 = numpy.concatenate((left_indices2, indices2, right_indices2))
#                 sample1 = curve[segment_indices1]
#                 sample2 = curve[segment_indices2]
#                 true_arclength[point_index, 1] = numpy.abs(curve_processing.calculate_equiaffine_arclength_by_euclidean_metrics(curve=sample1)[-1] - curve_processing.calculate_equiaffine_arclength_by_euclidean_metrics(curve=sample2)[-1])
#
#         true_arclength[point_index, 1] = true_arclength[point_index, 1] + true_arclength[i, 1]
#
#     return true_arclength


def calculate_curvature_by_index(curve, transform_type):
    true_curvature = numpy.zeros([curve.shape[0], 2])
    true_curvature[:, 0] = numpy.arange(curve.shape[0])

    if transform_type == 'euclidean':
        true_curvature[:, 1] = curve_processing.calculate_euclidean_curvature(curve=curve)
    elif transform_type == 'equiaffine':
        true_curvature[:, 1] = curve_processing.calculate_equiaffine_curvature(curve=curve)
    elif transform_type == 'affine':
        true_curvature[:, 1] = 0

    return true_curvature


# -------------------
# PREDICTION ROUTINES
# -------------------
def predict_curvature_by_index(model, curve_neighborhoods):
    sampled_neighborhoods = curve_neighborhoods['sampled_neighborhoods']
    predicted_curvature = numpy.zeros([len(sampled_neighborhoods), 2])
    for point_index, sampled_neighborhood in enumerate(sampled_neighborhoods):
        for (indices, sample) in zip(sampled_neighborhood['indices'], sampled_neighborhood['samples']):
            sample = curve_processing.normalize_curve(curve=sample)
            curvature_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample).double(), dim=0), dim=0).cuda()
            with torch.no_grad():
                predicted_curvature[point_index, 0] = point_index
                predicted_curvature[point_index, 1] = torch.squeeze(model(curvature_batch_data), dim=0).cpu().detach().numpy()
    return predicted_curvature


# def predict_arclength_by_index(model, curve_sections):
#     sampled_sections = curve_sections['sampled_sections']
#     predicted_arclength = numpy.zeros([len(sampled_sections) + 1, 2, 4])
#     for i, sampled_section in enumerate(sampled_sections):
#         point_index = i + 1
#         for j, (indices, sample, accumulate) in enumerate(zip(sampled_section['indices'], sampled_section['samples'], sampled_section['accumulate'])):
#             sample = curve_processing.normalize_curve(curve=sample)
#             arclength_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample).double(), dim=0), dim=0).cuda()
#             with torch.no_grad():
#                 predicted_arclength[point_index, 0, j] = point_index
#                 predicted_arclength[point_index, 1, j] = torch.squeeze(model(arclength_batch_data), dim=0).cpu().detach().numpy()
#
#             if accumulate is True:
#                 predicted_arclength[point_index, 1, j] = predicted_arclength[point_index, 1, j] + predicted_arclength[i, 1, j]
#
#     return predicted_arclength

# def predict_arclength_by_index2(model, curve_sections):
#     sampled_sections = curve_sections['sampled_extended_sections']
#     predicted_arclength = numpy.zeros([len(sampled_sections) + 1, 2])
#     for i, sampled_section in enumerate(sampled_sections):
#         point_index = i + 1
#
#         sample1 = curve_processing.normalize_curve(curve=sampled_section['samples'][0])
#         sample2 = curve_processing.normalize_curve(curve=sampled_section['samples'][1])
#
#         arclength_batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()
#         arclength_batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2).double(), dim=0), dim=0).cuda()
#
#         with torch.no_grad():
#             predicted_arclength[point_index, 0] = point_index
#             predicted_arclength[point_index, 1] = numpy.abs(torch.squeeze(model(arclength_batch_data1), dim=0).cpu().detach().numpy() - torch.squeeze(model(arclength_batch_data2), dim=0).cpu().detach().numpy())
#         predicted_arclength[point_index, 1] = predicted_arclength[point_index, 1] + predicted_arclength[i, 1]
#
#     return predicted_arclength


# def predict_arclength_by_index_new(model, curve_sections, indices_pool, anchor_indices, supporting_points_count):
#     sampled_sections = curve_sections['sampled_sections']
#     predicted_arclength = numpy.zeros([len(sampled_sections) + 1, 2])
#     arclength_at_index = {}
#     step = supporting_points_count - 1
#     curve = curve_sections['curve']
#
#     for i, sampled_section in enumerate(sampled_sections[:-1]):
#         index = i + 1
#         sample1 = curve_processing.normalize_curve(curve=sampled_section['samples'][0])
#         sample2 = curve_processing.normalize_curve(curve=sampled_section['samples'][1])
#         arclength_batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()
#         arclength_batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2).double(), dim=0), dim=0).cuda()
#
#         with torch.no_grad():
#             predicted_arclength[index, 0] = index
#             predicted_arclength[index, 1] = numpy.abs(torch.squeeze(model(arclength_batch_data1), dim=0).cpu().detach().numpy() - torch.squeeze(model(arclength_batch_data2), dim=0).cpu().detach().numpy())
#         predicted_arclength[index, 1] = predicted_arclength[index, 1] + predicted_arclength[i, 1]
#         arclength_at_index[sampled_section['reference_index']] = predicted_arclength[index, 1]
#
#     arclength_at_index[0] = 0
#
#     predicted_arclength_at_anchors = numpy.zeros([len(anchor_indices), 2])
#     for i, anchor_index in enumerate(anchor_indices[1:]):
#         modified_indices_pool = utils.insert_sorted(indices_pool, numpy.array([anchor_index]))
#         meta_index = int(numpy.where(modified_indices_pool == anchor_index)[0])
#         previous_meta_index = numpy.mod(meta_index - 1, modified_indices_pool.shape[0])
#
#         sampled_curve = curve[modified_indices_pool]
#
#         start_point_index = previous_meta_index - step
#         end_point_index = previous_meta_index
#         end_point_index2 = meta_index
#
#         sampled_indices1 = curve_sampling.sample_curve_section_indices_old(
#             curve=sampled_curve,
#             start_point_index=start_point_index,
#             end_point_index=end_point_index,
#             supporting_points_count=supporting_points_count,
#             uniform=True)
#
#         sampled_indices2 = curve_sampling.sample_curve_section_indices_old(
#             curve=sampled_curve,
#             start_point_index=start_point_index,
#             end_point_index=end_point_index2,
#             supporting_points_count=supporting_points_count,
#             uniform=True)
#
#         sampled_section1 = sampled_curve[sampled_indices1]
#         sampled_section2 = sampled_curve[sampled_indices2]
#
#         sample1 = curve_processing.normalize_curve(curve=sampled_section1)
#         sample2 = curve_processing.normalize_curve(curve=sampled_section2)
#         arclength_batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()
#         arclength_batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2).double(), dim=0), dim=0).cuda()
#
#         index = modified_indices_pool[end_point_index]
#         predicted_arclength = arclength_at_index[index]
#
#         anchor_meta_index = i+1
#         with torch.no_grad():
#             predicted_arclength_at_anchors[anchor_meta_index, 0] = anchor_meta_index
#             predicted_arclength_at_anchors[anchor_meta_index, 1] = predicted_arclength + numpy.abs(torch.squeeze(model(arclength_batch_data1), dim=0).cpu().detach().numpy() - torch.squeeze(model(arclength_batch_data2), dim=0).cpu().detach().numpy())
#
#     return predicted_arclength_at_anchors


def predict_arclength_by_index(model, curve, indices_pool, anchor_indices, supporting_points_count):
    modified_indices_pool = utils.insert_sorted(indices_pool, anchor_indices)
    sampled_curve = curve[modified_indices_pool]
    predicted_arclength = numpy.zeros(modified_indices_pool.shape[0])
    step = supporting_points_count - 1

    # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    anchor_meta_indices = numpy.where(numpy.in1d(modified_indices_pool, anchor_indices))[0]

    for meta_index in list(range(modified_indices_pool.shape[0]))[1:]:
        start_meta_index = meta_index - step
        end_meta_index = meta_index

        sampled_indices1 = curve_sampling.sample_curve_section_indices_old(
            curve=sampled_curve,
            start_point_index=start_meta_index,
            end_point_index=end_meta_index,
            supporting_points_count=supporting_points_count,
            uniform=True)

        sampled_indices2 = curve_sampling.sample_curve_section_indices_old(
            curve=sampled_curve,
            start_point_index=start_meta_index,
            end_point_index=end_meta_index+1,
            supporting_points_count=supporting_points_count,
            uniform=True)

        sampled_section1 = sampled_curve[sampled_indices1]
        sampled_section2 = sampled_curve[sampled_indices2]

        sample1 = curve_processing.normalize_curve(curve=sampled_section1)
        sample2 = curve_processing.normalize_curve(curve=sampled_section2)

        arclength_batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()
        arclength_batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2).double(), dim=0), dim=0).cuda()

        with torch.no_grad():
            predicted_arclength[meta_index] = numpy.abs(torch.squeeze(model(arclength_batch_data1), dim=0).cpu().detach().numpy() - torch.squeeze(model(arclength_batch_data2), dim=0).cpu().detach().numpy())
        predicted_arclength[meta_index] = predicted_arclength[meta_index] + predicted_arclength[meta_index-1]

    indices = numpy.array(list(range(anchor_meta_indices.shape[0])))
    values = predicted_arclength[anchor_meta_indices]
    return numpy.vstack((indices, values)).transpose()








def predict_arclength_by_index2(model, curve, indices_pool, anchor_indices, supporting_points_count):
    # modified_indices_pool = utils.insert_sorted(indices_pool, anchor_indices)
    # sampled_curve = curve[modified_indices_pool]
    # predicted_arclength = numpy.zeros(modified_indices_pool.shape[0])
    # step = supporting_points_count - 1
    #
    # # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    # anchor_meta_indices = numpy.where(numpy.in1d(modified_indices_pool, anchor_indices))[0]

    predicted_arclength = numpy.zeros(anchor_indices.shape[0])
    step = supporting_points_count - 1
    arclength_at_index = {}
    for i, anchor_index in enumerate(anchor_indices[1:]):
        modified_indices_pool = utils.insert_sorted(indices_pool, numpy.array([anchor_indices[0], anchor_index]))
        sampled_curve = curve[modified_indices_pool]
        anchor_meta_index = int(numpy.where(modified_indices_pool == anchor_index)[0])
        anchor_arclength = 0
        for meta_index in range(anchor_meta_index):
            start_meta_index = meta_index - step
            end_meta_index = meta_index

            sampled_indices1 = curve_sampling.sample_curve_section_indices_old(
                curve=sampled_curve,
                start_point_index=start_meta_index,
                end_point_index=end_meta_index,
                supporting_points_count=supporting_points_count,
                uniform=True)

            sampled_indices2 = curve_sampling.sample_curve_section_indices_old(
                curve=sampled_curve,
                start_point_index=start_meta_index,
                end_point_index=end_meta_index + 1,
                supporting_points_count=supporting_points_count,
                uniform=True)

            sampled_section1 = sampled_curve[sampled_indices1]
            sampled_section2 = sampled_curve[sampled_indices2]

            sample1 = curve_processing.normalize_curve(curve=sampled_section1)
            sample2 = curve_processing.normalize_curve(curve=sampled_section2)

            arclength_batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()
            arclength_batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2).double(), dim=0), dim=0).cuda()

            with torch.no_grad():
                anchor_arclength = anchor_arclength + numpy.abs(torch.squeeze(model(arclength_batch_data1), dim=0).cpu().detach().numpy() - torch.squeeze(model(arclength_batch_data2), dim=0).cpu().detach().numpy())

        predicted_arclength[i+1] = anchor_arclength

    #     # previous_meta_index = numpy.mod(meta_index - 1, modified_indices_pool.shape[0])
    #
    # for meta_index in list(range(modified_indices_pool.shape[0]))[1:]:
    #     start_meta_index = meta_index - step
    #     end_meta_index = meta_index
    #
    #     sampled_indices1 = curve_sampling.sample_curve_section_indices_old(
    #         curve=sampled_curve,
    #         start_point_index=start_meta_index,
    #         end_point_index=end_meta_index,
    #         supporting_points_count=supporting_points_count,
    #         uniform=True)
    #
    #     sampled_indices2 = curve_sampling.sample_curve_section_indices_old(
    #         curve=sampled_curve,
    #         start_point_index=start_meta_index,
    #         end_point_index=end_meta_index+1,
    #         supporting_points_count=supporting_points_count,
    #         uniform=True)
    #
    #     sampled_section1 = sampled_curve[sampled_indices1]
    #     sampled_section2 = sampled_curve[sampled_indices2]
    #
    #     sample1 = curve_processing.normalize_curve(curve=sampled_section1)
    #     sample2 = curve_processing.normalize_curve(curve=sampled_section2)
    #
    #     arclength_batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()
    #     arclength_batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2).double(), dim=0), dim=0).cuda()
    #
    #     with torch.no_grad():
    #         predicted_arclength[meta_index] = numpy.abs(torch.squeeze(model(arclength_batch_data1), dim=0).cpu().detach().numpy() - torch.squeeze(model(arclength_batch_data2), dim=0).cpu().detach().numpy())
    #     predicted_arclength[meta_index] = predicted_arclength[meta_index] + predicted_arclength[meta_index-1]

    indices = numpy.array(list(range(predicted_arclength.shape[0])))
    values = predicted_arclength
    return numpy.vstack((indices, values)).transpose()






# --------------------------
# RECORD GENERATION ROUTINES
# --------------------------
def generate_curve_records(arclength_model, curvature_model, curves, factor_extraction_curves, transform_type, comparison_curves_count, sampling_ratio, anchors_ratio, step, neighborhood_supporting_points_count, section_supporting_points_count):
    curve_records = []
    for curve_index, curve in enumerate(curves):
        comparison_curves = []
        for i in range(comparison_curves_count):
            if transform_type == 'euclidean':
                transform = euclidean_transform.generate_random_euclidean_transform_2d()
            elif transform_type == 'equiaffine':
                transform = affine_transform.generate_random_equiaffine_transform_2d()
            elif transform_type == 'affine':
                transform = affine_transform.generate_random_affine_transform_2d()
            transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
            comparison_curves.append(curve_processing.center_curve(curve=transformed_curve))

        curve_record = {
            'curve': curve_processing.center_curve(curve=curve),
            'comparisons': []
        }

        anchor_indices = numpy.linspace(start=0, stop=curve.shape[0], num=int(anchors_ratio * curve.shape[0]), endpoint=False, dtype=int)
        for i, comparison_curve in enumerate(comparison_curves):
            comparison_curve_points_count = comparison_curve.shape[0]
            sampling_points_count = int(sampling_ratio * comparison_curve_points_count)
            dist = discrete_distribution.random_discrete_dist(bins=comparison_curve_points_count, multimodality=60, max_density=1, count=1)[0]
            indices_pool = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=sampling_points_count)
            # modified_indices_pool = utils.insert_sorted(indices_pool, numpy.array([anchor_indices[0]]))

            curve_neighborhoods = extract_curve_neighborhoods(
                curve=comparison_curve,
                indices_pool=indices_pool,
                sampling_points_count=sampling_points_count,
                supporting_points_count=neighborhood_supporting_points_count,
                anchor_indices=anchor_indices)

            # curve_sections_new = extract_curve_sections_new(
            #     curve=comparison_curve,
            #     indices_pool=modified_indices_pool,
            #     supporting_points_count=section_supporting_points_count)

            # curve_sections = extract_curve_sections(
            #     curve=comparison_curve,
            #     step=step,
            #     sample_points=section_supporting_points_count)

            # curve_neighborhoods_from_sections = extract_curve_neighborhoods_from_curve_sections(
            #     curve=comparison_curve,
            #     curve_sections=curve_sections,
            #     supporting_points_count=neighborhood_supporting_points_count)

            true_arclength = calculate_arclength_by_index(
                curve=comparison_curve,
                anchor_indices=anchor_indices,
                transform_type=transform_type)

            # true_arclength2 = calculate_arclength_by_index2(
            #     curve_sections=curve_sections,
            #     transform_type=transform_type)

            # predicted_arclength = predict_arclength_by_index_new(
            #     model=arclength_model,
            #     curve_sections=curve_sections_new,
            #     indices_pool=modified_indices_pool,
            #     anchor_indices=anchor_indices,
            #     supporting_points_count=section_supporting_points_count)

            predicted_arclength = predict_arclength_by_index2(
                model=arclength_model,
                curve=comparison_curve,
                indices_pool=indices_pool,
                anchor_indices=anchor_indices,
                supporting_points_count=section_supporting_points_count)

            #     factor = numpy.mean(true_arclength[1:, 1, 0] / predicted_arclength[1:, 1, 0])
            #     factors.append(factor)

            # if transform_type != 'affine':
            #     factor = numpy.mean(true_arclength[1:, 1] / predicted_arclength[1:, 1])
            #     predicted_arclength[:, 1] *= factor

            # predicted_arclength2 = predict_arclength_by_index2(
            #     model=arclength_model,
            #     curve_sections=curve_sections)

            true_curvature = calculate_curvature_by_index(
                curve=curve,
                transform_type=transform_type)

            predicted_curvature = predict_curvature_by_index(
                model=curvature_model,
                curve_neighborhoods=curve_neighborhoods)

            # predicted_curvature_signature = predict_curvature_by_index(
            #     model=curvature_model,
            #     curve_neighborhoods=curve_neighborhoods_from_sections)

            # print(predicted_arclength2.shape)
            # print(curve.shape)

            sampled_indices = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=sampling_points_count)
            sampled_curve = comparison_curve[sampled_indices]
            anchors = comparison_curve[anchor_indices]

            arclength_comparison = {
                # 'curve_sections': curve_sections,
                'true_arclength': true_arclength,
                # 'true_arclength2': true_arclength2,
                'predicted_arclength': predicted_arclength,
                # 'predicted_arclength2': predicted_arclength2,
                # 'predicted_arclength_original': predicted_arclength.copy()
            }

            curvature_comparison = {
                'curve_neighborhoods': curve_neighborhoods,
                'true_curvature': true_curvature,
                'predicted_curvature': predicted_curvature,
                # 'predicted_curvature_signature': predicted_curvature_signature
            }

            curve_record['comparisons'].append({
                'curve': comparison_curve,
                'sampled_curve': sampled_curve,
                'sampled_indices': sampled_indices,
                'anchor_indices': anchor_indices,
                'anchors': anchors,
                'dist': dist,
                'arclength_comparison': arclength_comparison,
                'curvature_comparison': curvature_comparison
            })

        curve_records.append(curve_record)

    # factors = []
    # for curve_index, curve in enumerate(factor_extraction_curves):
    #     all_indices = numpy.array(list(range(curve.shape[0])))
    #     true_arclength = calculate_arclength_by_index(
    #         curve=curve,
    #         anchor_indices=all_indices,
    #         transform_type=transform_type)
    #
    #     predicted_arclength = predict_arclength_by_index2(
    #         model=arclength_model,
    #         curve=curve,
    #         indices_pool=all_indices,
    #         anchor_indices=all_indices,
    #         supporting_points_count=section_supporting_points_count)
    #
    #     factor = numpy.mean(true_arclength[1:, 1] / predicted_arclength[1:, 1])
    #     factors.append(factor)
    #
    # if transform_type != 'affine':
    #     factor = numpy.mean(numpy.array(factors))
    #     for curve_record in curve_records:
    #         for comparison in curve_record['comparisons']:
    #             comparison['arclength_comparison']['predicted_arclength'][:, 1] *= factor
    #             # comparison['arclength_comparison']['predicted_arclength2'][:, 1] *= factor

    return curve_records


def extract_curve_neighborhoods(curve, indices_pool, sampling_points_count, supporting_points_count, anchor_indices):
    sampled_neighborhoods = []
    for anchor_index in anchor_indices:
        sampled_indices = curve_sampling.sample_curve_neighborhood_indices(
            center_point_index=anchor_index,
            indices_pool=indices_pool,
            supporting_points_count=supporting_points_count)

        sampled_neighborhood = {
            'indices': [sampled_indices],
            'samples': [curve[sampled_indices]]
        }

        sampled_neighborhoods.append(sampled_neighborhood)

    return {
        'sampled_neighborhoods': sampled_neighborhoods,
        'curve': curve
    }


def extract_curve_sections_new(curve, indices_pool, supporting_points_count):
    sampled_curve = curve[indices_pool]
    sampled_sections = []
    step = supporting_points_count - 1
    for index in range(len(indices_pool)):
        start_point_index = index - step
        end_point_index = index
        sampled_indices1 = curve_sampling.sample_curve_section_indices_old(
            curve=sampled_curve,
            start_point_index=start_point_index,
            end_point_index=end_point_index,
            supporting_points_count=supporting_points_count,
            uniform=True)

        sampled_indices2 = curve_sampling.sample_curve_section_indices_old(
            curve=sampled_curve,
            start_point_index=start_point_index,
            end_point_index=end_point_index+1,
            supporting_points_count=supporting_points_count,
            uniform=True)

        reference_pool_index = numpy.mod(end_point_index+1, indices_pool.shape[0])

        sampled_section = {
            'indices': [sampled_indices1, sampled_indices2],
            'samples': [curve[sampled_indices1], curve[sampled_indices2]],
            # 'accumulate': [False, False],
            'reference_pool_index': reference_pool_index,
            'reference_index': indices_pool[reference_pool_index]
        }

        sampled_sections.append(sampled_section)

    return {
        'sampled_sections': sampled_sections,
        'curve': curve
    }


def extract_curve_neighborhoods_from_curve_sections(curve, curve_sections, supporting_points_count):
    sampled_neighborhoods = []
    sampled_sections = [curve_sections['sampled_sections'][-1]] + curve_sections['sampled_sections']
    for sampled_section in sampled_sections:
        indices12 = sampled_section['indices'][1]
        indices23 = sampled_section['indices'][2]
        indices = numpy.concatenate((indices12[-supporting_points_count-1:-1], [indices23[0]], indices23[1:supporting_points_count+1]))

        sampled_neighborhood = {
            'indices': [indices],
            'samples': [curve[indices]]
        }

        sampled_neighborhoods.append(sampled_neighborhood)

    return {
        'sampled_neighborhoods': sampled_neighborhoods,
        'curve': curve
    }


def extract_curve_sections(curve, step, sample_points):
    indices = list(range(curve.shape[0]))[::step]
    indices[-1] = 0
    indices.append(indices[1])

    sampled_sections = []
    full_sections = []

    for index1, index2, index3 in zip(indices, indices[1:], indices[2:]):
        sampled_indices1 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index1,
            end_point_index=index2,
            supporting_points_count=sample_points,
            uniform=True)

        sampled_indices3 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index2,
            end_point_index=index3,
            supporting_points_count=sample_points,
            uniform=True)

        sampled_indices4 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index1,
            end_point_index=index3,
            supporting_points_count=sample_points,
            uniform=True)

        sampled_section = {
            'indices': [sampled_indices1, sampled_indices1, sampled_indices3, sampled_indices4],
            'samples': [curve[sampled_indices1], curve[sampled_indices1], curve[sampled_indices3], curve[sampled_indices4]],
            'accumulate': [True, False, False, False]
        }

        sampled_sections.append(sampled_section)

        full_indices1 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index1,
            end_point_index=index2,
            uniform=True)

        full_indices3 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index2,
            end_point_index=index3,
            uniform=True)

        full_indices4 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index1,
            end_point_index=index3,
            uniform=True)

        full_section = {
            'indices': [full_indices1, full_indices1, full_indices3, full_indices4],
            'samples': [curve[full_indices1], curve[full_indices1], curve[full_indices3], curve[full_indices4]],
            'accumulate': [True, False, False, False]
        }

        full_sections.append(full_section)

    sampled_extended_sections = []
    for index in list(range(curve.shape[0])):
        start_point_index = index - step
        end_point_index = index
        sampled_indices1 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=start_point_index,
            end_point_index=end_point_index,
            supporting_points_count=sample_points,
            uniform=True)

        sampled_indices2 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=start_point_index,
            end_point_index=end_point_index+1,
            supporting_points_count=sample_points,
            uniform=True)

        sampled_extended_section = {
            'indices': [sampled_indices1, sampled_indices2],
            'samples': [curve[sampled_indices1], curve[sampled_indices2]],
            'accumulate': [False, False]
        }

        sampled_extended_sections.append(sampled_extended_section)

    return {
        'sampled_sections': sampled_sections,
        'full_sections': full_sections,
        'sampled_extended_sections': sampled_extended_sections,
        'curve': curve
    }


# ----------------
# METRICS ROUTINES
# ----------------
def calculate_signature_metrics(curve_records):
    curvature_offsets = numpy.array([])
    arclength_offsets = numpy.array([])
    for i, curve_record in enumerate(curve_records):
        comparisons = curve_record['comparisons']
        arclength_comparison_ref = comparisons[0]['arclength_comparison']
        curvature_comparison_ref = comparisons[0]['curvature_comparison']
        predicted_arclength_ref = arclength_comparison_ref['predicted_arclength'][1:, 1, 0].squeeze()
        predicted_curvature_signature_ref = curvature_comparison_ref['predicted_curvature_signature'][:, 1].squeeze()
        for comparison in comparisons[1:]:
            arclength_comparison = comparison['arclength_comparison']
            curvature_comparison = comparison['curvature_comparison']
            predicted_arclength = arclength_comparison['predicted_arclength'][1:, 1, 0].squeeze()
            predicted_curvature_signature = curvature_comparison['predicted_curvature_signature'][:, 1].squeeze()
            arclength_offset = (predicted_arclength - predicted_arclength_ref) / numpy.abs(predicted_arclength_ref)
            curvature_offset = (predicted_curvature_signature - predicted_curvature_signature_ref) / numpy.abs(predicted_curvature_signature_ref)
            # print(arclength_offsets.shape)
            # print(arclength_offset.shape)
            # print(arclength_offset[:, 1, 0].squeeze().shape)
            # print(curvature_offset[:, 1].squeeze().shape)
            arclength_offsets = numpy.concatenate((arclength_offsets, arclength_offset))
            curvature_offsets = numpy.concatenate((curvature_offsets, curvature_offset))

            # print(arclength_offsets)

    return {
        'arclength_offset_mean': numpy.mean(arclength_offsets),
        'arclength_offset_std': numpy.std(arclength_offsets),
        'curvature_offset_mean': numpy.mean(curvature_offsets),
        'curvature_offset_std': numpy.std(curvature_offsets),
        'curvature_offset_min': numpy.min(curvature_offsets),
        'curvature_offset_max': numpy.max(curvature_offsets),
    }


# -------------
# PLOT ROUTINES
# -------------
def plot_curve_curvature_comparisons(curve_records, curve_colors):
    for i, curve_record in enumerate(curve_records):
        display(HTML(f'<H1>Curve {i+1} - Curvature Comparison</H1>'))
        plot_curve_curvature_comparison(
            curve_index=i,
            curve_record=curve_record,
            curve_colors=curve_colors)


def plot_curve_curvature_comparison(curve_index, curve_record, curve_colors):
    dir_name = "./curvature_comparison"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    factor = 1.3

    def get_range():
        min_val1 = numpy.abs(numpy.min(curve_record['comparisons'][0]['curve'][:, 1]))
        min_val2 = numpy.abs(numpy.min(curve_record['comparisons'][1]['curve'][:, 1]))
        min_val3 = numpy.abs(numpy.min(curve_record['curve'][:, 1]))

        max_val1 = numpy.abs(numpy.max(curve_record['comparisons'][0]['curve'][:, 1]))
        max_val2 = numpy.abs(numpy.max(curve_record['comparisons'][1]['curve'][:, 1]))
        max_val3 = numpy.abs(numpy.max(curve_record['curve'][:, 1]))

        val1 = numpy.max(numpy.array([min_val1, min_val2, min_val3]))
        val2 = numpy.max(numpy.array([max_val1, max_val2, max_val3]))

        val = numpy.maximum(val1, val2) * factor
        return [-val, val]

    def get_curve_range(comparison_index):
        if comparison_index > -1:
            curve = curve_record['comparisons'][comparison_index]['curve']
        else:
            curve = curve_record['curve']

        min_val = numpy.abs(numpy.min(curve[:, 1]))
        max_val = numpy.abs(numpy.max(curve[:, 1]))
        val = numpy.maximum(min_val, max_val) * factor
        return [-val, val]

    # ---------------------
    # PLOT CURVES TOGETHER
    # ---------------------
    fig = make_subplots(rows=1, cols=3, subplot_titles=('<b>Reference Curve</b>', '<b>Transformed Curve #1</b>', '<b>Transformed Curve #2</b>'))

    curve = curve_record['curve']
    plot_curve_plotly(fig=fig, row=1, col=1, curve=curve, name='Reference Curve', line_width=settings.plotly_graph_line_width, line_color=curve_colors[-1])

    for i, comparison in enumerate(curve_record['comparisons']):
        curve = comparison['curve']
        plot_curve_plotly(fig=fig, row=1, col=i+2, curve=curve, name=f'Transformed Curve #{i+1}', line_width=settings.plotly_graph_line_width, line_color=curve_colors[i])

    for i in range(len(curve_record['comparisons']) + 1):
        fig.update_yaxes(
            scaleanchor=f'x{i+1}',
            scaleratio=1,
            row=1,
            col=i+1)

    fig.update_layout(
        # legend=dict(
        #     orientation="v",
        #     yanchor="bottom",
        #     xanchor="right"),
        font=dict(size=settings.plotly_axis_title_label_fontsize))


    # fig.update_layout(xaxis1=dict(range=[-100, 100]))
    fig.update_layout(yaxis1=dict(range=get_range()))
    # fig.update_layout(xaxis2=dict(range=[-100, 100]))
    fig.update_layout(yaxis2=dict(range=get_range()))
    # fig.update_layout(xaxis3=dict(range=[-100, 100]))
    fig.update_layout(yaxis3=dict(range=get_range()))

    fig['layout']['xaxis']['title'] = 'X Coordinate'
    fig['layout']['yaxis']['title'] = 'Y Coordinate'

    fig['layout']['xaxis2']['title'] = 'X Coordinate'
    fig['layout']['yaxis2']['title'] = 'Y Coordinate'

    fig['layout']['xaxis3']['title'] = 'X Coordinate'
    fig['layout']['yaxis3']['title'] = 'Y Coordinate'

    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig.write_image(os.path.join(dir_name, f'curves_together_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    fig.show()

    # # ---------------------
    # # PLOT CURVES TOGETHER
    # # ---------------------
    # fig = make_subplots(rows=1, cols=2, subplot_titles=('<b>Reference Curve</b>', '<b>Transformed Curves vs. Reference Curve</b>'))
    #
    # curve = curve_record['curve']
    # plot_curve_plotly(fig=fig, row=1, col=1, curve=curve, name='Reference Curve', line_width=settings.plotly_graph_line_width, line_color=curve_colors[-1])
    # plot_curve_plotly(fig=fig, row=1, col=2, curve=curve, name='Reference Curve', line_width=settings.plotly_graph_line_width, line_color=curve_colors[-1])
    #
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     curve = comparison['curve']
    #     plot_curve_plotly(fig=fig, row=1, col=2, curve=curve, name=f'Transformed Curve #{i+1}', line_width=settings.plotly_graph_line_width, line_color=curve_colors[i])
    #
    #     fig.update_yaxes(
    #         scaleanchor=f'x{i+1}',
    #         scaleratio=1,
    #         row=1,
    #         col=i+1)
    #
    # fig.update_layout(
    #     legend=dict(
    #         orientation="v",
    #         yanchor="bottom",
    #         xanchor="right"),
    #     font=dict(size=settings.plotly_axis_title_label_fontsize))
    #
    # fig['layout']['xaxis']['title'] = 'X Coordinate'
    # fig['layout']['yaxis']['title'] = 'Y Coordinate'
    #
    # fig['layout']['xaxis2']['title'] = 'X Coordinate'
    # fig['layout']['yaxis2']['title'] = 'Y Coordinate'
    #
    # fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    #
    # fig.write_image(os.path.join(dir_name, f'curves_together_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    # fig.show()

    # -------------------------------
    # PLOT CURVE SAMPLES SIDE BY SIDE
    # -------------------------------
    fig = make_subplots(rows=1, cols=len(curve_record['comparisons']), subplot_titles=('<b>Sampled Curve #1</b>', '<b>Sampled Curve #2</b>'))

    for i, comparison in enumerate(curve_record['comparisons']):
        sampled_curve = comparison['sampled_curve']
        curve = comparison['curve']
        plot_curve_sample_plotly(fig=fig, row=1, col=i+1, name=f'Sampled Curve {i+1}', curve=curve, curve_sample=sampled_curve, color=curve_colors[i], point_size=settings.plotly_sample_point_size)

    for i in range(len(curve_record['comparisons']) + 1):
        fig.update_yaxes(
            scaleanchor=f'x{i+1}',
            scaleratio=1,
            row=1,
            col=i+1)

    fig.update_layout(
        # legend=dict(
        #     orientation="v",
        #     yanchor="bottom",
        #     xanchor="right"),
        font=dict(size=settings.plotly_axis_title_label_fontsize))
    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig['layout']['xaxis']['title'] = 'X Coordinate'
    fig['layout']['yaxis']['title'] = 'Y Coordinate'

    fig['layout']['xaxis2']['title'] = 'X Coordinate'
    fig['layout']['yaxis2']['title'] = 'Y Coordinate'

    fig.update_layout(yaxis1=dict(range=get_range()))
    fig.update_layout(yaxis2=dict(range=get_range()))

    fig.write_image(os.path.join(dir_name, f'curve_samples_side_by_side_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    fig.show()

    # ----------------------------------------------------------------------------------
    # PLOT CURVE SAMPLES, ANCHORS AND PREDICTED CURVATURE SIDE BY SIDE (WITHOUT BUTTONS)
    # ----------------------------------------------------------------------------------
    left_width = 0.25
    for i, comparison in enumerate(curve_record['comparisons']):
        fig = make_subplots(rows=1, cols=3, column_widths=[left_width, left_width, 1 - (2*left_width)], subplot_titles=('<b>Sampled Curve</b>', '<b>Equally Spaced Anchors</b>',  '<b>Predicted Curvature at Anchors</b>'))
        sampled_curve = comparison['sampled_curve']
        anchors = comparison['anchors']
        anchor_indices = comparison['anchor_indices']
        curve = comparison['curve']
        curvature_comparison = comparison['curvature_comparison']
        predicted_curvature = curvature_comparison['predicted_curvature']

        plot_curve_sample_plotly(fig=fig, row=1, col=1, name="Sampled Curve", curve=curve, curve_sample=sampled_curve, color='grey', point_size=settings.plotly_sample_point_size)
        plot_curve_sample_plotly(fig=fig, row=1, col=2, name="Anchors", curve=curve, curve_sample=anchors, color=anchor_indices, point_size=settings.plotly_sample_point_size)
        plot_curvature_with_cmap_plotly(fig=fig, row=1, col=3, name="Predicted Curvature at Anchors", curve=curve, curvature=predicted_curvature[:, 1], indices=anchor_indices, line_color='grey', line_width=settings.plotly_graph_line_width, point_size=settings.plotly_sample_anchor_size, color_scale='hsv')

        fig.update_yaxes(
            scaleanchor="x1",
            scaleratio=1,
            row=1,
            col=1)

        fig.update_yaxes(
            scaleanchor="x2",
            scaleratio=1,
            row=1,
            col=2)

        fig['layout']['xaxis']['title'] = 'X Coordinate'
        fig['layout']['yaxis']['title'] = 'Y Coordinate'

        fig['layout']['xaxis2']['title'] = 'X Coordinate'
        fig['layout']['yaxis2']['title'] = 'Y Coordinate'

        fig['layout']['xaxis3']['title'] = 'Anchor Point Index'
        fig['layout']['yaxis3']['title'] = 'Predicted Curvature'

        curr_range = get_range()
        fig.update_layout(yaxis1=dict(range=curr_range))
        fig.update_layout(yaxis2=dict(range=curr_range))

        fig.update_layout(
            # legend=dict(
            #     orientation="v",
            #     yanchor="bottom",
            #     xanchor="right"),
            font=dict(size=settings.plotly_axis_title_label_fontsize))

        fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

        fig.write_image(os.path.join(dir_name, f'curve_samples_and_predicted_curvature_{curve_index}_{i}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
        fig.show()

    # # ----------------------------------------------------------------
    # # PLOT CURVE SAMPLES, ANCHORS AND PREDICTED CURVATURE SIDE BY SIDE
    # # ----------------------------------------------------------------
    #
    # button_offset = 0.1
    # buttonX = 0.1
    # buttonY = 1.3
    # buttons_count = 2
    # left_width = 0.25
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     fig = make_subplots(rows=1, cols=2, column_widths=[left_width, 1 - left_width])
    #     sampled_curve = comparison['sampled_curve']
    #     anchors = comparison['anchors']
    #     anchor_indices = comparison['anchor_indices']
    #     curve = comparison['curve']
    #     curvature_comparison = comparison['curvature_comparison']
    #     predicted_curvature = curvature_comparison['predicted_curvature']
    #
    #     plot_curve_sample_plotly(fig=fig, row=1, col=1, name="Sampled Curve", curve=curve, curve_sample=sampled_curve, color='grey')
    #     plot_curve_sample_plotly(fig=fig, row=1, col=1, name="Anchors", curve=curve, curve_sample=anchors, color=anchor_indices, point_size=3)
    #     plot_curvature_with_cmap_plotly(fig=fig, row=1, col=2, name="Predicted Curvature at Anchors", curve=curve, curvature=predicted_curvature[:, 1], indices=anchor_indices, line_color='grey', line_width=2, point_size=10, color_scale='hsv')
    #
    #     # https://stackoverflow.com/questions/65941253/plotly-how-to-toggle-traces-with-a-button-similar-to-clicking-them-in-legend
    #     update_menus = [{} for _ in range(buttons_count)]
    #     button_labels = ['Toggle Samples', 'Toggle Anchors']
    #     for j in range(buttons_count):
    #         button = dict(method='restyle',
    #                        label=button_labels[j],
    #                        visible=True,
    #                        args=[{'visible': True}, [j]],
    #                        args2=[{'visible': False}, [j]])
    #
    #         update_menus[j]['buttons'] = [button]
    #         update_menus[j]['showactive'] = False
    #         update_menus[j]['y'] = buttonY
    #         update_menus[j]['x'] = buttonX + j * button_offset
    #         update_menus[j]['type'] = 'buttons'
    #
    #     fig.update_layout(
    #         showlegend=True,
    #         updatemenus=update_menus)
    #
    #     fig.update_yaxes(
    #         scaleanchor="x",
    #         scaleratio=1,
    #         row=1,
    #         col=1)
    #
    #     fig.update_layout(
    #         legend=dict(
    #             orientation="v",
    #             yanchor="bottom",
    #             xanchor="right"))
    #
    #     fig.show()

    # ----------------------------------
    # PLOT PREDICTED CURVATURES TOGETHER
    # ----------------------------------

    fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted Curvature at Anchors (Transformed Curve #1 vs. Transformed Curve #2)</b>',))

    for i, comparison in enumerate(curve_record['comparisons']):
        curvature_comparison = comparison['curvature_comparison']
        predicted_curvature = curvature_comparison['predicted_curvature']

        plot_curvature_plotly(fig=fig, row=1, col=1, name=f'Predicted Curvature at Anchors #{i+1}', curvature=predicted_curvature[:, 1], line_width=settings.plotly_graph_line_width, line_color=curve_colors[i])

    fig['layout']['xaxis']['title'] = 'Anchor Point Index'
    fig['layout']['yaxis']['title'] = 'Predicted Curvature'

    fig.update_layout(
        # legend=dict(
        #     orientation="v",
        #     yanchor="bottom",
        #     xanchor="right"),
        font=dict(size=settings.plotly_axis_title_label_fontsize))

    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig.write_image(os.path.join(dir_name, f'predicted_curves_together_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    fig.show()


def plot_curve_curvature_comparison2(curve_record, curve_colors):
    fig, axes = plt.subplots(2, 1, figsize=(20,20))
    fig.patch.set_facecolor('white')
    for axis in axes:
        for label in (axis.get_xticklabels() + axis.get_yticklabels()):
            label.set_fontsize(10)

    axes[0].axis('equal')
    axes[0].set_xlabel('X Coordinate', fontsize=18)
    axes[0].set_ylabel('Y Coordinate', fontsize=18)

    for i, comparision in enumerate(curve_record['comparisons']):
        curve = comparision['curve']
        plot_curve(ax=axes[0], curve=curve, color=curve_colors[i], linewidth=settings.plotly_graph_line_width)


    # axis_index = 0
    # fontsize = 25
    # axes_count = 15
    # line_width = 2
    #
    # # ---------------------
    # # PLOT CURVES TOGETHER
    # # ---------------------
    # fig = make_subplots(rows=1, cols=1)
    #
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     curve = comparison['curve']
    #     plot_curve_plotly(fig=fig, row=1, col=1, curve=curve, line_width=line_width, line_color=curve_colors[i])
    #
    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=1,
    # )
    #
    # fig.show()

    # # -------------------------------
    # # PLOT CURVE SAMPLES SIDE BY SIDE
    # # -------------------------------
    # fig = make_subplots(rows=1, cols=len(curve_record['comparisons']))
    #
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     sampled_curve = comparison['sampled_curve']
    #     curve = comparison['curve']
    #
    #     plot_curve_sample_plotly(fig=fig, row=1, col=i+1, name=f'Sampled Curve {i+1}', curve=curve, curve_sample=sampled_curve, color=curve_colors[i], point_size=3)
    #
    #     fig.update_yaxes(
    #         scaleanchor=f'x{i+1}',
    #         scaleratio=1,
    #         row=1,
    #         col=i+1)
    #
    # fig.show()

    # # ----------------------------------------------------------------
    # # PLOT CURVE SAMPLES, ANCHORS AND PREDICTED CURVATURE SIDE BY SIDE
    # # ----------------------------------------------------------------
    #
    # button_offset = 0.1
    # buttonX = 0.1
    # buttonY = 1.3
    # buttons_count = 2
    # left_width = 0.25
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     fig = make_subplots(rows=1, cols=2, column_widths=[left_width, 1 - left_width])
    #     sampled_curve = comparison['sampled_curve']
    #     anchors = comparison['anchors']
    #     anchor_indices = comparison['anchor_indices']
    #     curve = comparison['curve']
    #     curvature_comparison = comparison['curvature_comparison']
    #     predicted_curvature = curvature_comparison['predicted_curvature']
    #
    #     plot_curve_sample_plotly(fig=fig, row=1, col=1, name="Sampled Curve", curve=curve, curve_sample=sampled_curve, color='grey')
    #     plot_curve_sample_plotly(fig=fig, row=1, col=1, name="Anchors", curve=curve, curve_sample=anchors, color=anchor_indices, point_size=3)
    #     plot_curvature_with_cmap_plotly(fig=fig, row=1, col=2, name="Predicted Curvature at Anchors", curve=curve, curvature=predicted_curvature[:, 1], indices=anchor_indices, line_color='grey', line_width=2, point_size=10, color_scale='hsv')
    #
    #     # https://stackoverflow.com/questions/65941253/plotly-how-to-toggle-traces-with-a-button-similar-to-clicking-them-in-legend
    #     update_menus = [{} for _ in range(buttons_count)]
    #     button_labels = ['Toggle Samples', 'Toggle Anchors']
    #     for j in range(buttons_count):
    #         button = dict(method='restyle',
    #                        label=button_labels[j],
    #                        visible=True,
    #                        args=[{'visible': True}, [j]],
    #                        args2=[{'visible': False}, [j]])
    #
    #         update_menus[j]['buttons'] = [button]
    #         update_menus[j]['showactive'] = False
    #         update_menus[j]['y'] = buttonY
    #         update_menus[j]['x'] = buttonX + j * button_offset
    #         update_menus[j]['type'] = 'buttons'
    #
    #     fig.update_layout(
    #         showlegend=True,
    #         updatemenus=update_menus)
    #
    #     fig.update_yaxes(
    #         scaleanchor="x",
    #         scaleratio=1,
    #         row=1,
    #         col=1)
    #
    #     fig.update_layout(
    #         legend=dict(
    #             orientation="v",
    #             yanchor="bottom",
    #             xanchor="right"))
    #
    #     fig.show()

    # ----------------------------------
    # PLOT PREDICTED CURVATURES TOGETHER
    # ----------------------------------
    # fig = make_subplots(rows=1, cols=1)

    axes[1].axis('equal')
    axes[1].set_xlabel('Index', fontsize=settings.plotly_axis_title_label_fontsize)
    axes[1].set_ylabel('Curvature', fontsize=settings.plotly_axis_title_label_fontsize)

    for i, comparison in enumerate(curve_record['comparisons']):
        curvature_comparison = comparison['curvature_comparison']
        predicted_curvature = curvature_comparison['predicted_curvature']

        plot_curvature(ax=axes[1], curvature=predicted_curvature[:, 1], color=curve_colors[i])

        # plot_curvature_plotly(fig=fig, row=1, col=1, name=f'Predicted Curvature at Anchors {i+1}', curvature=predicted_curvature[:, 1], line_width=line_width, line_color=curve_colors[i])

    fig.show()


def plot_curve_arclength_records(curve_records, true_arclength_colors, predicted_arclength_colors, curve_colors, curve_color='orange', anchor_color='blue', first_anchor_color='cyan', second_anchor_color='magenta'):
    for i, curve_record in enumerate(curve_records):
        display(HTML(f'<H1>Curve {i + 1} - Arc-Length Comparison</H1>'))
        plot_curve_arclength_record(
            curve_index=i,
            curve_arclength_record=curve_record,
            true_arclength_colors=true_arclength_colors,
            predicted_arclength_colors=predicted_arclength_colors,
            curve_colors=curve_colors,
            curve_color=curve_color,
            anchor_color=anchor_color,
            first_anchor_color=first_anchor_color,
            second_anchor_color=second_anchor_color)


def plot_curve_arclength_record(curve_index, curve_arclength_record, true_arclength_colors, predicted_arclength_colors, curve_colors, curve_color, anchor_color, first_anchor_color, second_anchor_color):
    factor = 1.3
    def get_range():
        min_val1 = numpy.abs(numpy.min(curve_arclength_record['comparisons'][0]['curve'][:, 1]))
        min_val2 = numpy.abs(numpy.min(curve_arclength_record['comparisons'][1]['curve'][:, 1]))
        min_val3 = numpy.abs(numpy.min(curve_arclength_record['curve'][:, 1]))

        max_val1 = numpy.abs(numpy.max(curve_arclength_record['comparisons'][0]['curve'][:, 1]))
        max_val2 = numpy.abs(numpy.max(curve_arclength_record['comparisons'][1]['curve'][:, 1]))
        max_val3 = numpy.abs(numpy.max(curve_arclength_record['curve'][:, 1]))

        val1 = numpy.max(numpy.array([min_val1, min_val2, min_val3]))
        val2 = numpy.max(numpy.array([max_val1, max_val2, max_val3]))

        val = numpy.maximum(val1, val2) * factor
        return [-val, val]

    dir_name = "./arclength_comparison"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    # fig, axes = plt.subplots(1, 2, figsize=settings.matplotlib_figsize)
    # fig.patch.set_facecolor('white')
    # for axis in axes:
    #     # axis.axis('equal')
    #     for label in (axis.get_xticklabels() + axis.get_yticklabels()):
    #         label.set_fontsize(settings.matplotlib_axis_tick_label_fontsize)
    #
    # curve_range = get_range()

    # for i, curve_comparison in enumerate(curve_arclength_record['comparisons']):
    #     curve_arclength = curve_comparison['arclength_comparison']
    #     curve_sections = curve_arclength['curve_sections']
    #     for j, sampled_section in enumerate(curve_sections['sampled_sections']):
    #         sample = sampled_section['samples'][0]
    #         plot_sample(ax=axes[i], sample=sample, point_size=settings.matplotlib_sample_point_size, color=curve_colors[i], zorder=150)
    #         plot_sample(ax=axes[i], sample=numpy.array([[sample[0,0] ,sample[0, 1]], [sample[-1,0] ,sample[-1, 1]]]), point_size=settings.matplotlib_sample_anchor_size, alpha=1, color=curve_colors[-1], zorder=200)
    #         if j == 0:
    #             plot_sample(ax=axes[i], sample=numpy.array([[sample[0, 0] ,sample[0, 1]]]), point_size=settings.matplotlib_sample_anchor_size, alpha=1, color=first_anchor_color, zorder=300)
    #             plot_sample(ax=axes[i], sample=numpy.array([[sample[-1, 0] ,sample[-1, 1]]]), point_size=settings.matplotlib_sample_anchor_size, alpha=1, color=second_anchor_color, zorder=300)
    #
    #     axes[i].set_title(f'Sampled Curve Sections #{i+1}', fontsize=settings.matplotlib_axis_title_label_fontsize)
    #     axes[i].set_xlabel('X Coordinate', fontsize=settings.matplotlib_axis_title_label_fontsize)
    #     axes[i].set_ylabel('Y Coordinate', fontsize=settings.matplotlib_axis_title_label_fontsize)
    #     axes[i].set_xlim((curve_range[0], curve_range[1]))
    #     axes[i].set_ylim((curve_range[0], curve_range[1]))

    # fig.savefig(os.path.join(dir_name, f'sectioned_curve_{curve_index}.svg'))

    fig, axis = plt.subplots(1, 1, figsize=settings.matplotlib_figsize)
    fig.patch.set_facecolor('white')
    for label in (axis.get_xticklabels() + axis.get_yticklabels()):
        label.set_fontsize(settings.matplotlib_axis_tick_label_fontsize)

    axis.set_xlabel('Anchor Point Index', fontsize=settings.matplotlib_axis_title_label_fontsize)
    axis.set_ylabel('Predicted Arc-Length', fontsize=settings.matplotlib_axis_title_label_fontsize)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    true_arclength_legend_labels = []
    predicted_arclength_legend_labels = []
    for i, curve_comparison in enumerate(curve_arclength_record['comparisons']):
        curve_arclength = curve_comparison['arclength_comparison']
        true_arclength = curve_arclength['true_arclength']
        # true_arclength2 = curve_arclength['true_arclength2']
        predicted_arclength = curve_arclength['predicted_arclength']
        # predicted_arclength2 = curve_arclength['predicted_arclength2']

        plot_sample(ax=axis, sample=true_arclength, point_size=settings.matplotlib_line_point_size, color=true_arclength_colors[i], zorder=250)
        plot_curve(ax=axis, curve=true_arclength, linewidth=settings.matplotlib_graph_line_width, color=true_arclength_colors[i], zorder=150)
        true_arclength_legend_labels.append(f'True Arclength (Transformed Curve #{i + 1})')

        # plot_sample(ax=axis, sample=true_arclength2, point_size=settings.matplotlib_line_point_size, color=true_arclength_colors[i], zorder=250)
        # plot_curve(ax=axis, curve=true_arclength2, linewidth=settings.matplotlib_graph_line_width, color=true_arclength_colors[i], zorder=150)
        # true_arclength_legend_labels.append(f'True Arclength (Transformed Curve #{i + 1})')

        plot_sample(ax=axis, sample=predicted_arclength, point_size=settings.matplotlib_line_point_size, color=predicted_arclength_colors[i], zorder=250)
        plot_curve(ax=axis, curve=predicted_arclength, linewidth=settings.matplotlib_graph_line_width, color=predicted_arclength_colors[i], zorder=150)
        predicted_arclength_legend_labels.append(f'Predicted Arclength (Transformed Curve #{i + 1})')

        # plot_sample(ax=axis, sample=predicted_arclength2, point_size=settings.matplotlib_line_point_size, color=predicted_arclength_colors[i], zorder=250)
        # plot_curve(ax=axis, curve=predicted_arclength2, linewidth=settings.matplotlib_graph_line_width, color=predicted_arclength_colors[i], zorder=150)
        # predicted_arclength_legend_labels.append(f'Predicted Arclength (Transformed Curve #{i + 1})')

        true_arclength_legend_lines = [matplotlib.lines.Line2D([0], [0], color=color, linewidth=settings.matplotlib_graph_line_width) for color in true_arclength_colors]
        predicted_arclength_legend_lines = [matplotlib.lines.Line2D([0], [0], color=color, linewidth=settings.matplotlib_graph_line_width) for color in predicted_arclength_colors]
        legend_labels = true_arclength_legend_labels + predicted_arclength_legend_labels
        legend_lines = true_arclength_legend_lines + predicted_arclength_legend_lines
        axis.legend(legend_lines, legend_labels, prop={'size': settings.matplotlib_legend_label_fontsize})

    axis.set_title(f'Predicted Arc-Length vs. Ground Truth Arc-Length at Anchors', fontsize=settings.matplotlib_axis_title_label_fontsize)
    fig.savefig(os.path.join(dir_name, f'arclength_{curve_index}.svg'))
















    # fig, axis = plt.subplots(1, 1, figsize=settings.matplotlib_figsize)
    # fig.patch.set_facecolor('white')
    # for label in (axis.get_xticklabels() + axis.get_yticklabels()):
    #     label.set_fontsize(settings.matplotlib_axis_tick_label_fontsize)
    #
    # axis.set_xlabel('Anchor Point Index', fontsize=settings.matplotlib_axis_title_label_fontsize)
    # axis.set_ylabel('Predicted Arc-Length', fontsize=settings.matplotlib_axis_title_label_fontsize)
    # axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # true_arclength_legend_labels = []
    # predicted_arclength_legend_labels = []
    # for i, curve_comparison in enumerate(curve_arclength_record['comparisons']):
    #     curve_arclength = curve_comparison['arclength_comparison']
    #     true_arclength = curve_arclength['true_arclength']
    #     true_arclength2 = curve_arclength['true_arclength2']
    #     predicted_arclength = curve_arclength['predicted_arclength']
    #     predicted_arclength2 = curve_arclength['predicted_arclength2']
    #
    #     # plot_sample(ax=axis, sample=true_arclength[:, :, 0], point_size=settings.matplotlib_line_point_size, color=true_arclength_colors[i], zorder=250)
    #     # plot_curve(ax=axis, curve=true_arclength[:, :, 0], linewidth=settings.matplotlib_graph_line_width, color=true_arclength_colors[i], zorder=150)
    #     # true_arclength_legend_labels.append(f'True Arclength (Transformed Curve #{i + 1})')
    #
    #     plot_sample(ax=axis, sample=true_arclength2, point_size=settings.matplotlib_line_point_size, color=true_arclength_colors[i], zorder=250)
    #     plot_curve(ax=axis, curve=true_arclength2, linewidth=settings.matplotlib_graph_line_width, color=true_arclength_colors[i], zorder=150)
    #     true_arclength_legend_labels.append(f'True Arclength (Transformed Curve #{i + 1})')
    #
    #     # plot_sample(ax=axis, sample=predicted_arclength[:, :, 0], point_size=settings.matplotlib_line_point_size, color=predicted_arclength_colors[i], zorder=250)
    #     # plot_curve(ax=axis, curve=predicted_arclength[:, :, 0], linewidth=settings.matplotlib_graph_line_width, color=predicted_arclength_colors[i], zorder=150)
    #     # predicted_arclength_legend_labels.append(f'Predicted Arclength (Transformed Curve #{i + 1})')
    #
    #     plot_sample(ax=axis, sample=predicted_arclength2, point_size=settings.matplotlib_line_point_size, color=predicted_arclength_colors[i], zorder=250)
    #     plot_curve(ax=axis, curve=predicted_arclength2, linewidth=settings.matplotlib_graph_line_width, color=predicted_arclength_colors[i], zorder=150)
    #     predicted_arclength_legend_labels.append(f'Predicted Arclength (Transformed Curve #{i + 1})')
    #
    #     true_arclength_legend_lines = [matplotlib.lines.Line2D([0], [0], color=color, linewidth=settings.matplotlib_graph_line_width) for color in true_arclength_colors]
    #     predicted_arclength_legend_lines = [matplotlib.lines.Line2D([0], [0], color=color, linewidth=settings.matplotlib_graph_line_width) for color in predicted_arclength_colors]
    #     legend_labels = true_arclength_legend_labels + predicted_arclength_legend_labels
    #     legend_lines = true_arclength_legend_lines + predicted_arclength_legend_lines
    #     axis.legend(legend_lines, legend_labels, prop={'size': settings.matplotlib_legend_label_fontsize})
    #
    # axis.set_title(f'Predicted Arc-Length2 vs. Ground Truth Arc-Length2 at Anchors', fontsize=settings.matplotlib_axis_title_label_fontsize)
    # fig.savefig(os.path.join(dir_name, f'arclength2_{curve_index}.svg'))
    #
    #
    #







    # for i, curve_comparison in enumerate(curve_arclength_record['comparisons']):
    #     curve_arclength = curve_comparison['arclength_comparison']
    #     true_arclength = curve_arclength['true_arclength']
    #     predicted_arclength = curve_arclength['predicted_arclength']
    #     predicted_arclength_original = curve_arclength['predicted_arclength_original']
    #
    #     d = {
    #         'True [i, i+1]': true_arclength[1:, 1, 1],
    #         'True [i+1, i+2]': true_arclength[1:, 1, 2],
    #         'True [i, i+2]': true_arclength[1:, 1, 3],
    #         'True [i, i+1] + True [i+1, i+2]': true_arclength[1:, 1, 1] + true_arclength[1:, 1, 2],
    #         'Pred [i, i+1]': predicted_arclength[1:, 1, 1],
    #         'Pred [i+1, i+2]': predicted_arclength[1:, 1, 2],
    #         'Pred [i, i+2]': predicted_arclength[1:, 1, 3],
    #         'Pred [i, i+1] + Pred [i+1, i+2]': predicted_arclength[1:, 1, 1] + predicted_arclength[1:, 1, 2],
    #         'Diff [i, i+2]': numpy.abs((true_arclength[1:, 1, 3] - predicted_arclength[1:, 1, 3]) / true_arclength[1:, 1, 3]) * 100,
    #         'PredOrg [i, i+1]': predicted_arclength_original[1:, 1, 1],
    #         'PredOrg [i+1, i+2]': predicted_arclength_original[1:, 1, 2],
    #         'PredOrg [i, i+2]': predicted_arclength_original[1:, 1, 3],
    #         'PredOrg [i, i+1] + PredOrg [i+1, i+2]': predicted_arclength_original[1:, 1, 1] + predicted_arclength_original[1:, 1, 2]
    #     }
    #
    #     df = pandas.DataFrame(data=d)
    #
    #     style = df.style.set_properties(**{'background-color': true_arclength_colors[i]}, subset=list(d.keys())[:4])
    #     style = style.set_properties(**{'background-color': predicted_arclength_colors[i]}, subset=list(d.keys())[4:8])
    #     style = style.set_properties(**{'color': 'white', 'border-color': 'black', 'border-style': 'solid', 'border-width': '1px'})

        # display(HTML(style.render()))

    # predicted_arclength1 = curve_arclength_record[0]['predicted_arclength']
    # predicted_arclength2 = curve_arclength_record[1]['predicted_arclength']
    # display(HTML((numpy.mean(predicted_arclength1[1:, 1, 3] - predicted_arclength2[1:, 1, 3])))

    # predicted_arclength1 = curve_arclength_record['comparisons'][0]['arclength_comparison']['predicted_arclength']
    # predicted_arclength2 = curve_arclength_record['comparisons'][1]['arclength_comparison']['predicted_arclength']
    #
    # d = {
    #     'Diff [i, i+2]': (((numpy.abs(predicted_arclength1[1:, 1, 3] - predicted_arclength2[1:, 1, 3]) / predicted_arclength1[1:, 1, 3]) + (numpy.abs(predicted_arclength1[1:, 1, 3] - predicted_arclength2[1:, 1, 3]) / predicted_arclength2[1:, 1, 3])) / 2) * 100
    # }
    #
    # df = pandas.DataFrame(data=d)

    # style = df.style.set_properties(**{'background-color': true_arclength_colors[i]}, subset=list(d.keys())[:4])
    # style = style.set_properties(**{'background-color': predicted_arclength_colors[i]}, subset=list(d.keys())[4:8])
    # style = style.set_properties(**{'color': 'white', 'border-color': 'black', 'border-style': 'solid', 'border-width': '1px'})

    # display(HTML(df.style.render()))


    plt.show()


def plot_curve_signature_comparisons(curve_records, curve_colors, sample_colors, curve_color='orange', anchor_color='blue', first_anchor_color='black', second_anchor_color='pink'):
    for i, curve_record in enumerate(curve_records):
        display(HTML(f'<H1>Curve {i+1} - Signature Comparison</H1>'))
        plot_curve_signature_comparision(
            curve_index=i,
            curve_record=curve_record,
            curve_colors=curve_colors,
            sample_colors=sample_colors,
            curve_color=curve_color,
            anchor_color=anchor_color,
            first_anchor_color=first_anchor_color,
            second_anchor_color=second_anchor_color)


def plot_curve_signature_comparision(curve_index, curve_record, curve_colors, sample_colors, curve_color, anchor_color, first_anchor_color, second_anchor_color):
    dir_name = "./signature_comparison"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    # fig, axes = plt.subplots(3, 1, figsize=(20,20))
    # fig.patch.set_facecolor('white')
    # for axis in axes:
    #     for label in (axis.get_xticklabels() + axis.get_yticklabels()):
    #         label.set_fontsize(10)
    #
    # axes[0].axis('equal')
    # axes[0].set_xlabel('X Coordinate', fontsize=18)
    # axes[0].set_ylabel('Y Coordinate', fontsize=18)
    #
    # for i, comparision in enumerate(curve_record['comparisons']):
    #     curve = comparision['curve']
    #     plot_curve(ax=axes[0], curve=curve, color=curve_colors[i], linewidth=3)

    fig, axis = plt.subplots(1, 1, figsize=settings.matplotlib_figsize)
    fig.patch.set_facecolor('white')
    for label in (axis.get_xticklabels() + axis.get_yticklabels()):
        label.set_fontsize(settings.matplotlib_axis_tick_label_fontsize)

    # axes[1].axis('equal')
    # for i, curve_comparison in enumerate(curve_record['comparisons']):
    #     curve_arclength = curve_comparison['arclength_comparison']
    #     curve_sections = curve_arclength['curve_sections']
    #     curve = curve_sections['curve']
    #     for j, sampled_section in enumerate(curve_sections['sampled_sections']):
    #         point_size_regular = 7
    #         point_size_anchor = 50
    #         sample = sampled_section['samples'][0]
    #         axes[1].set_xlabel('X Coordinate', fontsize=18)
    #         axes[1].set_ylabel('Y Coordinate', fontsize=18)
    #         plot_curve(ax=axes[1], curve=curve, color=curve_color, linewidth=3)
    #         plot_sample(ax=axes[1], sample=sample, point_size=point_size_regular, color=sample_colors[i], zorder=150)
    #         plot_sample(ax=axes[1], sample=numpy.array([[sample[0,0] ,sample[0, 1]], [sample[-1,0] ,sample[-1, 1]]]), point_size=point_size_anchor, alpha=1, color=anchor_color, zorder=200)
    #         if j == 0:
    #             plot_sample(ax=axes[1], sample=numpy.array([[sample[0, 0] ,sample[0, 1]]]), point_size=point_size_anchor, alpha=1, color=first_anchor_color, zorder=300)
    #             plot_sample(ax=axes[1], sample=numpy.array([[sample[-1, 0] ,sample[-1, 1]]]), point_size=point_size_anchor, alpha=1, color=second_anchor_color, zorder=300)

    axis.set_xlabel('Predicted Arc-Length at Anchors', fontsize=settings.matplotlib_axis_title_label_fontsize)
    axis.set_ylabel('Predicted Curvature at Anchors', fontsize=settings.matplotlib_axis_title_label_fontsize)
    # axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    signature_curves_legend_labels = []

    for i, comparision in enumerate(curve_record['comparisons']):
        arclength_comparison = comparision['arclength_comparison']
        curvature_comparison = comparision['curvature_comparison']
        predicted_arclength = arclength_comparison['predicted_arclength'][:, 1]
        # predicted_arclength = numpy.concatenate((numpy.array([0]), predicted_arclength))
        predicted_curvature = curvature_comparison['predicted_curvature'][:, 1]
        plot_graph(ax=axis, x=predicted_arclength, y=predicted_curvature, color=curve_colors[i], linewidth=settings.matplotlib_graph_line_width)
        plot_sample(ax=axis, sample=None, x=predicted_arclength, y=predicted_curvature, point_size=settings.matplotlib_line_point_size, color=curve_colors[i], zorder=250)
        # signature_curves_legend_labels.append(f'Predicted Signature Curve (Transformed Curve #{i + 1})')

    signature_curves_legend_lines = [matplotlib.lines.Line2D([], [], color=color, linewidth=settings.matplotlib_graph_line_width, label=f'Predicted Signature Curve (Transformed Curve #{i + 1})') for i, color in enumerate(curve_colors[:2])]
    # axis.legend(signature_curves_legend_labels, signature_curves_legend_lines, prop={'size': 20})
    # axis.legend(signature_curves_legend_labels, signature_curves_legend_lines, prop={'size': 20})
    axis.legend(handles=signature_curves_legend_lines, prop={'size': settings.matplotlib_legend_label_fontsize})
    axis.set_title(f'Predicted Signature Curve at Anchors (Transformed Curve #1 vs. Transformed Curve #2)', fontsize=settings.matplotlib_axis_title_label_fontsize)
    # print(len(signature_curves_legend_labels))
    # print(len(signature_curves_legend_lines))
    plt.savefig(os.path.join(dir_name, f'signature_{curve_index}.svg'))
    plt.show()
