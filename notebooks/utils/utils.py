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


def plot_graph_plotly(fig, row, col, x, y, name, point_size=2, line_width=2, line_color='green', mode='lines+markers'):
    fig.add_trace(
        trace=graph_objects.Scatter(
            name=name,
            x=x,
            y=y,
            mode=mode,
            marker={
                'size': point_size
            },
            line={
                'color': line_color,
                'width': line_width
            }),
        row=row,
        col=col)


def plot_curve_plotly(fig, row, col, curve, name, point_size=2, line_width=2, line_color='green', mode='lines+markers'):
    x = curve[:, 0]
    y = curve[:, 1]
    plot_graph_plotly(fig=fig, row=row, col=col, x=x, y=y, name=name, point_size=point_size, line_width=line_width, line_color=line_color, mode=mode)



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


def plot_graph(ax, x, y, linewidth=2, color='red', alpha=1, zorder=1, label=None):
    return ax.plot(x, y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder, label=label)


def plot_curve(ax, curve, linewidth=2, color='red', alpha=1, zorder=1, label=None):
    x = curve[:, 0]
    y = curve[:, 1]
    return plot_graph(ax=ax, x=x, y=y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder, label=label)


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
def predict_curvature_by_index(model, curve_neighborhoods, factor=-1):
    sampled_neighborhoods = curve_neighborhoods['sampled_neighborhoods']
    predicted_curvature = numpy.zeros([len(sampled_neighborhoods), 2])
    for point_index, sampled_neighborhood in enumerate(sampled_neighborhoods):
        for (indices, sample) in zip(sampled_neighborhood['indices'], sampled_neighborhood['samples']):
            sample = curve_processing.normalize_curve(curve=sample)
            curvature_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample).double(), dim=0), dim=0).cuda()
            with torch.no_grad():
                predicted_curvature[point_index, 0] = point_index
                predicted_curvature[point_index, 1] = torch.squeeze(model(curvature_batch_data), dim=0).cpu().detach().numpy() * factor
    return predicted_curvature


def predict_arclength_by_index(model, curve, indices_pool, supporting_points_count, anchor_indices=None):
    anchor_indices = anchor_indices if anchor_indices is not None else indices_pool
    predicted_arclength = numpy.zeros(anchor_indices.shape[0])
    step = supporting_points_count - 1
    arclength_at_index = {}
    arclength_at_index[anchor_indices[0]] = 0
    for i, anchor_index in enumerate(anchor_indices[1:]):
        modified_indices_pool = utils.insert_sorted(indices_pool, numpy.array([anchor_indices[0], anchor_index]))
        sampled_curve = curve[modified_indices_pool]
        anchor_meta_index = int(numpy.where(modified_indices_pool == anchor_index)[0])
        max_index = max(arclength_at_index, key=arclength_at_index.get)
        max_meta_index = int(numpy.where(modified_indices_pool == max_index)[0])
        anchor_arclength = arclength_at_index[max_index]
        for meta_index in range(max_meta_index, anchor_meta_index):
            start_meta_index = meta_index - step
            end_meta_index = meta_index
            end_meta_index2 = end_meta_index + 1

            sampled_indices1 = curve_sampling.sample_curve_section_indices_old(
                curve=sampled_curve,
                start_point_index=start_meta_index,
                end_point_index=end_meta_index,
                supporting_points_count=supporting_points_count,
                uniform=True)

            sampled_indices2 = curve_sampling.sample_curve_section_indices_old(
                curve=sampled_curve,
                start_point_index=start_meta_index,
                end_point_index=end_meta_index2,
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

            current_index = modified_indices_pool[end_meta_index2]
            if current_index != anchor_index:
                arclength_at_index[current_index] = anchor_arclength

        predicted_arclength[i+1] = anchor_arclength

    indices = numpy.array(list(range(predicted_arclength.shape[0])))
    values = predicted_arclength
    return numpy.vstack((indices, values)).transpose()


# --------------------------
# RECORD GENERATION ROUTINES
# --------------------------
def generate_curve_records(arclength_model, curvature_model, curves, factor_extraction_curves, transform_type, comparison_curves_count, sampling_ratio, anchors_ratio, step, neighborhood_supporting_points_count, section_supporting_points_count):
    curve_records = []
    for curve_index, curve in enumerate(curves):
        curve = curve_processing.enforce_cw(curve=curve)

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
            modified_indices_pool = utils.insert_sorted(indices_pool, numpy.array([0]))

            true_arclength = calculate_arclength_by_index(
                curve=comparison_curve,
                anchor_indices=anchor_indices,
                transform_type=transform_type)

            predicted_arclength = predict_arclength_by_index(
                model=arclength_model,
                curve=comparison_curve,
                indices_pool=indices_pool,
                supporting_points_count=section_supporting_points_count,
                anchor_indices=anchor_indices)

            predicted_arclength_without_anchors = predict_arclength_by_index(
                model=arclength_model,
                curve=comparison_curve,
                indices_pool=indices_pool,
                supporting_points_count=section_supporting_points_count)

            curve_neighborhoods = extract_curve_neighborhoods(
                curve=comparison_curve,
                indices_pool=indices_pool,
                supporting_points_count=neighborhood_supporting_points_count,
                anchor_indices=anchor_indices)

            curve_neighborhoods_without_anchors = extract_curve_neighborhoods(
                curve=comparison_curve,
                indices_pool=modified_indices_pool,
                supporting_points_count=neighborhood_supporting_points_count)

            predicted_curvature = predict_curvature_by_index(
                model=curvature_model,
                curve_neighborhoods=curve_neighborhoods)

            predicted_curvature_without_anchors = predict_curvature_by_index(
                model=curvature_model,
                curve_neighborhoods=curve_neighborhoods_without_anchors)

            true_curvature = calculate_curvature_by_index(
                curve=curve,
                transform_type=transform_type)

            sampled_indices = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=sampling_points_count)
            sampled_curve = comparison_curve[sampled_indices]
            anchors = comparison_curve[anchor_indices]

            arclength_comparison = {
                'true_arclength': true_arclength,
                'predicted_arclength': predicted_arclength,
                'predicted_arclength_without_anchors': predicted_arclength_without_anchors
            }

            curvature_comparison = {
                'curve_neighborhoods': curve_neighborhoods,
                'true_curvature': true_curvature,
                'predicted_curvature': predicted_curvature,
                'predicted_curvature_without_anchors': predicted_curvature_without_anchors
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

    factors = []
    for curve_index, curve in enumerate(factor_extraction_curves):
        all_indices = numpy.array(list(range(curve.shape[0])))
        true_arclength = calculate_arclength_by_index(
            curve=curve,
            anchor_indices=all_indices,
            transform_type=transform_type)

        predicted_arclength = predict_arclength_by_index(
            model=arclength_model,
            curve=curve,
            indices_pool=all_indices,
            anchor_indices=all_indices,
            supporting_points_count=section_supporting_points_count)

        factor = numpy.mean(true_arclength[1:, 1] / predicted_arclength[1:, 1])
        factors.append(factor)

    if transform_type != 'affine':
        factor = numpy.mean(numpy.array(factors))
        for curve_record in curve_records:
            for comparison in curve_record['comparisons']:
                comparison['arclength_comparison']['predicted_arclength'][:, 1] *= factor

    return curve_records


def extract_curve_neighborhoods(curve, indices_pool, supporting_points_count, anchor_indices=None):
    sampled_neighborhoods = []
    anchor_indices = anchor_indices if anchor_indices is not None else indices_pool
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
        predicted_arclength_ref = arclength_comparison_ref['predicted_arclength'][1:, 1].squeeze()
        predicted_curvature_ref = curvature_comparison_ref['predicted_curvature'][:, 1].squeeze()
        for comparison in comparisons[1:]:
            arclength_comparison = comparison['arclength_comparison']
            curvature_comparison = comparison['curvature_comparison']
            predicted_arclength = arclength_comparison['predicted_arclength'][1:, 1].squeeze()
            predicted_curvature = curvature_comparison['predicted_curvature'][:, 1].squeeze()
            arclength_offset = numpy.abs(predicted_arclength - predicted_arclength_ref) / numpy.abs(predicted_arclength_ref)
            curvature_offset = numpy.abs(predicted_curvature - predicted_curvature_ref) / numpy.abs(predicted_curvature_ref)
            arclength_offsets = numpy.concatenate((arclength_offsets, arclength_offset))
            curvature_offsets = numpy.concatenate((curvature_offsets, curvature_offset))

    curvature_offsets.sort()
    print(curvature_offsets)

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

    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))

    fig.update_layout(yaxis1=dict(range=get_range()))
    fig.update_layout(yaxis2=dict(range=get_range()))
    fig.update_layout(yaxis3=dict(range=get_range()))

    fig['layout']['xaxis']['title'] = 'X Coordinate'
    fig['layout']['yaxis']['title'] = 'Y Coordinate'

    fig['layout']['xaxis2']['title'] = 'X Coordinate'
    fig['layout']['yaxis2']['title'] = 'Y Coordinate'

    fig['layout']['xaxis3']['title'] = 'X Coordinate'
    fig['layout']['yaxis3']['title'] = 'Y Coordinate'

    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig.update_layout(showlegend=False)

    fig.write_image(os.path.join(dir_name, f'curves_together_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    fig.show()

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

    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig['layout']['xaxis']['title'] = 'X Coordinate'
    fig['layout']['yaxis']['title'] = 'Y Coordinate'

    fig['layout']['xaxis2']['title'] = 'X Coordinate'
    fig['layout']['yaxis2']['title'] = 'Y Coordinate'

    fig.update_layout(yaxis1=dict(range=get_range()))
    fig.update_layout(yaxis2=dict(range=get_range()))

    fig.update_layout(showlegend=False)

    fig.write_image(os.path.join(dir_name, f'curve_samples_side_by_side_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    fig.show()

    # ----------------------------------------------------------------------------------
    # PLOT CURVE SAMPLES, ANCHORS AND PREDICTED CURVATURE SIDE BY SIDE (WITHOUT BUTTONS)
    # ----------------------------------------------------------------------------------
    left_width = 0.25
    for i, comparison in enumerate(curve_record['comparisons']):
        fig = make_subplots(rows=1, cols=3, column_widths=[left_width, left_width, 1 - (2*left_width)], subplot_titles=('<b>Sampled Curve</b>', '<b>Anchors</b>',  '<b>Predicted Curvature at Anchors</b>'))
        sampled_curve = comparison['sampled_curve']
        anchors = comparison['anchors']
        anchor_indices = comparison['anchor_indices']
        curve = comparison['curve']
        curvature_comparison = comparison['curvature_comparison']
        predicted_curvature = curvature_comparison['predicted_curvature']

        plot_curve_sample_plotly(fig=fig, row=1, col=1, name="Sampled Curve", curve=curve, curve_sample=sampled_curve, color=curve_colors[i], point_size=settings.plotly_sample_point_size)
        plot_curve_sample_plotly(fig=fig, row=1, col=2, name="Anchors", curve=curve, curve_sample=anchors, color=anchor_indices, point_size=settings.plotly_sample_point_size)
        plot_curvature_with_cmap_plotly(fig=fig, row=1, col=3, name="Predicted Curvature", curve=curve, curvature=predicted_curvature[:, 1], indices=anchor_indices, line_color='grey', line_width=settings.plotly_graph_line_width, point_size=settings.plotly_sample_anchor_size, color_scale='hsv')

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

        fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize), showlegend=False)

        fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

        fig.write_image(os.path.join(dir_name, f'curve_samples_and_predicted_curvature_{curve_index}_{i}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
        fig.show()

    fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted Curvature at Anchors (Transformed Curve #1 vs. Transformed Curve #2)</b>',))

    for i, comparison in enumerate(curve_record['comparisons']):
        curvature_comparison = comparison['curvature_comparison']
        predicted_curvature = curvature_comparison['predicted_curvature']

        plot_curve_plotly(fig=fig, row=1, col=1, name=f'Predicted Curvature at Anchors #{i+1}', curve=predicted_curvature, line_width=settings.plotly_graph_line_width, line_color=curve_colors[i], mode='lines')

    fig['layout']['xaxis']['title'] = 'Anchor Point Index'
    fig['layout']['yaxis']['title'] = 'Predicted Curvature'

    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))

    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig.write_image(os.path.join(dir_name, f'predicted_curves_together_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    fig.show()

    # ------------------------------------
    # CURVATURE VS. INDEX OF SAMPLE POINTS
    # ------------------------------------
    fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted Curvature as a Function of Point Index</b>',))
    for i, comparison in enumerate(curve_record['comparisons']):
        curvature_comparison = comparison['curvature_comparison']
        predicted_curvature = curvature_comparison['predicted_curvature_without_anchors']

        plot_curve_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', curve=predicted_curvature, point_size=settings.plotly_sample_point_size, line_width=settings.plotly_graph_line_width, line_color=curve_colors[i], mode='markers')

    fig['layout']['xaxis']['title'] = 'Sample Point Index'
    fig['layout']['yaxis']['title'] = 'Predicted Curvature'

    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))

    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig.write_image(os.path.join(dir_name, f'predicted_curvature_as_function_of_index_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    fig.show()

    # -----------------------------------------
    # CURVATURE VS. ARC-LENGTH OF SAMPLE POINTS
    # -----------------------------------------
    fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted Curvature as a Function of Predicted Arc-Length</b>',))
    for i, comparison in enumerate(curve_record['comparisons']):
        curvature_comparison = comparison['curvature_comparison']
        arclength_comparison = comparison['arclength_comparison']
        predicted_curvature = curvature_comparison['predicted_curvature_without_anchors']
        predicted_arclength = arclength_comparison['predicted_arclength_without_anchors']

        plot_graph_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', x=predicted_arclength[:, 1], y=predicted_curvature[:, 1], point_size=settings.plotly_sample_point_size, line_width=settings.plotly_graph_line_width, line_color=curve_colors[i], mode='markers')

    fig['layout']['xaxis']['title'] = 'Predicted Arc-Length'
    fig['layout']['yaxis']['title'] = 'Predicted Curvature'

    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))

    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig.write_image(os.path.join(dir_name, f'predicted_curvature_as_function_of_arclength_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
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
    dir_name = "./arclength_comparison"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(1, 1, figsize=settings.matplotlib_figsize)
    fig.patch.set_facecolor('white')
    for label in (axis.get_xticklabels() + axis.get_yticklabels()):
        label.set_fontsize(settings.matplotlib_axis_tick_label_fontsize)

    axis.set_xlabel('Point Index', fontsize=settings.matplotlib_axis_title_label_fontsize)
    axis.set_ylabel('Arc-Length', fontsize=settings.matplotlib_axis_title_label_fontsize)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for i, curve_comparison in enumerate(curve_arclength_record['comparisons']):
        curve_arclength = curve_comparison['arclength_comparison']
        true_arclength = curve_arclength['true_arclength']
        predicted_arclength = curve_arclength['predicted_arclength']

        # plot_sample(ax=axis, sample=true_arclength, point_size=settings.matplotlib_line_point_size, color=true_arclength_colors[i], zorder=250)
        plot_curve(ax=axis, curve=true_arclength, linewidth=settings.matplotlib_graph_line_width, color=true_arclength_colors[i], zorder=150, label=f'True Arc-Length (Transformed Curve #{i + 1})')

        # plot_sample(ax=axis, sample=predicted_arclength, point_size=settings.matplotlib_line_point_size, color=predicted_arclength_colors[i], zorder=250)
        plot_curve(ax=axis, curve=predicted_arclength, linewidth=settings.matplotlib_graph_line_width, color=predicted_arclength_colors[i], zorder=150, label=f'Predicted Arc-Length (Transformed Curve #{i + 1})')

        axis.legend(prop={'size': settings.matplotlib_legend_label_fontsize})

    axis.set_title(f'Predicted Arc-Length vs. Ground Truth Arc-Length (at Anchors)', fontsize=settings.matplotlib_axis_title_label_fontsize)
    fig.savefig(os.path.join(dir_name, f'arclength_{curve_index}.svg'))

    plt.show()


def plot_curve_signature_comparisons(curve_records, true_signature_colors, predicted_signature_colors, sample_colors, curve_color='orange', anchor_color='blue', first_anchor_color='black', second_anchor_color='pink'):
    for i, curve_record in enumerate(curve_records):
        display(HTML(f'<H1>Curve {i+1} - Signature Comparison</H1>'))
        plot_curve_signature_comparision(
            curve_index=i,
            curve_record=curve_record,
            true_signature_colors=true_signature_colors,
            predicted_signature_colors=predicted_signature_colors,
            sample_colors=sample_colors,
            curve_color=curve_color,
            anchor_color=anchor_color,
            first_anchor_color=first_anchor_color,
            second_anchor_color=second_anchor_color)


def plot_curve_signature_comparision(curve_index, curve_record, true_signature_colors, predicted_signature_colors, sample_colors, curve_color, anchor_color, first_anchor_color, second_anchor_color):
    dir_name = "./signature_comparison"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(1, 1, figsize=settings.matplotlib_figsize)
    fig.patch.set_facecolor('white')
    for label in (axis.get_xticklabels() + axis.get_yticklabels()):
        label.set_fontsize(settings.matplotlib_axis_tick_label_fontsize)

    axis.set_xlabel('Arc-Length', fontsize=settings.matplotlib_axis_title_label_fontsize)
    axis.set_ylabel('Curvature', fontsize=settings.matplotlib_axis_title_label_fontsize)

    for i, comparision in enumerate(curve_record['comparisons']):
        arclength_comparison = comparision['arclength_comparison']
        curvature_comparison = comparision['curvature_comparison']
        predicted_arclength = arclength_comparison['predicted_arclength'][:, 1]
        predicted_curvature = curvature_comparison['predicted_curvature'][:, 1]

        true_arclength = arclength_comparison['true_arclength'][:, 1]
        true_curvature = 150*curvature_comparison['true_curvature'][:, 1]

        plot_graph(ax=axis, x=predicted_arclength, y=predicted_curvature, color=predicted_signature_colors[i], linewidth=settings.matplotlib_graph_line_width, label=f'Predicted Signature Curve (Transformed Curve #{i + 1})')
        # plot_sample(ax=axis, sample=None, x=predicted_arclength, y=predicted_curvature, point_size=settings.matplotlib_line_point_size, color=predicted_signature_colors[i], zorder=250)

        plot_graph(ax=axis, x=true_arclength, y=true_curvature, color=true_signature_colors[i], linewidth=settings.matplotlib_graph_line_width, label=f'True Signature Curve (Transformed Curve #{i + 1})')
        # plot_sample(ax=axis, sample=None, x=true_arclength, y=true_curvature, point_size=settings.matplotlib_line_point_size, color=true_signature_colors[i], zorder=250)

    axis.legend(prop={'size': settings.matplotlib_legend_label_fontsize})
    axis.set_title(f'Predicted Signature Curve vs. Ground Truth Signature Curve (at Anchors)', fontsize=settings.matplotlib_axis_title_label_fontsize)
    plt.savefig(os.path.join(dir_name, f'signature_{curve_index}.svg'))
    plt.show()
