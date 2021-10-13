# python peripherals
import random

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


def plot_curve_plotly(fig, row, col, curve, line_width=2, line_color='green'):
    x = curve[:, 0]
    y = curve[:, 1]

    fig.add_trace(
        trace=graph_objects.Scatter(
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


def plot_sample(ax, sample, color, zorder, point_size=10, alpha=1):
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
def calculate_arclength_by_index(curve_sections, transform_type, modifier=None):
    curve = curve_sections['curve']
    full_sections = curve_sections['full_sections']
    true_arclength = numpy.zeros([len(full_sections) + 1, 2, 4])
    for i, full_section in enumerate(full_sections):
        point_index = i + 1
        for j, (indices, sample, accumulate) in enumerate(zip(full_section['indices'], full_section['samples'], full_section['accumulate'])):
            true_arclength[point_index, 0, j] = point_index
            if transform_type == 'equiaffine':
                if modifier == 'calabi':
                    left_indices = numpy.mod(numpy.array([indices[0] - 1]), curve.shape[0])
                    right_indices = numpy.mod(numpy.array([indices[-1] + 1]), curve.shape[0])
                    segment_indices = numpy.concatenate((left_indices, indices, right_indices))
                    sample = curve[segment_indices]
                else:
                    left_indices = numpy.mod(numpy.array([indices[0] - 2, indices[0] - 1]), curve.shape[0])
                    right_indices = numpy.mod(numpy.array([indices[-1] + 1, indices[-1] + 2]), curve.shape[0])
                    segment_indices = numpy.concatenate((left_indices, indices, right_indices))
                    sample = curve[segment_indices]

            if transform_type == 'euclidean':
                true_arclength[point_index, 1, j] = curve_processing.calculate_euclidean_arclength(curve=sample)[-1]
            elif transform_type == 'equiaffine':
                if modifier == 'calabi':
                    true_arclength[point_index, 1, j] = curve_processing.calculate_equiaffine_arclength(curve=sample)[-1]
                else:
                    true_arclength[point_index, 1, j] = curve_processing.calculate_equiaffine_arclength_by_euclidean_metrics(curve=sample)[-1]

            if accumulate is True:
                true_arclength[point_index, 1, j] = true_arclength[point_index, 1, j] + true_arclength[i, 1, j]

    return true_arclength


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


def predict_arclength_by_index(model, curve_sections):
    sampled_sections = curve_sections['sampled_sections']
    predicted_arclength = numpy.zeros([len(sampled_sections) + 1, 2, 4])
    for i, sampled_section in enumerate(sampled_sections):
        point_index = i + 1
        for j, (indices, sample, accumulate) in enumerate(zip(sampled_section['indices'], sampled_section['samples'], sampled_section['accumulate'])):
            sample = curve_processing.normalize_curve(curve=sample)
            arclength_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample).double(), dim=0), dim=0).cuda()
            with torch.no_grad():
                predicted_arclength[point_index, 0, j] = point_index
                predicted_arclength[point_index, 1, j] = torch.squeeze(model(arclength_batch_data), dim=0).cpu().detach().numpy()

            if accumulate is True:
                predicted_arclength[point_index, 1, j] = predicted_arclength[point_index, 1, j] + predicted_arclength[i, 1, j]

    return predicted_arclength


# --------------------------
# RECORD GENERATION ROUTINES
# --------------------------
def generate_curve_records(arclength_model, curvature_model, curves, transform_type, comparison_curves_count, sampling_ratio, anchors_ratio, step, neighborhood_supporting_points_count, section_supporting_points_count):
    curve_records = []
    factors = []

    for curve_index, curve in enumerate(curves):
        comparison_curves = [curve_processing.center_curve(curve=curve)]
        for i in range(comparison_curves_count):
            if transform_type == 'euclidean':
                transform = euclidean_transform.generate_random_euclidean_transform_2d()
            elif transform_type == 'equiaffine':
                transform = affine_transform.generate_random_equiaffine_transform_2d()
            elif transform_type == 'affine':
                transform = affine_transform.generate_random_affine_transform_2d(max_scale=1, min_eig_value_ratio=1, max_eig_value_ratio=1.3, min_eig_value=0.5, max_eig_value=2)
            transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
            comparison_curves.append(curve_processing.center_curve(curve=transformed_curve))

        curve_record = {
            'curve': curve,
            'comparisons': []
        }

        anchor_indices = numpy.linspace(start=0, stop=curve.shape[0], num=int(anchors_ratio * curve.shape[0]), endpoint=False, dtype=int)
        for i, comparison_curve in enumerate(comparison_curves):
            comparison_curve_points_count = comparison_curve.shape[0]
            sampling_points_count = int(sampling_ratio * comparison_curve_points_count)
            dist = discrete_distribution.random_discrete_dist(bins=comparison_curve_points_count, multimodality=60, max_density=1, count=1)[0]

            curve_neighborhoods = extract_curve_neighborhoods(
                curve=comparison_curve,
                dist=dist,
                sampling_points_count=sampling_points_count,
                supporting_points_count=neighborhood_supporting_points_count,
                anchor_indices=anchor_indices)

            curve_sections = extract_curve_sections(
                curve=comparison_curve,
                step=step,
                sample_points=section_supporting_points_count)

            curve_neighborhoods_from_sections = extract_curve_neighborhoods_from_curve_sections(
                curve=comparison_curve,
                curve_sections=curve_sections,
                supporting_points_count=neighborhood_supporting_points_count)

            true_arclength = calculate_arclength_by_index(
                curve_sections=curve_sections,
                transform_type=transform_type)

            predicted_arclength = predict_arclength_by_index(
                model=arclength_model,
                curve_sections=curve_sections)

            true_curvature = calculate_curvature_by_index(
                curve=curve,
                transform_type=transform_type)

            predicted_curvature = predict_curvature_by_index(
                model=curvature_model,
                curve_neighborhoods=curve_neighborhoods)

            predicted_curvature_signature = predict_curvature_by_index(
                model=curvature_model,
                curve_neighborhoods=curve_neighborhoods_from_sections)

            sampled_indices = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=sampling_points_count)
            sampled_curve = comparison_curve[sampled_indices]
            anchors = comparison_curve[anchor_indices]

            arclength_comparison = {
                'curve_sections': curve_sections,
                'true_arclength': true_arclength,
                'predicted_arclength': predicted_arclength,
                'predicted_arclength_original': predicted_arclength.copy()
            }

            curvature_comparison = {
                'curve_neighborhoods': curve_neighborhoods,
                'true_curvature': true_curvature,
                'predicted_curvature': predicted_curvature,
                'predicted_curvature_signature': predicted_curvature_signature
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

            factor = numpy.mean(true_arclength[1:, 1, 0] / predicted_arclength[1:, 1, 0])
            factors.append(factor)

        curve_records.append(curve_record)

    if transform_type != 'affine':
        factor = numpy.mean(numpy.array(factors))
        for curve_record in curve_records:
            for comparison in curve_record['comparisons']:
                comparison['arclength_comparison']['predicted_arclength'][:, 1, :] *= factor

    return curve_records


def extract_curve_neighborhoods(curve, dist, sampling_points_count, supporting_points_count, anchor_indices):
    sampled_neighborhoods = []
    indices_pool = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=sampling_points_count)
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
            supporting_points_count=sample_points)

        sampled_indices3 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index2,
            end_point_index=index3,
            supporting_points_count=sample_points)

        sampled_indices4 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index1,
            end_point_index=index3,
            supporting_points_count=sample_points)

        sampled_section = {
            'indices': [sampled_indices1, sampled_indices1, sampled_indices3, sampled_indices4],
            'samples': [curve[sampled_indices1], curve[sampled_indices1], curve[sampled_indices3], curve[sampled_indices4]],
            'accumulate': [True, False, False, False]
        }

        sampled_sections.append(sampled_section)

        full_indices1 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index1,
            end_point_index=index2)

        full_indices3 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index2,
            end_point_index=index3)

        full_indices4 = curve_sampling.sample_curve_section_indices_old(
            curve=curve,
            start_point_index=index1,
            end_point_index=index3)

        full_section = {
            'indices': [full_indices1, full_indices1, full_indices3, full_indices4],
            'samples': [curve[full_indices1], curve[full_indices1], curve[full_indices3], curve[full_indices4]],
            'accumulate': [True, False, False, False]
        }

        full_sections.append(full_section)

    return {
        'sampled_sections': sampled_sections,
        'full_sections': full_sections,
        'curve': curve
    }


# -------------
# PLOT ROUTINES
# -------------
def plot_curve_curvature_comparisons(curve_records, curve_colors):
    for i, curve_record in enumerate(curve_records):
        display(HTML(f'<H1>Curve {i+1} - Curvature Comparison</H1>'))
        plot_curve_curvature_comparison(
            curve_record=curve_record,
            curve_colors=curve_colors)


def plot_curve_curvature_comparison(curve_record, curve_colors):
    axis_index = 0
    fontsize = 25
    axes_count = 15
    line_width = 2

    # ---------------------
    # PLOT CURVES TOGETHER
    # ---------------------
    fig = make_subplots(rows=1, cols=1)

    for i, comparison in enumerate(curve_record['comparisons']):
        curve = comparison['curve']
        plot_curve_plotly(fig=fig, row=1, col=1, curve=curve, line_width=line_width, line_color=curve_colors[i])

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.show()

    # -------------------------------
    # PLOT CURVE SAMPLES SIDE BY SIDE
    # -------------------------------
    fig = make_subplots(rows=1, cols=len(curve_record['comparisons']))

    for i, comparison in enumerate(curve_record['comparisons']):
        sampled_curve = comparison['sampled_curve']
        curve = comparison['curve']

        plot_curve_sample_plotly(fig=fig, row=1, col=i+1, name=f'Sampled Curve {i+1}', curve=curve, curve_sample=sampled_curve, color=curve_colors[i], point_size=3)

        fig.update_yaxes(
            scaleanchor=f'x{i+1}',
            scaleratio=1,
            row=1,
            col=i+1)

    fig.show()

    # ----------------------------------------------------------------
    # PLOT CURVE SAMPLES, ANCHORS AND PREDICTED CURVATURE SIDE BY SIDE
    # ----------------------------------------------------------------

    button_offset = 0.1
    buttonX = 0.1
    buttonY = 1.3
    buttons_count = 2
    left_width = 0.25
    for i, comparison in enumerate(curve_record['comparisons']):
        fig = make_subplots(rows=1, cols=2, column_widths=[left_width, 1 - left_width])
        sampled_curve = comparison['sampled_curve']
        anchors = comparison['anchors']
        anchor_indices = comparison['anchor_indices']
        curve = comparison['curve']
        curvature_comparison = comparison['curvature_comparison']
        predicted_curvature = curvature_comparison['predicted_curvature']

        plot_curve_sample_plotly(fig=fig, row=1, col=1, name="Sampled Curve", curve=curve, curve_sample=sampled_curve, color='grey')
        plot_curve_sample_plotly(fig=fig, row=1, col=1, name="Anchors", curve=curve, curve_sample=anchors, color=anchor_indices, point_size=3)
        plot_curvature_with_cmap_plotly(fig=fig, row=1, col=2, name="Predicted Curvature at Anchors", curve=curve, curvature=predicted_curvature[:, 1], indices=anchor_indices, line_color='grey', line_width=2, point_size=10, color_scale='hsv')

        # https://stackoverflow.com/questions/65941253/plotly-how-to-toggle-traces-with-a-button-similar-to-clicking-them-in-legend
        update_menus = [{} for _ in range(buttons_count)]
        button_labels = ['Toggle Samples', 'Toggle Anchors']
        for j in range(buttons_count):
            button = dict(method='restyle',
                           label=button_labels[j],
                           visible=True,
                           args=[{'visible': True}, [j]],
                           args2=[{'visible': False}, [j]])

            update_menus[j]['buttons'] = [button]
            update_menus[j]['showactive'] = False
            update_menus[j]['y'] = buttonY
            update_menus[j]['x'] = buttonX + j * button_offset
            update_menus[j]['type'] = 'buttons'

        fig.update_layout(
            showlegend=True,
            updatemenus=update_menus)

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=1)

        fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="bottom",
                xanchor="right"))

        fig.show()

    # ----------------------------------
    # PLOT PREDICTED CURVATURES TOGETHER
    # ----------------------------------
    fig = make_subplots(rows=1, cols=1)

    for i, comparison in enumerate(curve_record['comparisons']):
        curvature_comparison = comparison['curvature_comparison']
        predicted_curvature = curvature_comparison['predicted_curvature']

        plot_curvature_plotly(fig=fig, row=1, col=1, name=f'Predicted Curvature at Anchors {i+1}', curvature=predicted_curvature[:, 1], line_width=line_width, line_color=curve_colors[i])

    fig.show()


def plot_curve_arclength_records(curve_records, true_arclength_colors, predicted_arclength_colors, sample_colors, curve_color='orange', anchor_color='blue', first_anchor_color='black'):
    for curve_record in curve_records:
        plot_curve_arclength_record(
            curve_arclength_record=curve_record,
            true_arclength_colors=true_arclength_colors,
            predicted_arclength_colors=predicted_arclength_colors,
            sample_colors=sample_colors,
            curve_color=curve_color,
            anchor_color=anchor_color,
            first_anchor_color=first_anchor_color)


def plot_curve_arclength_record(curve_arclength_record, true_arclength_colors, predicted_arclength_colors, sample_colors, curve_color, anchor_color, first_anchor_color):
    fig, axes = plt.subplots(3, 1, figsize=(20,20))
    fig.patch.set_facecolor('white')
    for axis in axes:
        for label in (axis.get_xticklabels() + axis.get_yticklabels()):
            label.set_fontsize(10)

    axes[0].axis('equal')
    for i, curve_comparison in enumerate(curve_arclength_record['comparisons']):
        curve_arclength = curve_comparison['arclength_comparison']
        curve_sections = curve_arclength['curve_sections']
        curve = curve_sections['curve']
        for j, sampled_section in enumerate(curve_sections['sampled_sections']):
            sample = sampled_section['samples'][0]
            axes[0].set_xlabel('X Coordinate', fontsize=18)
            axes[0].set_ylabel('Y Coordinate', fontsize=18)
            plot_curve(ax=axes[0], curve=curve, color=curve_color, linewidth=3)
            plot_sample(ax=axes[0], sample=sample, point_size=10, color=sample_colors[i], zorder=150)
            plot_sample(ax=axes[0], sample=numpy.array([[sample[0,0] ,sample[0, 1]], [sample[-1,0] ,sample[-1, 1]]]), point_size=70, alpha=1, color=anchor_color, zorder=200)
            if j == 0:
                plot_sample(ax=axes[0], sample=numpy.array([[sample[0,0] ,sample[0, 1]]]), point_size=70, alpha=1, color=first_anchor_color, zorder=300)

    axes[1].set_xlabel('Index', fontsize=18)
    axes[1].set_ylabel('Arc-Length', fontsize=18)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    true_arclength_legend_labels = []
    predicted_arclength_legend_labels = []
    for i, curve_comparison in enumerate(curve_arclength_record['comparisons']):
        curve_arclength = curve_comparison['arclength_comparison']
        true_arclength = curve_arclength['true_arclength']
        predicted_arclength = curve_arclength['predicted_arclength']

        plot_sample(ax=axes[1], sample=true_arclength[:, :, 0], point_size=40, color=true_arclength_colors[i], zorder=250)
        plot_curve(ax=axes[1], curve=true_arclength[:, :, 0], linewidth=2, color=true_arclength_colors[i], zorder=150)
        true_arclength_legend_labels.append(f'True Arclength (Curve #{i + 1})')

        plot_sample(ax=axes[1], sample=predicted_arclength[:, :, 0], point_size=40, color=predicted_arclength_colors[i], zorder=250)
        plot_curve(ax=axes[1], curve=predicted_arclength[:, :, 0], linewidth=2, color=predicted_arclength_colors[i], zorder=150)
        predicted_arclength_legend_labels.append(f'Predicted Arclength (Curve #{i + 1})')

        true_arclength_legend_lines = [matplotlib.lines.Line2D([0], [0], color=color, linewidth=3) for color in true_arclength_colors]
        predicted_arclength_legend_lines = [matplotlib.lines.Line2D([0], [0], color=color, linewidth=3) for color in predicted_arclength_colors]
        legend_labels = true_arclength_legend_labels + predicted_arclength_legend_labels
        legend_lines = true_arclength_legend_lines + predicted_arclength_legend_lines
        axes[1].legend(legend_lines, legend_labels, prop={'size': 20})

    for i, curve_comparison in enumerate(curve_arclength_record['comparisons']):
        curve_arclength = curve_comparison['arclength_comparison']
        true_arclength = curve_arclength['true_arclength']
        predicted_arclength = curve_arclength['predicted_arclength']
        predicted_arclength_original = curve_arclength['predicted_arclength_original']

        d = {
            'True [i, i+1]': true_arclength[1:, 1, 1],
            'True [i+1, i+2]': true_arclength[1:, 1, 2],
            'True [i, i+2]': true_arclength[1:, 1, 3],
            'True [i, i+1] + True [i+1, i+2]': true_arclength[1:, 1, 1] + true_arclength[1:, 1, 2],
            'Pred [i, i+1]': predicted_arclength[1:, 1, 1],
            'Pred [i+1, i+2]': predicted_arclength[1:, 1, 2],
            'Pred [i, i+2]': predicted_arclength[1:, 1, 3],
            'Pred [i, i+1] + Pred [i+1, i+2]': predicted_arclength[1:, 1, 1] + predicted_arclength[1:, 1, 2],
            'Diff [i, i+2]': numpy.abs((true_arclength[1:, 1, 3] - predicted_arclength[1:, 1, 3]) / true_arclength[1:, 1, 3]) * 100,
            'PredOrg [i, i+1]': predicted_arclength_original[1:, 1, 1],
            'PredOrg [i+1, i+2]': predicted_arclength_original[1:, 1, 2],
            'PredOrg [i, i+2]': predicted_arclength_original[1:, 1, 3],
            'PredOrg [i, i+1] + PredOrg [i+1, i+2]': predicted_arclength_original[1:, 1, 1] + predicted_arclength_original[1:, 1, 2]
        }

        df = pandas.DataFrame(data=d)

        style = df.style.set_properties(**{'background-color': true_arclength_colors[i]}, subset=list(d.keys())[:4])
        style = style.set_properties(**{'background-color': predicted_arclength_colors[i]}, subset=list(d.keys())[4:8])
        style = style.set_properties(**{'color': 'white', 'border-color': 'black', 'border-style': 'solid', 'border-width': '1px'})

        display(HTML(style.render()))

    # predicted_arclength1 = curve_arclength_record[0]['predicted_arclength']
    # predicted_arclength2 = curve_arclength_record[1]['predicted_arclength']
    # display(HTML((numpy.mean(predicted_arclength1[1:, 1, 3] - predicted_arclength2[1:, 1, 3])))

    predicted_arclength1 = curve_arclength_record['comparisons'][0]['arclength_comparison']['predicted_arclength']
    predicted_arclength2 = curve_arclength_record['comparisons'][1]['arclength_comparison']['predicted_arclength']

    d = {
        'Diff [i, i+2]': (((numpy.abs(predicted_arclength1[1:, 1, 3] - predicted_arclength2[1:, 1, 3]) / predicted_arclength1[1:, 1, 3]) + (numpy.abs(predicted_arclength1[1:, 1, 3] - predicted_arclength2[1:, 1, 3]) / predicted_arclength2[1:, 1, 3])) / 2) * 100
    }

    df = pandas.DataFrame(data=d)

    # style = df.style.set_properties(**{'background-color': true_arclength_colors[i]}, subset=list(d.keys())[:4])
    # style = style.set_properties(**{'background-color': predicted_arclength_colors[i]}, subset=list(d.keys())[4:8])
    # style = style.set_properties(**{'color': 'white', 'border-color': 'black', 'border-style': 'solid', 'border-width': '1px'})

    display(HTML(df.style.render()))

    plt.show()


def plot_curve_signature_comparisons(curve_records, curve_colors):
    for i, curve_record in enumerate(curve_records):
        display(HTML(f'<H1>Curve {i+1} - Signature Comparison</H1>'))
        plot_curve_signature_comparision(
            curve_record=curve_record,
            curve_colors=curve_colors)


def plot_curve_signature_comparision(curve_record, curve_colors):
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
        plot_curve(ax=axes[0], curve=curve, color=curve_colors[i], linewidth=3)

    axes[1].set_xlabel('Arc-Length', fontsize=18)
    axes[1].set_ylabel('Curvature', fontsize=18)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for i, comparision in enumerate(curve_record['comparisons']):
        arclength_comparison = comparision['arclength_comparison']
        curvature_comparison = comparision['curvature_comparison']
        predicted_arclength = arclength_comparison['predicted_arclength'][:, 1, 0]
        # predicted_arclength = numpy.concatenate((numpy.array([0]), predicted_arclength))
        predicted_curvature = curvature_comparison['predicted_curvature_signature'][:, 1]
        plot_graph(ax=axes[1], x=predicted_arclength, y=predicted_curvature, color=curve_colors[i], linewidth=3)

    plt.show()