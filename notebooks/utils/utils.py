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

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines


# https://stackoverflow.com/questions/36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar
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
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

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
    colorline(ax=ax, x=x, y=y, cmap='hsv')


def plot_curve_sample(ax, curve, curve_sample, indices, zorder, point_size=10, alpha=1, cmap='hsv'):
    x = curve_sample[:, 0]
    y = curve_sample[:, 1]
    c = numpy.linspace(0.0, 1.0, curve.shape[0])

    ax.scatter(
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
    ax.add_artist(circle)


def plot_curve(ax, curve, linewidth=2, color='red', alpha=1, zorder=1):
    x = curve[:, 0]
    y = curve[:, 1]
    ax.plot(x, y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder)


def plot_curvature(ax, curvature, color='red', linewidth=2, alpha=1):
    x = range(curvature.shape[0])
    y = curvature
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)


def plot_sample(ax, sample, color, zorder, point_size=10, alpha=1):
    x = sample[:, 0]
    y = sample[:, 1]

    ax.scatter(
        x=x,
        y=y,
        s=point_size,
        color=color,
        alpha=alpha,
        zorder=zorder)


def extract_curve_sections(curve, step, sample_points):
    indices = list(range(curve.shape[0]))[::step]
    sampled_sections = []
    full_sections = []

    for index1, index2, index3 in zip(indices, indices[1:], indices[2:]):
        sampled_indices1 = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=sample_points,
            start_point_index=index1,
            end_point_index=index2)

        sampled_indices2 = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=sample_points,
            start_point_index=index1,
            end_point_index=index2)

        sampled_indices3 = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=sample_points,
            start_point_index=index2,
            end_point_index=index3)

        sampled_indices4 = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=sample_points,
            start_point_index=index1,
            end_point_index=index3)

        sampled_section = {
            'indices': [sampled_indices1, sampled_indices2, sampled_indices3, sampled_indices4],
            'samples': [curve[sampled_indices1], curve[sampled_indices2], curve[sampled_indices3], curve[sampled_indices4]],
            'accumulate': [True, False, False, False]
        }

        sampled_sections.append(sampled_section)

        full_indices1 = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=step+1,
            start_point_index=index1,
            end_point_index=index2)

        full_indices2 = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=step+1,
            start_point_index=index1,
            end_point_index=index2)

        full_indices3 = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=step+1,
            start_point_index=index2,
            end_point_index=index3)

        full_indices4 = curve_sampling.sample_curve_section_indices(
            curve=curve,
            supporting_points_count=2*step + 1,
            start_point_index=index1,
            end_point_index=index3)

        full_section = {
            'indices': [full_indices1, full_indices2, full_indices3, full_indices4],
            'samples': [curve[full_indices1], curve[full_indices2], curve[full_indices3], curve[full_indices4]],
            'accumulate': [True, False, False, False]
        }

        full_sections.append(full_section)

    return {
        'sampled_sections': sampled_sections,
        'full_sections': full_sections,
        'curve': curve
    }


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


def predict_arclength_by_index(model, curve_sections):
    sampled_sections = curve_sections['sampled_sections']
    predicted_arclength = numpy.zeros([len(sampled_sections) + 1, 2, 4])
    for i, sampled_section in enumerate(sampled_sections):
        point_index = i + 1
        for j, (indices, sample, accumulate) in enumerate(zip(sampled_section['indices'], sampled_section['samples'], sampled_section['accumulate'])):
            sample = curve_processing.normalize_curve(curve=sample, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
            arclength_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample).double(), dim=0), dim=0).cuda()
            with torch.no_grad():
                predicted_arclength[point_index, 0, j] = point_index
                predicted_arclength[point_index, 1, j] = torch.squeeze(model(arclength_batch_data), dim=0).cpu().detach().numpy()

            if accumulate is True:
                predicted_arclength[point_index, 1, j] = predicted_arclength[point_index, 1, j] + predicted_arclength[i, 1, j]

    return predicted_arclength


def generate_curve_arclength_records(model, curves, transform_type, comparision_curves_count, step, sample_points):
    curve_arclength_records = []

    factors = []
    for curve_index, curve in enumerate(curves):
        comparision_curves = [curve_processing.center_curve(curve=curve)]
        for i in range(comparision_curves_count):
            if transform_type == 'euclidean':
                transform = euclidean_transform.random_euclidean_transform_2d()
            elif transform_type == 'equiaffine':
                transform = affine_transform.random_equiaffine_transform_2d()
            transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
            comparision_curves.append(curve_processing.center_curve(curve=transformed_curve))

        curve_arclength_record = []
        for i, comparision_curve in enumerate(comparision_curves):
            curve_sections = extract_curve_sections(
                curve=comparision_curve,
                step=step,
                sample_points=sample_points)

            true_arclength = calculate_arclength_by_index(
                curve_sections=curve_sections,
                transform_type=transform_type)

            predicted_arclength = predict_arclength_by_index(
                model=model,
                curve_sections=curve_sections)

            curve_arclength_record.append({
                'curve_sections': curve_sections,
                'true_arclength': true_arclength,
                'predicted_arclength': predicted_arclength
            })

            # if i == 0:
            factor = numpy.mean(true_arclength[1:, 1, 0] / predicted_arclength[1:, 1, 0])
            factors.append(factor)

        curve_arclength_records.append(curve_arclength_record)

    factor = numpy.mean(numpy.array(factors))
    for curve_arclength_record in curve_arclength_records:
        for curve_arclength in curve_arclength_record:
            curve_arclength['predicted_arclength'][:, 1, :] *= factor

    return curve_arclength_records


def plot_curve_arclength_record(curve_arclength_record, true_arclength_colors, predicted_arclength_colors, sample_colors, curve_color, anchor_color, first_anchor_color):
    fig, axes = plt.subplots(3, 1, figsize=(20,20))
    fig.patch.set_facecolor('white')
    for axis in axes:
        for label in (axis.get_xticklabels() + axis.get_yticklabels()):
            label.set_fontsize(10)

    axes[0].axis('equal')
    for i, curve_arclength in enumerate(curve_arclength_record):
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
    for i, curve_arclength in enumerate(curve_arclength_record):
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

    for i, curve_arclength in enumerate(curve_arclength_record):
        true_arclength = curve_arclength['true_arclength']
        predicted_arclength = curve_arclength['predicted_arclength']

        d = {
            'True [i, i+1]': true_arclength[1:, 1, 1],
            'True [i+1, i+2]': true_arclength[1:, 1, 2],
            'True [i, i+2]': true_arclength[1:, 1, 3],
            'True [i, i+1] + True [i+1, i+2]': true_arclength[1:, 1, 1] + true_arclength[1:, 1, 2],
            'Pred [i, i+1]': predicted_arclength[1:, 1, 1],
            'Pred [i+1, i+2]': predicted_arclength[1:, 1, 2],
            'Pred [i, i+2]': predicted_arclength[1:, 1, 3],
            'Pred [i, i+1] + Pred [i+1, i+2]': predicted_arclength[1:, 1, 1] + predicted_arclength[1:, 1, 2],
            'Diff [i, i+2]': numpy.abs((true_arclength[1:, 1, 3] - predicted_arclength[1:, 1, 3]) / true_arclength[1:, 1, 3]) * 100
        }

        df = pandas.DataFrame(data=d)

        style = df.style.set_properties(**{'background-color': true_arclength_colors[i]}, subset=list(d.keys())[:4])
        style = style.set_properties(**{'background-color': predicted_arclength_colors[i]}, subset=list(d.keys())[4:8])
        style = style.set_properties(**{'color': 'white', 'border-color': 'black', 'border-style': 'solid', 'border-width': '1px'})

        display(HTML(style.render()))

    # predicted_arclength1 = curve_arclength_record[0]['predicted_arclength']
    # predicted_arclength2 = curve_arclength_record[1]['predicted_arclength']
    # display(HTML((numpy.mean(predicted_arclength1[1:, 1, 3] - predicted_arclength2[1:, 1, 3])))

    predicted_arclength1 = curve_arclength_record[0]['predicted_arclength']
    predicted_arclength2 = curve_arclength_record[1]['predicted_arclength']

    d = {
        'Diff [i, i+2]': (((numpy.abs(predicted_arclength1[1:, 1, 3] - predicted_arclength2[1:, 1, 3]) / predicted_arclength1[1:, 1, 3]) + (numpy.abs(predicted_arclength1[1:, 1, 3] - predicted_arclength2[1:, 1, 3]) / predicted_arclength2[1:, 1, 3])) / 2) * 100
    }

    df = pandas.DataFrame(data=d)

    # style = df.style.set_properties(**{'background-color': true_arclength_colors[i]}, subset=list(d.keys())[:4])
    # style = style.set_properties(**{'background-color': predicted_arclength_colors[i]}, subset=list(d.keys())[4:8])
    # style = style.set_properties(**{'color': 'white', 'border-color': 'black', 'border-style': 'solid', 'border-width': '1px'})

    display(HTML(df.style.render()))


    axes[2].set_xlabel('Index', fontsize=18)
    axes[2].set_ylabel(r'$\kappa^{\frac{1}{3}}$', fontsize=18)
    for i, curve_arclength in enumerate(curve_arclength_record):
        curve_sections = curve_arclength['curve_sections']
        curve = curve_sections['curve']
        curvature = curve_processing.calculate_euclidean_curvature(curve=curve)
        plot_curvature(ax=axes[2], curvature=numpy.cbrt(curvature), color=sample_colors[i])

    plt.show()


def plot_curve_arclength_records(curve_arclength_records, true_arclength_colors, predicted_arclength_colors, sample_colors, curve_color='orange', anchor_color='blue', first_anchor_color='black'):
    for curve_arclength_record in curve_arclength_records:
        plot_curve_arclength_record(
            curve_arclength_record=curve_arclength_record,
            true_arclength_colors=true_arclength_colors,
            predicted_arclength_colors=predicted_arclength_colors,
            sample_colors=sample_colors,
            curve_color=curve_color,
            anchor_color=anchor_color,
            first_anchor_color=first_anchor_color)


# def plot_sectioned_curve(axes_list, axis_index, evaluated_curves_arclength, sample_colors, curve_color='orange', anchor_color='blue', first_anchor_color='black'):
#     for axes_index, evaluated_curve_arclength in enumerate(evaluated_curves_arclength):
#         axes = axes_list[axes_index]
#         ax = axes[axis_index]
#         for i, evaluated_comparision_curve_arclength in enumerate(evaluated_curve_arclength):
#             curve_sections = evaluated_comparision_curve_arclength['curve_sections']
#             curve = curve_sections['curve']
#             for j, sampled_section in enumerate(curve_sections['sampled_sections']):
#                 sample = sampled_section['samples'][0]
#                 ax.set_xlabel('X Coordinate', fontsize=18)
#                 ax.set_ylabel('Y Coordinate', fontsize=18)
#                 plot_curve(ax=ax, curve=curve, color=curve_color, linewidth=3)
#                 plot_sample(ax=ax, sample=sample, point_size=10, color=sample_colors[i], zorder=150)
#                 plot_sample(ax=ax, sample=numpy.array([[sample[0,0] ,sample[0, 1]], [sample[-1,0] ,sample[-1, 1]]]), point_size=70, alpha=1, color=anchor_color, zorder=200)
#                 if j == 0:
#                     plot_sample(ax=ax, sample=numpy.array([[sample[0,0] ,sample[0, 1]]]), point_size=70, alpha=1, color=first_anchor_color, zorder=300)
#         plt.show()
#
#
# def plot_arclength_by_index(axes_list, axis_index, evaluated_curves_arclength, true_arclength_colors, predicted_arclength_colors):
#     for axes_index, evaluated_curve_arclength in enumerate(evaluated_curves_arclength):
#         true_arclength_legend_labels = []
#         predicted_arclength_legend_labels = []
#         axes = axes_list[axes_index]
#         ax = axes[axis_index]
#         ax.set_xlabel('Index', fontsize=18)
#         ax.set_ylabel('Arc-Length', fontsize=18)
#         ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#         for i, evaluated_comparision_curve_arclength in enumerate(evaluated_curve_arclength):
#             true_arclength = evaluated_comparision_curve_arclength['true_arclength']
#             predicted_arclength = evaluated_comparision_curve_arclength['predicted_arclength']
#
#             plot_sample(ax=ax, sample=true_arclength[:, :, 0], point_size=40, color=true_arclength_colors[i], zorder=250)
#             plot_curve(ax=ax, curve=true_arclength[:, :, 0], linewidth=2, color=true_arclength_colors[i], zorder=150)
#             true_arclength_legend_labels.append(f'True Arclength (Curve #{i + 1})')
#
#             plot_sample(ax=ax, sample=predicted_arclength[:, :, 0], point_size=40, color=predicted_arclength_colors[i], zorder=250)
#             plot_curve(ax=ax, curve=predicted_arclength[:, :, 0], linewidth=2, color=predicted_arclength_colors[i], zorder=150)
#             predicted_arclength_legend_labels.append(f'Predicted Arclength (Curve #{i + 1})')
#
#             d = {
#                 'True [i, i+1]': true_arclength[1:, 1, 1],
#                 'True [i+1, i+2]': true_arclength[1:, 1, 2],
#                 'True [i, i+2]': true_arclength[1:, 1, 3],
#                 'True [i, i+1] + [i+1, i+2]': true_arclength[1:, 1, 1] + true_arclength[1:, 1, 2],
#                 'Pred [i, i+1]': predicted_arclength[1:, 1, 1],
#                 'Pred [i+1, i+2]': predicted_arclength[1:, 1, 2],
#                 'Pred [i, i+2]': predicted_arclength[1:, 1, 3],
#                 'Pred [i, i+1] + [i+1, i+2]': predicted_arclength[1:, 1, 1] + predicted_arclength[1:, 1, 2]
#             }
#
#             df = pandas.DataFrame(data=d)
#
#             style = df.style.set_properties(**{'background-color': '#AA0000'}, subset=list(d.keys())[:4])
#             style = style.set_properties(**{'background-color': '#0000AA'}, subset=list(d.keys())[4:8])
#             style = style.set_properties(**{'color': 'white', 'border-color': 'black','border-style' :'solid' ,'border-width': '1px'})
#
#             display(HTML(style.render()))
#
#         true_arclength_legend_lines = [matplotlib.lines.Line2D([0], [0], color=color, linewidth=3) for color in true_arclength_colors]
#         predicted_arclength_legend_lines = [matplotlib.lines.Line2D([0], [0], color=color, linewidth=3) for color in predicted_arclength_colors]
#         legend_labels = true_arclength_legend_labels + predicted_arclength_legend_labels
#         legend_lines = true_arclength_legend_lines + predicted_arclength_legend_lines
#         ax.legend(legend_lines, legend_labels, prop={'size': 20})
#         plt.show()
