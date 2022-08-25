# python peripherals
import pathlib
import os

# numpy
import numpy

# matplotlib
import matplotlib.collections as mcoll

# ipython
from IPython.display import display, HTML

# utils
from utils import settings

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# plotly
from plotly.subplots import make_subplots
from plotly import graph_objects


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



def plot_curvature_plotly(fig, row, col, name, curvature, point_size=2, line_width=2, line_color='green', mode='markers'):
    x = numpy.array(range(curvature.shape[0]))
    y = curvature

    fig.add_trace(
        trace=graph_objects.Scatter(
            name=name,
            x=x,
            y=y,
            mode=mode,
            line={
                'color': line_color,
                'width': line_width
            },
            marker={
                'color': line_color,
                'size': point_size
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


# -------------
# PLOT ROUTINES
# -------------
def plot_curve_comparisons(curve_records, curve_colors, sampling_ratio, transformation_group_type, normalize_signature=False, plot_to_screen=True, plots_dir_name='signature_plots'):
    for i, curve_record in enumerate(curve_records):
        display(HTML(f'<H1>Curve {i+1} - Comparison</H1>'))
        plot_curve_comparison(
            curve_index=i,
            curve_record=curve_record,
            curve_colors=curve_colors,
            sampling_ratio=sampling_ratio,
            transformation_group_type=transformation_group_type,
            plot_to_screen=plot_to_screen,
            normalize_signature=normalize_signature,
            plots_dir_name=plots_dir_name)


def plot_curve_comparison(curve_index, curve_record, curve_colors, sampling_ratio, transformation_group_type, plot_to_screen, normalize_signature=False, plots_dir_name='signature_plots'):
    plots_dir_path = os.path.normpath(os.path.join(settings.plots_dir, f"./{plots_dir_name}_{sampling_ratio}_{transformation_group_type}"))
    pathlib.Path(plots_dir_path).mkdir(parents=True, exist_ok=True)
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
    fig = make_subplots(rows=1, cols=4, subplot_titles=('<b>Reference Curve</b>', '<b>Transformed Curve #1</b>', '<b>Transformed Curve #2</b>', '<b>Down-Sampled Transformed Curves</b>'))

    orig_curve = curve_record['curve']
    plot_curve_plotly(fig=fig, row=1, col=1, curve=orig_curve, name='Reference Curve', line_width=settings.plotly_graph_line_width, line_color=curve_colors[-1])

    for i, comparison in enumerate(curve_record['comparisons']):
        curve = comparison['curve']
        sampled_curve = comparison['sampled_curve']

        plot_curve_plotly(fig=fig, row=1, col=i+2, curve=curve, name=f'Transformed Curve #{i+1}', line_width=settings.plotly_graph_line_width, line_color=curve_colors[i])
        plot_curve_sample_plotly(fig=fig, row=1, col=i + 2, name=f'', curve=curve, curve_sample=numpy.expand_dims(curve[0,:], axis=0), color='black', point_size=settings.plotly_sample_point_size)
        # plot_curve_sample_plotly(fig=fig, row=1, col=i + 2, name=f'', curve=curve, curve_sample=numpy.expand_dims(curve[200,:], axis=0), color='black', point_size=settings.plotly_sample_point_size)

        plot_curve_plotly(fig=fig, row=1, col=i+2, curve=orig_curve, name=f'', line_width=settings.plotly_graph_line_width, line_color=curve_colors[-1])
        plot_curve_sample_plotly(fig=fig, row=1, col=i + 2, name=f'', curve=orig_curve, curve_sample=numpy.expand_dims(orig_curve[0,:], axis=0), color='black', point_size=settings.plotly_sample_point_size)
        # plot_curve_sample_plotly(fig=fig, row=1, col=i + 2, name=f'', curve=orig_curve, curve_sample=numpy.expand_dims(orig_curve[200,:], axis=0), color='black', point_size=settings.plotly_sample_point_size)

        # plot_curve_plotly(fig=fig, row=1, col=4, curve=curve, name='', line_width=settings.plotly_graph_line_width, line_color=curve_colors[i])
        plot_curve_sample_plotly(fig=fig, row=1, col=4, name='', curve=curve, curve_sample=sampled_curve, color=curve_colors[i], point_size=int(settings.plotly_sample_point_size * 0.5))
        plot_curve_sample_plotly(fig=fig, row=1, col=4, name=f'', curve=curve, curve_sample=numpy.expand_dims(curve[0, :], axis=0), color='black', point_size=int(settings.plotly_sample_point_size * 1.5))
        # plot_curve_sample_plotly(fig=fig, row=1, col=4, name=f'', curve=curve, curve_sample=numpy.expand_dims(curve[100, :], axis=0), color='black', point_size=int(settings.plotly_sample_point_size * 1.5))

        # plot_curve_plotly(fig=fig, row=1, col=5, name='', curve=curve, line_color=curve_colors[0], line_width=settings.plotly_graph_line_width)
        # plot_curve_sample_plotly(fig=fig, row=1, col=5, name=f'', curve=curve, curve_sample=numpy.expand_dims(curve[0, :], axis=0), color='black', point_size=int(settings.plotly_sample_point_size * 1.5))
        # plot_curve_sample_plotly(fig=fig, row=1, col=5, name=f'', curve=curve, curve_sample=numpy.expand_dims(curve[100, :], axis=0), color='black', point_size=int(settings.plotly_sample_point_size * 1.5))

    for i in range(len(curve_record['comparisons']) + 3):
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

    fig.write_image(os.path.join(plots_dir_path, f'curves_together_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    if plot_to_screen is True:
        fig.show()





    # ----------------------
    # PLOT CURVES TOGETHER 2
    # ----------------------
    fig = make_subplots(rows=1, cols=1, subplot_titles=(''))
    for i, comparison in enumerate(curve_record['comparisons']):
        curve = comparison['curve']
        if i == 0:
            plot_curve_plotly(fig=fig, row=1, col=1, curve=curve, name='', line_width=settings.plotly_graph_line_width, line_color=curve_colors[-1])
        else:
            sampled_curve = comparison['sampled_curve']
            plot_curve_sample_plotly(fig=fig, row=1, col=1, name='', curve=curve, curve_sample=sampled_curve, color='black', point_size=int(settings.plotly_sample_point_size * 0.8))
            # plot_curve_sample_plotly(fig=fig, row=1, col=1, name=f'', curve=curve, curve_sample=numpy.expand_dims(curve[0, :], axis=0), color='black', point_size=settings.plotly_sample_point_size)

    fig.update_yaxes(
        scaleanchor='x1',
        scaleratio=1,
        row=1,
        col=1)

    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))

    fig.update_layout(yaxis1=dict(range=get_range()))

    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig.update_layout(showlegend=False)

    fig.write_image(os.path.join(plots_dir_path, f'curves_together2_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    if plot_to_screen is True:
        fig.show()







    # ---------------------
    # PLOT ORIGINAL CURVE
    # ---------------------
    # fig = make_subplots(rows=1, cols=1, subplot_titles=['<b>Curve</b>'])
    #
    # orig_curve = curve_record['curve']
    # plot_curve_plotly(fig=fig, row=1, col=1, curve=orig_curve, name='Reference Curve', line_width=settings.plotly_graph_line_width, line_color=curve_colors[0])
    # plot_curve_sample_plotly(fig=fig, row=1, col=1, name=f'', curve=orig_curve, curve_sample=numpy.expand_dims(orig_curve[0, :], axis=0), color='black', point_size=int(settings.plotly_sample_point_size * 1.25))
    #
    # fig.update_yaxes(
    #     scaleanchor=f'x',
    #     scaleratio=1,
    #     row=1,
    #     col=1)
    #
    # fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    #
    # fig['layout']['xaxis']['title'] = 'X Coordinate'
    # fig['layout']['yaxis']['title'] = 'Y Coordinate'
    #
    # fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    #
    # fig.update_layout(showlegend=False)
    #
    # fig.write_image(os.path.join(plots_dir_path, f'orig_curve{curve_index}.svg'), width=settings.plotly_write_image_height, height=settings.plotly_write_image_height)
    # if plot_to_screen is True:
    #     fig.show()

    # -------------------------------
    # PLOT CURVE SAMPLES SIDE BY SIDE
    # -------------------------------
    # fig = make_subplots(rows=1, cols=len(curve_record['comparisons']), subplot_titles=('<b>Sampled Curve #1</b>', '<b>Sampled Curve #2</b>'))
    #
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     sampled_curve = comparison['sampled_curve']
    #     curve = comparison['curve']
    #     plot_curve_sample_plotly(fig=fig, row=1, col=i+1, name=f'Sampled Curve {i+1}', curve=curve, curve_sample=sampled_curve, color=curve_colors[i], point_size=settings.plotly_sample_point_size)
    #
    # for i in range(len(curve_record['comparisons']) + 1):
    #     fig.update_yaxes(
    #         scaleanchor=f'x{i+1}',
    #         scaleratio=1,
    #         row=1,
    #         col=i+1)
    #
    # fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    # fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    #
    # fig['layout']['xaxis']['title'] = 'X Coordinate'
    # fig['layout']['yaxis']['title'] = 'Y Coordinate'
    #
    # fig['layout']['xaxis2']['title'] = 'X Coordinate'
    # fig['layout']['yaxis2']['title'] = 'Y Coordinate'
    #
    # fig.update_layout(yaxis1=dict(range=get_range()))
    # fig.update_layout(yaxis2=dict(range=get_range()))
    #
    # fig.update_layout(showlegend=False)
    #
    # fig.write_image(os.path.join(plots_dir_path, f'curve_samples_side_by_side_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    # if plot_to_screen is True:
    #     fig.show()

    # # ----------------------------------------------------------------------------------
    # # PLOT CURVE SAMPLES, ANCHORS AND PREDICTED CURVATURE SIDE BY SIDE (WITHOUT BUTTONS)
    # # ----------------------------------------------------------------------------------
    # left_width = 0.25
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     fig = make_subplots(rows=1, cols=3, column_widths=[left_width, left_width, 1 - (2*left_width)], subplot_titles=('<b>Sampled Curve</b>', '<b>Anchors</b>',  '<b>Predicted Curvature at Anchors</b>'))
    #     sampled_curve = comparison['sampled_curve']
    #     anchors = comparison['anchors']
    #     anchor_indices = comparison['anchor_indices']
    #     curve = comparison['curve']
    #     curvature_comparison = comparison['curvature_comparison']
    #     predicted_curvature = curvature_comparison['predicted_curvature']
    #
    #     plot_curve_sample_plotly(fig=fig, row=1, col=1, name="Sampled Curve", curve=curve, curve_sample=sampled_curve, color=curve_colors[i], point_size=settings.plotly_sample_point_size)
    #     plot_curve_sample_plotly(fig=fig, row=1, col=2, name="Anchors", curve=curve, curve_sample=anchors, color=anchor_indices, point_size=settings.plotly_sample_point_size)
    #     plot_curvature_with_cmap_plotly(fig=fig, row=1, col=3, name="Predicted Curvature", curve=curve, curvature=predicted_curvature[:, 1], indices=anchor_indices, line_color='grey', line_width=settings.plotly_graph_line_width, point_size=settings.plotly_sample_anchor_size, color_scale='hsv')
    #
    #     fig.update_yaxes(
    #         scaleanchor="x1",
    #         scaleratio=1,
    #         row=1,
    #         col=1)
    #
    #     fig.update_yaxes(
    #         scaleanchor="x2",
    #         scaleratio=1,
    #         row=1,
    #         col=2)
    #
    #     fig['layout']['xaxis']['title'] = 'X Coordinate'
    #     fig['layout']['yaxis']['title'] = 'Y Coordinate'
    #
    #     fig['layout']['xaxis2']['title'] = 'X Coordinate'
    #     fig['layout']['yaxis2']['title'] = 'Y Coordinate'
    #
    #     fig['layout']['xaxis3']['title'] = 'Anchor Point Index'
    #     fig['layout']['yaxis3']['title'] = 'Predicted Curvature'
    #
    #     curr_range = get_range()
    #     fig.update_layout(yaxis1=dict(range=curr_range))
    #     fig.update_layout(yaxis2=dict(range=curr_range))
    #
    #     fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize), showlegend=False)
    #
    #     fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    #
    #     fig.write_image(os.path.join(plots_dir_path, f'curve_samples_and_predicted_curvature_{curve_index}_{i}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    #     if plot_to_screen is True:
    #         fig.show()
    #
    # # --------------------
    # # CURVATURE AT ANCHORS
    # # --------------------
    # fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted Curvature at Anchors (Transformed Curve #1 vs. Transformed Curve #2)</b>',))
    #
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     curvature_comparison = comparison['curvature_comparison']
    #     predicted_curvature = curvature_comparison['predicted_curvature']
    #
    #     plot_curve_plotly(fig=fig, row=1, col=1, name=f'Predicted Curvature at Anchors #{i+1}', curve=predicted_curvature, line_width=settings.plotly_graph_line_width, line_color=curve_colors[i], mode='lines')
    #
    # fig['layout']['xaxis']['title'] = 'Anchor Point Index'
    # fig['layout']['yaxis']['title'] = 'Predicted Curvature'
    #
    # fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    #
    # fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    #
    # fig.write_image(os.path.join(plots_dir_path, f'predicted_curves_together_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    # fig.show()

    # ------------------------------------
    # CURVATURE VS. INDEX OF SAMPLE POINTS
    # ------------------------------------
    # fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted Curvature as a Function of Point Index</b>',))
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     curvature_comparison = comparison['curvature_comparison']
    #     predicted_curvature = curvature_comparison['predicted_curvature']
    #
    #     plot_curve_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', curve=predicted_curvature, point_size=settings.plotly_sample_point_size, line_width=settings.plotly_graph_line_width, line_color=curve_colors[i], mode='markers')
    #
    # fig['layout']['xaxis']['title'] = 'Sample Point Index'
    # fig['layout']['yaxis']['title'] = 'Predicted Curvature'
    #
    # fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    #
    # fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    #
    # fig.write_image(os.path.join(plots_dir_path, f'predicted_curvature_as_function_of_index_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    # if plot_to_screen is True:
    #     fig.show()

    # -----------------------------------------
    # CURVATURE VS. ARC-LENGTH OF SAMPLE POINTS
    # -----------------------------------------
    # fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted Curvature as a Function of Predicted Arc-Length</b>',))
    # comparisons = curve_record['comparisons']
    # for i, comparison in enumerate(comparisons):
    #     curvature_comparison = comparison['curvature_comparison']
    #     arclength_comparison = comparison['arclength_comparison']
    #     predicted_curvature = curvature_comparison['predicted_curvature']
    #     predicted_arclength = arclength_comparison['predicted_arclength']
    #     predicted_signature = comparison['predicted_signature']
    #     true_curvature = curvature_comparison['true_curvature']
    #     true_arclength = arclength_comparison['true_arclength']
    #
    #     # if transformation_group_type != 'affine':
    #     if transformation_group_type == 'euclidean':
    #         if i == 0:
    #             if transformation_group_type == 'equiaffine':
    #                 true_curvature[:, 1] = numpy.clip(true_curvature[:, 1], a_min=numpy.min(predicted_curvature[:, 1]), a_max=numpy.max(predicted_curvature[:, 1]))
    #                 ratio = 1
    #             elif transformation_group_type == 'euclidean':
    #                 ratio = float(numpy.max(numpy.abs(true_curvature[:, 1])) / numpy.max(numpy.abs(predicted_curvature[:, 1])))
    #
    #             plot_graph_plotly(fig=fig, row=1, col=1, name=f'Ground Truth', x=true_arclength[:, 1], y=true_curvature[:, 1], point_size=int(settings.plotly_sample_point_size * 1.3), line_width=settings.plotly_graph_line_width, line_color=curve_colors[-1], mode='markers')
    #     else:
    #         ratio = 1
    #
    #     signature_arclength = predicted_signature[:, 0].copy()
    #     if normalize_signature is True:
    #         predicted_signature_1 = comparisons[1]['predicted_signature']
    #         signature_arclength = signature_arclength * (predicted_signature_1[-1, 0] / predicted_signature[-1, 0])
    #
    #     plot_graph_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', x=signature_arclength, y=ratio*predicted_signature[:, 1], point_size=int(settings.plotly_sample_point_size * 1.3), line_width=settings.plotly_graph_line_width, line_color=curve_colors[i], mode='markers')
    #
    # fig['layout']['xaxis']['title'] = 'Predicted Arc-Length'
    # fig['layout']['yaxis']['title'] = 'Predicted Curvature'
    #
    # fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    #
    # fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    #
    # fig.write_image(os.path.join(plots_dir_path, f'predicted_curvature_as_function_of_arclength_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    # if plot_to_screen is True:
    #     fig.show()



    # ------------------------------------
    # KAPPA VS. INDEX OF SAMPLE POINTS
    # ------------------------------------
    fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>True KAPPA as a Function of Point Index</b>',))
    for i, comparison in enumerate(curve_record['comparisons']):
        differential_invariants_comparison = comparison['differential_invariants_comparison']
        true_curvature = differential_invariants_comparison['true_curvature']
        plot_curvature_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', curvature=true_curvature[:, 1], point_size=settings.plotly_sample_point_size, line_width=settings.plotly_graph_line_width, line_color=curve_colors[i])

    fig['layout']['xaxis']['title'] = 'Sample Point Index'
    fig['layout']['yaxis']['title'] = 'True Curvature'
    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    if plot_to_screen is True:
        fig.show()

    fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>True KAPPA_S as a Function of Point Index</b>',))
    for i, comparison in enumerate(curve_record['comparisons']):
        differential_invariants_comparison = comparison['differential_invariants_comparison']
        true_ks = differential_invariants_comparison['true_ks']
        plot_curvature_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', curvature=true_ks[:, 1], point_size=settings.plotly_sample_point_size, line_width=settings.plotly_graph_line_width, line_color=curve_colors[i])

    fig['layout']['xaxis']['title'] = 'Sample Point Index'
    fig['layout']['yaxis']['title'] = 'True K_S'
    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    if plot_to_screen is True:
        fig.show()



    fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>True Signature Curve</b>',))
    comparisons = curve_record['comparisons']
    for i, comparison in enumerate(comparisons):
        differential_invariants_comparison = comparison['differential_invariants_comparison']
        true_curvature = differential_invariants_comparison['true_curvature']
        true_ks = differential_invariants_comparison['true_ks']
        plot_graph_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', x=true_ks[:, 1], y=true_curvature[:, 1], point_size=int(settings.plotly_sample_point_size), line_width=1, line_color=curve_colors[i], mode='markers')

    fig['layout']['xaxis']['title'] = 'Kappa_s'
    fig['layout']['yaxis']['title'] = 'Kappa'

    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    fig.update_yaxes(
        scaleanchor='x1',
        scaleratio=1,
        row=1,
        col=1)

    fig.update_layout(width=1000, height=1000)
    if plot_to_screen is True:
        fig.show()






    # ------------------------------------
    # KAPPA VS. INDEX OF SAMPLE POINTS
    # ------------------------------------
    for k in range(2):
        fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted KAPPA as a Function of Point Index</b>',))
        for i, comparison in enumerate(curve_record['comparisons']):
            differential_invariants_comparison = comparison['differential_invariants_comparison']
            predicted_differential_invariants = differential_invariants_comparison['predicted_differential_invariants']
            curvature_comparison = comparison['curvature_comparison']
            predicted_curvature = curvature_comparison['predicted_curvature']

            plot_curvature_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', curvature=predicted_differential_invariants[:, k], point_size=settings.plotly_sample_point_size, line_width=settings.plotly_graph_line_width, line_color=curve_colors[i])

        fig['layout']['xaxis']['title'] = 'Sample Point Index'
        fig['layout']['yaxis']['title'] = 'Predicted Curvature'

        fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))

        fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

        fig.write_image(os.path.join(plots_dir_path, f'predicted_curvature_as_function_of_index_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
        if plot_to_screen is True:
            fig.show()


    # ------------------------------------
    # KAPPA_S VS. INDEX OF SAMPLE POINTS
    # ------------------------------------
    # fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted KAPPA_S as a Function of Point Index</b>',))
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     differential_invariants_comparison = comparison['differential_invariants_comparison']
    #     predicted_differential_invariants = differential_invariants_comparison['predicted_differential_invariants']
    #     curvature_comparison = comparison['curvature_comparison']
    #     predicted_curvature = curvature_comparison['predicted_curvature']
    #
    #     plot_curvature_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', curvature=predicted_differential_invariants[:, 1], point_size=settings.plotly_sample_point_size, line_width=settings.plotly_graph_line_width, line_color=curve_colors[i])
    #
    # fig['layout']['xaxis']['title'] = 'Sample Point Index'
    # fig['layout']['yaxis']['title'] = 'Predicted Curvature'
    #
    # fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    #
    # fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    #
    # fig.write_image(os.path.join(plots_dir_path, f'predicted_curvature_as_function_of_index_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    # if plot_to_screen is True:
    #     fig.show()


    # -----------------------------------------
    # KAPPA VS KAPPA_S
    # -----------------------------------------
    fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Predicted Signature Curve</b>',))
    comparisons = curve_record['comparisons']
    for i, comparison in enumerate(comparisons):
        differential_invariants_comparison = comparison['differential_invariants_comparison']
        predicted_differential_invariants = differential_invariants_comparison['predicted_differential_invariants']
        plot_graph_plotly(fig=fig, row=1, col=1, name=f'Sampled Curve #{i+1}', x=predicted_differential_invariants[:, 1], y=predicted_differential_invariants[:, 0], point_size=int(settings.plotly_sample_point_size), line_width=1, line_color=curve_colors[i], mode='markers')

    fig['layout']['xaxis']['title'] = 'Kappa_s'
    fig['layout']['yaxis']['title'] = 'Kappa'

    fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))

    fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)

    fig.write_image(os.path.join(plots_dir_path, f'predicted_signature_curve_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)

    fig.update_yaxes(
        scaleanchor='x1',
        scaleratio=1,
        row=1,
        col=1)

    fig.update_layout(width=1000, height=1000)


    # fig.update_layout(yaxis1=dict(range=get_range()))

    if plot_to_screen is True:
        fig.show()

    # -----------------------------------------
    # CURVATURE VS. ARC-LENGTH OF SAMPLE POINTS
    # -----------------------------------------
    # fig = make_subplots(rows=1, cols=1, subplot_titles=('<b>Euclidean Curvature as a Function of Euclidean Arc-Length</b>',))
    # for i, comparison in enumerate(curve_record['comparisons']):
    #     curvature_comparison = comparison['curvature_comparison']
    #     arclength_comparison = comparison['arclength_comparison']
    #     predicted_curvature = curvature_comparison['predicted_curvature']
    #     predicted_arclength = arclength_comparison['predicted_arclength']
    #     true_curvature = curvature_comparison['true_curvature']
    #     true_arclength = arclength_comparison['true_arclength']
    #
    #     plot_graph_plotly(fig=fig, row=1, col=1, name=f'Ground Truth', x=true_arclength[:, 1], y=true_curvature[:, 1], point_size=settings.plotly_sample_point_size, line_width=settings.plotly_graph_line_width, line_color=curve_colors[0], mode='lines')
    #
    # fig['layout']['xaxis']['title'] = 'Euclidean Arc-Length'
    # fig['layout']['yaxis']['title'] = 'Euclidean Curvature'
    #
    # fig.update_layout(font=dict(size=settings.plotly_axis_title_label_fontsize))
    #
    # fig.update_annotations(font_size=settings.plotly_fig_title_label_fontsize)
    #
    # fig.write_image(os.path.join(plots_dir_path, f'euclidean_curvature_as_function_of_arclength_{curve_index}.svg'), width=settings.plotly_write_image_width, height=settings.plotly_write_image_height)
    # if plot_to_screen is True:
    #     fig.show()


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


def plot_curve_signature_comparisons(curve_records, true_signature_colors, predicted_signature_colors):
    for i, curve_record in enumerate(curve_records):
        display(HTML(f'<H1>Curve {i+1} - Signature Comparison</H1>'))
        plot_curve_signature_comparision(
            curve_index=i,
            curve_record=curve_record,
            true_signature_colors=true_signature_colors,
            predicted_signature_colors=predicted_signature_colors)


def plot_curve_signature_comparision(curve_index, curve_record, true_signature_colors, predicted_signature_colors):
    dir_name = "./signature_comparison"
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(1, 1, figsize=settings.matplotlib_figsize)
    fig.patch.set_facecolor('white')
    for label in (axis.get_xticklabels() + axis.get_yticklabels()):
        label.set_fontsize(settings.matplotlib_axis_tick_label_fontsize)

    axis.set_xlabel('Arc-Length', fontsize=settings.matplotlib_axis_title_label_fontsize)
    axis.set_ylabel('Curvature', fontsize=settings.matplotlib_axis_title_label_fontsize)

    comparisons = curve_record['comparisons']
    for i, comparision in enumerate(comparisons):
        arclength_comparison = comparision['arclength_comparison']
        curvature_comparison = comparision['curvature_comparison']
        predicted_arclength = arclength_comparison['predicted_arclength'][:, 1]
        predicted_curvature = curvature_comparison['predicted_curvature'][:, 1]
        predicted_signature = comparision['predicted_signature']

        true_arclength = arclength_comparison['true_arclength'][:, 1]
        true_curvature = 150*curvature_comparison['true_curvature'][:, 1]

        plot_graph(ax=axis, x=predicted_signature[:, 0], y=predicted_signature[:, 1], color=predicted_signature_colors[i], linewidth=settings.matplotlib_graph_line_width, label=f'Predicted Signature Curve (Transformed Curve #{i + 1})')
        # plot_sample(ax=axis, sample=None, x=predicted_arclength, y=predicted_curvature, point_size=settings.matplotlib_line_point_size, color=predicted_signature_colors[i], zorder=250)

        plot_graph(ax=axis, x=true_arclength, y=true_curvature, color=true_signature_colors[i], linewidth=settings.matplotlib_graph_line_width, label=f'True Signature Curve (Transformed Curve #{i + 1})')
        # plot_sample(ax=axis, sample=None, x=true_arclength, y=true_curvature, point_size=settings.matplotlib_line_point_size, color=true_signature_colors[i], zorder=250)

    axis.legend(prop={'size': settings.matplotlib_legend_label_fontsize})
    axis.set_title(f'Predicted Signature Curve vs. Ground Truth Signature Curve (at Anchors)', fontsize=settings.matplotlib_axis_title_label_fontsize)
    plt.savefig(os.path.join(dir_name, f'signature_{curve_index}.svg'))
    plt.show()
