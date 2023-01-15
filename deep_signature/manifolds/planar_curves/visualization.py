# python peripheral
from typing import Tuple, Optional, Union

# numpy
import numpy

# matplotlib
import matplotlib
import matplotlib.pyplot
import matplotlib.collections
import matplotlib.axes


def plot_line(x: numpy.ndarray, y: numpy.ndarray, ax: matplotlib.axes.Axes, closed: bool = True, line_width: float = 2, markersize: float = 2, line_style: str = '-',  marker: str = '.', alpha: float = 1.0, color: str = 'red', zorder: int = 1, equal_axis: bool = False, force_limits: bool = True):
    if force_limits is True:
        ax.set_xlim(left=numpy.min(x), right=numpy.max(x))
        ax.set_ylim(bottom=numpy.min(y), top=numpy.max(y))

    if closed is True:
        x = numpy.append(arr=x, values=[x[0]])
        y = numpy.append(arr=y, values=[y[0]])

    if equal_axis is True:
        ax.axis('equal')
    ax.plot(x, y, linestyle=line_style, linewidth=line_width, marker=marker, markersize=markersize, alpha=alpha, color=color, zorder=zorder)


# https://nbviewer.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
def plot_multicolor_line(
        x: numpy.ndarray,
        y: numpy.ndarray,
        ax: matplotlib.axes.Axes,
        closed: bool = True,
        line_width: float = 2,
        alpha: float = 1.0,
        cmap: str = 'hsv',
        z_range: Optional[int] = None,
        reference_x: Optional[numpy.ndarray] = None,
        color: Optional[str] = None,
        zorder: int = 1,
        equal_axis: bool = False):
    # indices = list(range(x.shape[0]))
    # z = numpy.linspace(0.0, 1.0, len(indices))
    # z = numpy.linspace(0.0, 1.0, 1300)
    if z_range is None:
        z_range = x.shape[0]
        z = x / z_range
    else:
        z = reference_x / z_range

    ax.set_xlim(left=numpy.min(x), right=numpy.max(x))
    ax.set_ylim(bottom=numpy.min(y), top=numpy.max(y))

    if closed is True:
        x = numpy.append(arr=x, values=[x[0]])
        y = numpy.append(arr=y, values=[y[0]])

    if equal_axis is True:
        ax.axis('equal')

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)

    # if norm is None:
    norm = matplotlib.pyplot.Normalize(z.min(), z.max())

    line_collection = matplotlib.collections.LineCollection(
        segments=segments,
        array=z,
        cmap=cmap,
        colors=color,
        norm=norm,
        linewidth=line_width,
        alpha=alpha,
        zorder=zorder)

    ax.add_collection(line_collection)


def plot_multicolor_scatter(
        x: numpy.ndarray,
        y: numpy.ndarray,
        c: numpy.ndarray,
        ax: matplotlib.axes.Axes,
        point_size: float = 2,
        alpha: float = 1.0,
        cmap: Optional[str] = 'red',
        color: Optional[str] = '#FF0000',
        zorder: int = 1,
        norm: matplotlib.pyplot.Normalize = matplotlib.pyplot.Normalize(0.0, 1.0),
        xlim: Optional[numpy.ndarray] = None,
        ylim: Optional[numpy.ndarray] = None):
    if cmap is not None:
        ax.scatter(
            x=x,
            y=y,
            c=c,
            s=point_size,
            cmap=cmap,
            alpha=alpha,
            norm=norm,
            zorder=zorder)
    else:
        ax.scatter(
            x=x,
            y=y,
            s=point_size,
            color=color,
            alpha=alpha,
            zorder=zorder)

    ax.axis('equal')

    # if xlim is not None:
    #     ax.set_xlim(left=xlim[0], right=xlim[1])
    #
    # if ylim is not None:
    #     ax.set_ylim(bottom=ylim[0], top=ylim[1])
    #
    # ax.set_aspect(aspect='equal')


