# python peripheral
from typing import Tuple

# numpy
import numpy

# matplotlib
import matplotlib
import matplotlib.pyplot
import matplotlib.collections
import matplotlib.axes


# https://nbviewer.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
def plot_multicolor_line(x: numpy.ndarray, y: numpy.ndarray, ax: matplotlib.axes.Axes, line_width: float = 2, alpha: float = 1.0, cmap: str = 'red', zorder: int = 1):
    indices = list(range(x.shape[0]))
    z = numpy.linspace(0.0, 1.0, len(indices))

    ax.set_xlim(left=numpy.min(x), right=numpy.max(x))
    ax.set_ylim(bottom=numpy.min(y), top=numpy.max(y))

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    norm = matplotlib.pyplot.Normalize(0.0, 1.0)
    line_collection = matplotlib.collections.LineCollection(
        segments=segments,
        array=z,
        cmap=cmap,
        norm=norm,
        linewidth=line_width,
        alpha=alpha,
        zorder=zorder)

    ax.add_collection(line_collection)


def plot_multicolor_scatter(x: numpy.ndarray, y: numpy.ndarray, c: numpy.ndarray, ax: matplotlib.axes.Axes, point_size: float = 2, alpha: float = 1.0, cmap: str = 'red', zorder: int = 1):
    norm = matplotlib.pyplot.Normalize(0.0, 1.0)

    if cmap in matplotlib.pyplot.colormaps():
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
            color=cmap,
            alpha=alpha,
            zorder=zorder)

    ax.axis('equal')
