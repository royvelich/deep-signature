# python peripherals
import os
from argparse import ArgumentParser
from os import walk
from pathlib import Path
import pathlib
import queue
from multiprocessing import Process, Queue, cpu_count

# numpy
import numpy

# skimage
import skimage.io
import skimage.color
import skimage.measure

# common
from utils import common as common_utils
from utils import settings

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# deep-signature
from deep_signature.data_manipulation import curve_sampling, curve_processing
from deep_signature.stats import discrete_distribution
from deep_signature.linalg import transformations

# shapely
from shapely.geometry import Polygon


def plot_graph(ax, x, y, linewidth=2, color='red', alpha=1, zorder=1, label=None):
    return ax.plot(x, y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder, label=label)


def plot_curve(ax, curve, linewidth=2, color='red', alpha=1, zorder=1, label=None):
    x = curve[:, 0]
    y = curve[:, 1]
    return plot_graph(ax=ax, x=x, y=y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder, label=label)


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


if __name__ == '__main__':
    numpy.random.seed(30)
    parser = ArgumentParser()
    parser.add_argument("--images_dir", dest="images_dir", type=str)
    parser.add_argument("--curves_dir", dest="curves_dir", type=str)
    args = parser.parse_args()

    image_file_paths = []
    for (dir_path, dir_names, file_names) in walk(args.images_dir):
        for file_name in file_names:
            image_file_paths.append(os.path.normpath(os.path.join(dir_path, file_name)))
        break

    aggregated_curves = []
    for image_file_index, image_file_path in enumerate(image_file_paths):
        stem = Path(image_file_path).stem

        image = skimage.io.imread(image_file_path)
        gray_image = skimage.color.rgb2gray(image)
        curves = skimage.measure.find_contours(gray_image)

        closed_curves = []
        for curve in curves:
            first_point = curve[0]
            last_point = curve[-1]
            distance = numpy.linalg.norm(x=first_point - last_point, ord=2)
            if distance < 1:
                closed_curves.append(curve)

        external_curves = []
        for i, curve in enumerate(closed_curves):
            polygon = Polygon(curve)
            remove = False
            for j, ref_curve in enumerate(closed_curves):
                ref_polygon = Polygon(ref_curve)
                if (polygon.within(ref_polygon) is True) and (i != j):
                    remove = True

            if not remove:
                if curve.shape[0] > 100:
                    centered_curve = curve_processing.center_curve(curve=curve)
                    external_curves.append(centered_curve)

        aggregated_curves.extend(external_curves)

    curves_file_path = os.path.normpath(os.path.join(args.curves_dir, f'curves.npy'))
    # pathlib.Path(curves_file_path).mkdir(parents=True, exist_ok=True)
    numpy.save(curves_file_path, aggregated_curves)
