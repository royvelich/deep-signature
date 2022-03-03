# python peripherals
import os
from argparse import ArgumentParser
from os import walk
from pathlib import Path
import pathlib

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
    parser = ArgumentParser()
    parser.add_argument("--images_dir", dest="images_dir", type=str)
    parser.add_argument("--curves_dir", dest="curves_dir", type=str)
    args = parser.parse_args()

    image_file_paths = []
    for (dir_path, dir_names, file_names) in walk(args.images_dir):
        for file_name in file_names:
            image_file_paths.append(os.path.normpath(os.path.join(dir_path, file_name)))
        break

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
            if distance < 1e-3:
                closed_curves.append(curve)

        curves = closed_curves

        limit = 0
        if stem == 'butterflies':
            limit = 57
        elif stem == 'cats':
            limit = 7
        elif stem == 'dogs':
            limit = 9
        elif stem == 'trees':
            limit = 4
        elif stem == 'chickens':
            limit = 7
        elif stem == 'birds':
            limit = 30
        elif stem == 'leaves':
            limit = 35
        elif stem == 'bears':
            limit = 5
        elif stem == 'insects':
            limit = 300
        elif stem == 'basketball':
            limit = 52
        elif stem == 'animals':
            limit = 28
        elif stem == 'whales':
            limit = 1
        elif stem == 'branches':
            limit = 7
        elif stem == 'shapes':
            limit = 0
        elif stem == 'rats':
            limit = 2
        elif stem == 'profiles':
            limit = 1
        elif stem == 'clouds':
            limit = 0

        curves.sort(key=lambda curve: curve.shape[0], reverse=False)
        curves = curves[limit:]
        sampling_ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        multimodalies = [10, 15, 20, 25, 30, 35]
        for multimodality in multimodalies:
            for sampling_ratio in sampling_ratios:
                downsampled_curves = []
                plots_file_path = os.path.normpath(os.path.join(args.curves_dir, 'plots', stem, f'multimodality_{multimodality}', str(sampling_ratio).replace(".", "_")))
                pathlib.Path(plots_file_path).mkdir(parents=True, exist_ok=True)
                for i, curve in enumerate(curves):
                    curve_points_count = curve.shape[0]
                    sampling_points_count = int(sampling_ratio * curve_points_count)
                    dist = discrete_distribution.random_discrete_dist(bins=curve_points_count, multimodality=multimodality, max_density=1, count=1)[0]
                    indices_pool = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=sampling_points_count)
                    downsampled_curve = curve[indices_pool]
                    downsampled_curves.append(downsampled_curve)

                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40, 40))
                    plot_sample(ax=ax, sample=downsampled_curve, color='red', zorder=10, point_size=150)
                    plt.axis('off')
                    plot_file_path = os.path.normpath(os.path.join(plots_file_path, f'{stem}_{i}.png'))
                    fig.savefig(plot_file_path)
                    plt.close(fig)

                curves_base_dir_path = os.path.normpath(os.path.join(args.curves_dir, f'multimodality_{multimodality}'))
                pathlib.Path(curves_base_dir_path).mkdir(parents=True, exist_ok=True)
                curves_file_path = os.path.normpath(os.path.join(curves_base_dir_path, f'{stem}_{str(sampling_ratio).replace(".", "_")}.npy'))
                numpy.save(curves_file_path, downsampled_curves)
