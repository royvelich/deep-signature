# python peripherals
import os
from argparse import ArgumentParser

# numpy
import numpy

# skimage
import skimage.io
import skimage.color
import skimage.measure
from skimage import metrics

# deep signature
from deep_signature.data_manipulation import curve_processing
from deep_signature.linalg import transformations
from deep_signature.data_generation.curve_generation import LevelCurvesGenerator

# common
from utils import common as common_utils
from utils import evaluation as evaluation_utils
from utils import settings

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from deep_signature.linalg import euclidean_transform

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
    transform_horz = euclidean_transform.generate_horizontal_reflection_transform_2d()
    transform_vert = euclidean_transform.generate_vertical_reflection_transform_2d()

    curve = numpy.array([[1, 2], [2, 4], [3, 4], [4, 3]])
    curve = curve_processing.translate_curve(curve=curve, offset=-curve[0])
    # curve1 = numpy.matmul(curve, transform_horz)
    # curve2 = numpy.matmul(curve, transform_vert)
    # curve3 = numpy.matmul(numpy.matmul(curve, transform_horz), transform_vert)
    # curve4 = numpy.matmul(numpy.matmul(curve, transform_vert), transform_horz)
    curve_flipped = numpy.flip(m=curve, axis=0)
    # curve1_flipped = numpy.matmul(curve_flipped, transform_horz)
    # curve2_flipped = numpy.matmul(curve_flipped, transform_vert)
    # curve3_flipped = numpy.matmul(numpy.matmul(curve_flipped, transform_horz), transform_vert)
    # curve4_flipped = numpy.matmul(numpy.matmul(curve_flipped, transform_vert), transform_horz)


    # norm_curve = curve_processing.normalize_curve2(curve=curve)
    # norm_curve_flipped = curve_processing.normalize_curve2(curve=curve_flipped)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    plot_sample(ax=ax, sample=curve, color='red', zorder=10, point_size=100)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve), color='red', zorder=10, point_size=100)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve_flipped), color='red', zorder=10, point_size=100)
    plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve2), color='red', zorder=10, point_size=100)
    # plt.show()
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve3), color='red', zorder=10, point_size=100)
    # plt.show()
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve4), color='red', zorder=10, point_size=100)
    # plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve_flipped), color='red', zorder=10, point_size=100)
    # plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve1_flipped), color='red', zorder=10, point_size=100)
    # plt.show()
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve2_flipped), color='red', zorder=10, point_size=100)
    # plt.show()
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve3_flipped), color='red', zorder=10, point_size=100)
    # plt.show()
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # plot_sample(ax=ax, sample=curve_processing.normalize_curve2(curve=curve4_flipped), color='red', zorder=10, point_size=100)
    # plt.show()
