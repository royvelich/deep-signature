# python peripherals
import sys
from pathlib import Path
from typing import Callable, List
sys.path.append('../../.')

# numpy
import numpy

# torch
import torch

# matplotlib
import matplotlib
import matplotlib.pyplot
import matplotlib.axes
matplotlib.pyplot.rcParams['text.usetex'] = True

# deep-signature
from deep_signature.core.base import SeedableObject
from deep_signature.manifolds.planar_curves.implementation import PlanarCurvesManager, PlanarCurve

# pandas
import pandas


# SeedableObject.set_seed(seed=42)


if __name__ == '__main__':
    # planar_curves_manager_smooth = PlanarCurvesManager(curves_file_paths=[Path("C:/deep-signature-data-new/curves/train/2022-12-31-15-12-51/curves.npy")])
    # planar_curves_manager_train_smooth = PlanarCurvesManager(curves_file_paths=[Path("C:/deep-signature-data/curves/train/curves.npy")])
    # planar_curves_manager_val_smooth = PlanarCurvesManager(curves_file_paths=[Path("C:/deep-signature-data/curves/validation/curves.npy")])
    # planar_curves_manager_test_smooth = PlanarCurvesManager(curves_file_paths=[Path("C:/deep-signature-data/curves/test/curves.npy")])
    # planar_curves_manager_corr_exp_smooth = PlanarCurvesManager(curves_file_paths=[Path("C:/deep-signature-data-new/curves/corr_experiment/2023-04-02-12-25-24/curves.npy")])

    # point_size = 30
    # label_size = 50
    # axis_title_size = 60
    # line_width = 3
    # order = 6
    # for i in range(20):
    #
    #     planar_curve = planar_curves_manager_val_smooth.planar_curves[i]
    #     # planar_curve = planar_curve.rotate_curve(radians=1.5)
    #
    #     dk_ds_plots_count = order + 1
    #     signature_plots_count = int((dk_ds_plots_count*dk_ds_plots_count - dk_ds_plots_count) / 2)
    #
    #     fig, axes = matplotlib.pyplot.subplots(nrows=(1 + dk_ds_plots_count + signature_plots_count), ncols=1, figsize=(50, 250))
    #     for ax in axes:
    #         ax.set_facecolor('#AAAAAA')
    #
    #     # fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(20, 20))
    #     planar_curve.plot_scattered_curve(ax=axes[0], label_size=label_size, cmap='hsv', color=None, point_size=point_size)
    #     # fig.savefig(f"C:/Users/Roy/OneDrive - Technion/Thesis/SSVM/pearson_plots/curve{i}.svg")
    #     # matplotlib.pyplot.close(fig)
    #     # fig.tight_layout()
    #     # fig, axes = matplotlib.pyplot.subplots(nrows=3, ncols=1, figsize=(20, 40))
    #     planar_curve.plot_euclidean_signature(ax=axes[1:], label_size=label_size, x_axis_title_size=axis_title_size, y_axis_title_size=axis_title_size, point_size=point_size, line_width=line_width, order=order)
    #     fig.tight_layout()
    #     # matplotlib.pyplot.subplots_adjust(top=0.85)
    #     # fig.tight_layout()
    #
    #     fig.savefig(f"C:/Users/Roy/OneDrive - Technion/Thesis/SSVM/pearson_plots5/signature{i}.png")
    #
    #     matplotlib.pyplot.close(fig)


    # signatures = []
    # points_count = []
    # # planar_curves_managers = [planar_curves_manager_corr_exp_smooth]
    # planar_curves_managers = [planar_curves_manager_train_smooth, planar_curves_manager_val_smooth, planar_curves_manager_test_smooth]
    # for planar_curves_manager in planar_curves_managers:
    #     curves_count = planar_curves_manager.planar_curves_count
    #     for i in range(curves_count):
    #         curve = planar_curves_manager.planar_curves[i]
    #         signatures.append(curve.approximate_euclidean_signature(order=order))
    #         points_count.append(curve.points_count)
    #
    # signatures = numpy.concatenate(signatures)
    # numpy.random.shuffle(signatures)
    # numpy.save(file='C:/deep-signature-data-new/notebooks_output/signatures_old.npy', arr=signatures, allow_pickle=True)

    bla = numpy.array([1,2,3])
    bla2 = numpy.sum(bla * bla)
    j = 6


    signatures = numpy.load(file='C:/deep-signature-data-new/notebooks_output/signatures_old.npy', allow_pickle=True)
    # print(signatures.shape[0])
    # signatures = signatures[~numpy.isnan(signatures).any(axis=1)]
    # print(signatures.shape[0])
    # corrcoef = numpy.corrcoef([signatures[:, 0], signatures[:, 1], signatures[:, 2], signatures[:, 3], signatures[:, 4], signatures[:, 5], signatures[:, 6]])
    # cur_corrs = []
    # print(corrcoef)

    def approximate_correlation(diff_inv: numpy.ndarray, diff_inv_s: numpy.ndarray, diff_inv_ss: numpy.ndarray, subtract_mean: bool = True) -> float:
        j = 5
        if subtract_mean is True:
            diff_inv = diff_inv - numpy.mean(diff_inv)
            diff_inv_s = diff_inv_s - numpy.mean(diff_inv_s)
            diff_inv_ss = diff_inv_ss - numpy.mean(diff_inv_ss)

        return -numpy.sum(diff_inv_s * diff_inv_s) / numpy.sqrt(numpy.sum(diff_inv * diff_inv) * numpy.sum(diff_inv_ss * diff_inv_ss))

    for i in range(5):
        print(f'approximated corr for (k{i}, k{i+2}) = {approximate_correlation(diff_inv=signatures[:, i], diff_inv_s=signatures[:, i+1], diff_inv_ss=signatures[:, i+2], subtract_mean=True)}')

    for i in range(5):
        print(f'approximated corr for (k{i}, k{i+2}) = {approximate_correlation(diff_inv=signatures[:, i], diff_inv_s=signatures[:, i+1], diff_inv_ss=signatures[:, i+2], subtract_mean=False)}')