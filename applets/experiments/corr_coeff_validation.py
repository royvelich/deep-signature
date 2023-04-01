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
    planar_curves_manager_train_smooth = PlanarCurvesManager(curves_file_paths=[Path("C:/deep-signature-data/curves/train/curves.npy")])
    planar_curves_manager_val_smooth = PlanarCurvesManager(curves_file_paths=[Path("C:/deep-signature-data/curves/validation/curves.npy")])
    planar_curves_manager_test_smooth = PlanarCurvesManager(curves_file_paths=[Path("C:/deep-signature-data/curves/test/curves.npy")])

    point_size = 30
    label_size = 100
    axis_title_size = 120
    line_width = 3

    for i in range(7):
        planar_curve = planar_curves_manager_train_smooth.get_random_planar_curve()

        fig, axes = matplotlib.pyplot.subplots(nrows=4, ncols=1, figsize=(30, 60))
        # fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(20, 20))
        planar_curve.plot_scattered_curve(ax=axes[0], label_size=label_size, cmap='hsv', color=None, point_size=point_size)
        # fig.savefig(f"C:/Users/Roy/OneDrive - Technion/Thesis/SSVM/pearson_plots/curve{i}.svg")
        # matplotlib.pyplot.close(fig)

        # fig, axes = matplotlib.pyplot.subplots(nrows=3, ncols=1, figsize=(20, 40))
        planar_curve.plot_euclidean_signature(ax=axes[1:], label_size=label_size, x_axis_title_size=axis_title_size, y_axis_title_size=axis_title_size, point_size=point_size, line_width=line_width)
        fig.savefig(f"C:/Users/Roy/OneDrive - Technion/Thesis/SSVM/pearson_plots5/signature{i}.svg")
        matplotlib.pyplot.close(fig)

    # signatures = []
    # points_count = []
    #
    # planar_curves_managers = [planar_curves_manager_train_smooth, planar_curves_manager_val_smooth, planar_curves_manager_test_smooth]
    #
    # for planar_curves_manager in planar_curves_managers:
    #     curves_count = planar_curves_manager.planar_curves_count
    #     for i in range(curves_count):
    #         curve = planar_curves_manager.planar_curves[i]
    #         signatures.append(curve.approximate_euclidean_signature(order=2))
    #         points_count.append(curve.points_count)
    #
    # signatures = numpy.concatenate(signatures)
    # numpy.random.shuffle(signatures)
    # numpy.save(file='C:/deep-signature-data-new/notebooks_output/signatures.npy', arr=signatures, allow_pickle=True)

    # signatures = numpy.load(file='C:/deep-signature-data-new/notebooks_output/signatures.npy', allow_pickle=True)

    # corrs = []
    # for _ in range(10):
    #     samples_counts = [50000]
    #     for samples_count in samples_counts:
    #         indices = numpy.random.choice(signatures.shape[0], samples_count, replace=False)
    #         signatures_sampled = signatures[indices]
    #
    #         corrcoef = numpy.corrcoef([signatures_sampled[:, 0], signatures_sampled[:, 1], signatures_sampled[:, 2]])
    #         cur_corrs = []
    #         cur_corrs.append(numpy.abs(corrcoef[0, 1]))
    #         cur_corrs.append(numpy.abs(corrcoef[1, 2]))
    #         cur_corrs.append(numpy.abs(corrcoef[0, 2]))
    #         corrs.append(cur_corrs)
    #         print(cur_corrs)
    #
    # corrs = numpy.stack(corrs)
    # print(corrs.mean(axis=0))

    # corrs = []
    # times = 1
    # for k in samples_count:
    #     current_corrs = numpy.zeros(shape=(times, 3))
    #     for j in range(times):
    #         samples = numpy.zeros(shape=[k, 3])
    #         for i in range(k):
    #             # curve_index = numpy.random.randint(curves_count)
    #             # point_index = numpy.random.randint(points_count[curve_index])
    #             curve_index = numpy.random.randint(len(signatures))
    #             point_index = numpy.random.randint(signatures[curve_index].shape[0])
    #             signature = signatures[curve_index]
    #             samples[i] = signature[point_index]
    #
    #         corrcoef = numpy.corrcoef([samples[:, 0], samples[:, 1], samples[:, 2]])
    #         current_corrs[j, 0] = numpy.abs(corrcoef[0, 1])
    #         current_corrs[j, 1] = numpy.abs(corrcoef[0, 2])
    #         current_corrs[j, 2] = numpy.abs(corrcoef[1, 2])
    #
    #     corrs.append(numpy.mean(current_corrs, axis=0))
    #     print(numpy.mean(current_corrs, axis=0))
    #
    # data = {
    #     'samples_count': samples_count,
    #     'corr': corrs,
    # }
    #
    # df = pandas.DataFrame(data=data)
    # print(df)
    # # df.to_csv("C:/deep-signature-data-new/notebooks_output/pearson.csv")