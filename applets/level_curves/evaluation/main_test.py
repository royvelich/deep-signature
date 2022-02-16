# python peripherals
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

# numpy
import numpy

# ipython
from IPython.display import display, HTML

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# pytorch
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader

# deep signature
from deep_signature.data_generation.curve_generation import LevelCurvesGenerator

# utils
from utils import common as common_utils
from utils import evaluation as evaluation_utils
from utils import plot as plot_utils
from utils import settings


def plot_learning_curve(results_dir_path, title):
    latest_subdir = common_utils.get_latest_subdirectory(results_dir_path)
    results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()

    start_index = 0
    end_index = None
    train_loss_array = results['train_loss_array'][start_index:] if end_index is None else results['train_loss_array'][start_index:end_index]
    validation_loss_array = results['validation_loss_array'][start_index:] if end_index is None else results['validation_loss_array'][start_index:end_index]

    train_loss_array_no_nan = train_loss_array[~numpy.isnan(train_loss_array)]
    validation_loss_array_no_nan = validation_loss_array[~numpy.isnan(validation_loss_array)]

    epochs_list = numpy.array(range(len(train_loss_array)))

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)

    ax.plot(epochs_list, train_loss_array, label='Train Loss', linewidth=4.0)
    ax.plot(epochs_list, validation_loss_array, label='Validation Loss', linewidth=4.0)
    ax.set_title(title, fontsize=30)

    plt.legend(fontsize=20, title_fontsize=20)

    print(numpy.min(train_loss_array_no_nan))
    print(numpy.min(validation_loss_array_no_nan))

    plt.show()

if __name__ == '__main__':
    transform_type = 'euclidean'
    level_curves_curvature_tuplets_results_dir_path, level_curves_arclength_tuplets_results_dir_path = common_utils.get_results_dir_path(transform_type=transform_type)
    plot_learning_curve(level_curves_curvature_tuplets_results_dir_path, 'Curvature Learning Curve')
    plot_learning_curve(level_curves_arclength_tuplets_results_dir_path, 'Arc-Length Learning Curve')

    import warnings

    warnings.filterwarnings("ignore")

    # constants
    true_arclength_colors = ['#FF8C00', '#444444']
    predicted_arclength_colors = ['#AA0000', '#0000AA']
    sample_colors = ['#FF0000', '#0000FF']
    curve_colors = ['#FF0000', '#0000FF', '#FF9D11']
    limit = 2
    factor_extraction_limit = -1
    comparison_curves_count = 2
    sampling_ratio = 0.8
    anchors_ratio = 1

    # randomness
    numpy.random.seed(20)

    # models
    curvature_model, arclength_model = common_utils.load_models(transform_type=transform_type)

    # curves
    # curves_full = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_test)

    curves_full = LevelCurvesGenerator.load_curves(dir_path=os.path.normpath("C:/GitHub/deep-signature/applets/level_curves/evaluation/"))
    curves = []
    for curve in curves_full:
        if 1000 < curve.shape[0] < 1400:
            curves.append(curve)

    print(len(curves))

    numpy.random.shuffle(curves)
    curves_limited = curves[:limit]
    factor_extraction_curves = curves[factor_extraction_limit:]

    # create color map
    color_map = plt.get_cmap('rainbow', limit)

    # generate curve records
    curve_records = evaluation_utils.generate_curve_records(
        arclength_model=arclength_model,
        curvature_model=curvature_model,
        curves=curves_limited,
        factor_extraction_curves=factor_extraction_curves,
        transform_type=transform_type,
        comparison_curves_count=comparison_curves_count,
        sampling_ratio=sampling_ratio,
        anchors_ratio=anchors_ratio,
        neighborhood_supporting_points_count=settings.curvature_default_supporting_points_count,
        section_supporting_points_count=settings.arclength_default_supporting_points_count)

    plot_utils.plot_curve_comparisons(
        curve_records=curve_records,
        curve_colors=curve_colors,
        sampling_ratio=sampling_ratio,
        transformation_group_type=transform_type,
        plot_to_screen=True)