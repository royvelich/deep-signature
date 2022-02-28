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
from deep_signature.data_manipulation import curve_processing

# utils
from utils import common as common_utils
from utils import evaluation as evaluation_utils
from utils import plot as plot_utils
from utils import settings


if __name__ == '__main__':
    transform_type = 'affine'
    import warnings

    warnings.filterwarnings("ignore")

    # constants
    true_arclength_colors = ['#FF8C00', '#444444']
    predicted_arclength_colors = ['#AA0000', '#0000AA']
    sample_colors = ['#FF0000', '#0000FF']
    curve_colors = ['#FF0000', '#0000FF', '#FF9D11']
    limit = 4
    factor_extraction_limit = -1
    comparison_curves_count = 2
    sampling_ratio = 0.7
    anchors_ratio = None

    # randomness
    # numpy.random.seed(30)

    # models
    curvature_model, arclength_model = common_utils.load_models(transform_type=transform_type)

    # curves
    curves_full = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_test)

    # curves_full = LevelCurvesGenerator.load_curves(dir_path=os.path.normpath("C:/GitHub/deep-signature/applets/level_curves/evaluation/"))
    curves = []
    for curve in curves_full:
        if 1000 < curve.shape[0] < 1200:
            # curve = curve_processing.smooth_curve(
            #     curve=curve,
            #     iterations=1,
            #     window_length=33,
            #     poly_order=2)
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