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


def calculate_signature_curve(curve, transform_type, sampling_ratio, anchors_ratio, curvature_model, arclength_model, rng=None, plot=False, transform_curve=True):
    curve = curve_processing.center_curve(curve=curve)

    if transform_curve is True:
        transform = transformations.generate_random_transform_2d_evaluation(transform_type=transform_type)

        transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
    else:
        transformed_curve = curve

    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
        plot_curve(ax=ax, curve=curve, color='red', zorder=10, linewidth=10)
        plot_curve(ax=ax, curve=transformed_curve, color='green', zorder=10, linewidth=10)
        plt.show()

    transformed_curve = curve_processing.center_curve(curve=transformed_curve)
    indices_shift = int(numpy.random.randint(transformed_curve.shape[0], size=1))
    predicted_curve_invariants = evaluation_utils.predict_curve_invariants(
        curve=transformed_curve,
        arclength_model=arclength_model,
        curvature_model=curvature_model,
        sampling_ratio=sampling_ratio,
        anchors_ratio=anchors_ratio,
        neighborhood_supporting_points_count=settings.curvature_default_supporting_points_count,
        section_supporting_points_count=settings.arclength_default_supporting_points_count,
        indices_shift=indices_shift,
        rng=rng)

    signature_curve = predicted_curve_invariants['predicted_signature']
    return signature_curve


def calculate_hausdorff_distances(curve1, curve2):
    shift_distances = []
    for shift in range(curve2.shape[0]):
        shifted_curve2 = evaluation_utils.shift_signature_curve(curve=curve2, shift=shift)
        hausdorff_distance = evaluation_utils.calculate_hausdorff_distance(curve1=curve1, curve2=shifted_curve2, distance_type='euclidean')
        shift_distances.append(hausdorff_distance)
    return numpy.array(shift_distances)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--curves_base_dir_path", dest="curves_base_dir_path", type=str)
    args = parser.parse_args()

    seed = 30
    rng = numpy.random.default_rng(seed=seed)
    numpy.random.seed(seed)
    multimodality = 25
    sampling_ratio = 0.8
    dataset_name = 'cartoon'
    anchors_ratio = None
    transform_type = 'affine'
    curvature_model, arclength_model = common_utils.load_models(transform_type=transform_type)
    curves_dir_path = os.path.normpath(os.path.join(args.curves_base_dir_path, transform_type, f'multimodality_{multimodality}'))
    downsampled_curves_path = os.path.normpath(os.path.join(curves_dir_path, f'{dataset_name}_{str(sampling_ratio).replace(".", "_")}.npy'))
    raw_curves_path = os.path.normpath(os.path.join(curves_dir_path, f'{dataset_name}_1.npy'))

    downsampled_curves = numpy.load(downsampled_curves_path, allow_pickle=True)
    raw_curves = numpy.load(raw_curves_path, allow_pickle=True)

    correct = 0
    signatures = []
    for i, curve in enumerate(raw_curves):
        signature_curve = calculate_signature_curve(
            curve=curve,
            transform_type=transform_type,
            sampling_ratio=1,
            anchors_ratio=anchors_ratio,
            curvature_model=curvature_model,
            arclength_model=arclength_model,
            rng=rng,
            transform_curve=False)

        signatures.append(signature_curve)

    distances = numpy.zeros((len(downsampled_curves), len(downsampled_curves)))
    for i, curve in enumerate(downsampled_curves):
        anchor_signature_curve = calculate_signature_curve(
            curve=curve,
            transform_type=transform_type,
            sampling_ratio=1,
            anchors_ratio=anchors_ratio,
            curvature_model=curvature_model,
            arclength_model=arclength_model,
            rng=rng,
            transform_curve=False,
            plot=False)

        anchor_arc_length = anchor_signature_curve[-1, 0]
        for j, signature_curve in enumerate(signatures):
            current_arc_length = signature_curve[-1, 0]

            arclength_ratio = current_arc_length / anchor_arc_length

            anchor_signature_curve_copy = anchor_signature_curve.copy()
            anchor_signature_curve_copy[:, 0] = anchor_signature_curve_copy[:, 0] * arclength_ratio

            shift_distances = calculate_hausdorff_distances(curve1=anchor_signature_curve_copy, curve2=signature_curve)
            distances[i, j] = numpy.min(shift_distances)

        curve_id = numpy.argmin(distances[i, :])
        if curve_id == i:
            correct = correct + 1
            print(f'curve #{i} correctly identified')
        else:
            print(f'curve #{i} failed to be identified')

    print(f'{correct} identifications out of {len(downsampled_curves)}')
