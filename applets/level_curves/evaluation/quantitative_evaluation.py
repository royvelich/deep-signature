# python peripherals
import os
from argparse import ArgumentParser
from multiprocessing import Process, Queue, cpu_count
import multiprocessing

# numpy
import numpy

# skimage
import skimage.io
import skimage.color
import skimage.measure
from skimage import metrics

# pandas
import bottleneck
import numexpr
import pandas

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

# pytorch
import torch.multiprocessing as torch_mp


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


def calculate_signature_curve(curve, transform_type, sampling_ratio, curvature_model, arclength_model, rng=None, plot=False, transform_curve=True):
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
        neighborhood_supporting_points_count=settings.curvature_default_supporting_points_count,
        section_supporting_points_count=settings.arclength_default_supporting_points_count,
        indices_shift=indices_shift,
        device='cpu',
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


def evaluate_signatures_worker(queue, curvature_model, arclength_model, raw_curves, downsampled_curves_path, sampling_ratio, transform_type, dataset_name):
    print(f'Starting - dataset-name: {dataset_name}, transform-type: {transform_type}, sampling-ratio: {sampling_ratio}')

    downsampled_curves = numpy.load(downsampled_curves_path, allow_pickle=True)

    correct = 0
    signatures = []
    for i, curve in enumerate(raw_curves):
        signature_curve = calculate_signature_curve(
            curve=curve,
            transform_type=transform_type,
            sampling_ratio=1,
            curvature_model=curvature_model,
            arclength_model=arclength_model,
            transform_curve=False)

        signatures.append(signature_curve)

    distances = numpy.zeros((len(downsampled_curves), len(downsampled_curves)))
    for i, curve in enumerate(downsampled_curves):
        anchor_signature_curve = calculate_signature_curve(
            curve=curve,
            transform_type=transform_type,
            sampling_ratio=1,
            curvature_model=curvature_model,
            arclength_model=arclength_model,
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
            print(f'dataset-name: {dataset_name}, transform-type: {transform_type}, sampling-ratio: {sampling_ratio}, curve #{i} correctly identified')
        else:
            print(f'dataset-name: {dataset_name}, transform-type: {transform_type}, sampling-ratio: {sampling_ratio}, curve #{i} failed to be identified')

    print(f'Finished - dataset-name: {dataset_name}, transform-type: {transform_type}, sampling-ratio: {sampling_ratio}, result: {correct} / {len(downsampled_curves)}')

    queue.put({
        'dataset_name': dataset_name,
        'sampling_ratio': sampling_ratio,
        'transform_type': transform_type,
        'correct': correct,
        'curves_count': len(downsampled_curves)
    })


if __name__ == '__main__':
    torch_mp.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument("--curves_base_dir_path", dest="curves_base_dir_path", type=str)
    args = parser.parse_args()

    seed = 30
    rng = numpy.random.default_rng(seed=seed)
    numpy.random.seed(seed)

    multimodality = 25
    sampling_ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    transform_types = ['euclidean', 'equiaffine', 'affine']
    dataset_names = ['animals', 'basketball', 'bats', 'bears', 'birds', 'branches', 'butterflies', 'cartoon', 'cats', 'chickens', 'clouds', 'dogs', 'flames', 'guitars', 'hearts', 'insects', 'leaves', 'pieces', 'profiles', 'rabbits', 'rats', 'shapes', 'shields', 'signs', 'trees', 'vegetables', 'whales']

    dataset_name_col = []
    sampling_ratio_col = []
    transform_type_col = []
    correct_col = []
    curves_count_col = []
    ratio_col = []

    queue = torch_mp.Queue()
    processes = []
    for transform_type in transform_types:
        curvature_model, arclength_model = common_utils.load_models(transform_type=transform_type, device='cpu')
        curvature_model.share_memory()
        arclength_model.share_memory()
        curves_dir_path = os.path.normpath(os.path.join(args.curves_base_dir_path, transform_type, f'multimodality_{multimodality}'))
        for dataset_name in dataset_names:
            raw_curves_path = os.path.normpath(os.path.join(curves_dir_path, f'{dataset_name}_1.npy'))
            raw_curves = numpy.load(raw_curves_path, allow_pickle=True)
            for sampling_ratio in sampling_ratios:
                downsampled_curves_path = os.path.normpath(os.path.join(curves_dir_path, f'{dataset_name}_{str(sampling_ratio).replace(".", "_")}.npy'))
                p = torch_mp.Process(target=evaluate_signatures_worker, args=(queue, curvature_model, arclength_model, raw_curves, downsampled_curves_path, sampling_ratio, transform_type, dataset_name,))
                processes.append(p)
                p.start()

    for p in processes:
        p.join()

    while queue.empty() is False:
        result = queue.get()
        dataset_name_col.append(result['dataset_name'])
        sampling_ratio_col.append(result['sampling_ratio'])
        transform_type_col.append(result['transform_type'])
        correct_col.append(result['correct'])
        curves_count_col.append(result['curves_count'])
        ratio_col.append(f'{(result["correct"] / result["curves_count"]):.3f}')

    d = {
        'dataset_name': dataset_name_col,
        'sampling_ratio': sampling_ratio_col,
        'transform_type': transform_type_col,
        'correct': correct_col,
        'count': curves_count_col,
        'ratio': ratio_col
    }

    df = pandas.DataFrame(data=d)
    df.to_excel("output.xlsx")
