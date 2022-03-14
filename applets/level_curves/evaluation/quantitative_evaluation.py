# python peripherals
import os
from argparse import ArgumentParser
import queue

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


def evaluate_raw_curves_signatures_worker(queue, curvature_model, arclength_model, raw_curves, transform_type, dataset_name):
    print(f'Starting - Raw Curves Evaluation - dataset-name: {dataset_name}, transform-type: {transform_type}')

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

    print(f'Finished - Raw Curves Evaluation - dataset-name: {dataset_name}, transform-type: {transform_type}')

    queue.put({
        'dataset_name': dataset_name,
        'transform_type': transform_type,
        'signatures': signatures,
    })


def compare_signatures_worker(queue, curvature_model, arclength_model, raw_curves_signatures, downsampled_curves, sampling_ratio, transform_type, dataset_name):
    print(f'Starting - Signature Comparison - dataset-name: {dataset_name}, transform-type: {transform_type}, sampling-ratio: {sampling_ratio}')

    correct = 0
    matches = []

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
        for j, signature_curve in enumerate(raw_curves_signatures):
            current_arc_length = signature_curve[-1, 0]

            arclength_ratio = current_arc_length / anchor_arc_length

            anchor_signature_curve_copy = anchor_signature_curve.copy()
            anchor_signature_curve_copy[:, 0] = anchor_signature_curve_copy[:, 0] * arclength_ratio

            shift_distances = calculate_hausdorff_distances(curve1=anchor_signature_curve_copy, curve2=signature_curve)
            distances[i, j] = numpy.min(shift_distances)

        curve_id = numpy.argmin(distances[i, :])
        matches.append((i, curve_id))
        if curve_id == i:
            correct = correct + 1
            print(f'dataset-name: {dataset_name}, transform-type: {transform_type}, sampling-ratio: {sampling_ratio}, curve #{i} correctly identified')
        else:
            print(f'dataset-name: {dataset_name}, transform-type: {transform_type}, sampling-ratio: {sampling_ratio}, curve #{i} failed to be identified')

    print(f'Finished - Signature Comparison - dataset-name: {dataset_name}, transform-type: {transform_type}, sampling-ratio: {sampling_ratio}, result: {correct} / {len(downsampled_curves)}')

    queue.put({
        'dataset_name': dataset_name,
        'sampling_ratio': sampling_ratio,
        'transform_type': transform_type,
        'correct': correct,
        'curves_count': len(downsampled_curves),
        'matches': str(matches)
    })


def save_excel_file(results, file_name):
    dataset_name_col = []
    sampling_ratio_col = []
    transform_type_col = []
    correct_col = []
    curves_count_col = []
    ratio_col = []
    matches_col = []
    for result in results:
        dataset_name_col.append(result['dataset_name'])
        sampling_ratio_col.append(result['sampling_ratio'])
        transform_type_col.append(result['transform_type'])
        correct_col.append(result['correct'])
        curves_count_col.append(result['curves_count'])
        ratio_col.append(f'{(result["correct"] / result["curves_count"]):.3f}')
        matches_col.append(result['matches'])

    d = {
        'dataset_name': dataset_name_col,
        'sampling_ratio': sampling_ratio_col,
        'transform_type': transform_type_col,
        'correct': correct_col,
        'count': curves_count_col,
        'ratio': ratio_col,
        'matches': matches_col
    }

    df = pandas.DataFrame(data=d)
    df.to_excel(file_name)


if __name__ == '__main__':
    torch_mp.set_start_method('spawn', force=True)
    parser = ArgumentParser()
    parser.add_argument("--curves_base_dir_path", dest="curves_base_dir_path", type=str)
    args = parser.parse_args()

    seed = 30
    max_cpu_count = 16
    rng = numpy.random.default_rng(seed=seed)
    numpy.random.seed(seed)

    resolution = ''
    multimodality = 25
    # sampling_ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
    # transform_types = ['affine', 'equiaffine', 'euclidean']
    # dataset_names = ['basketball', 'bats', 'bears', 'birds', 'branches', 'cartoon', 'cats', 'chickens', 'clouds', 'dogs', 'flames', 'guitars', 'hearts', 'pieces', 'profiles', 'rabbits', 'rats', 'shapes', 'shields', 'signs', 'trees', 'vegetables', 'whales']

    sampling_ratios = [0.8]
    transform_types = ['affine']
    dataset_names = ['clouds', 'profiles']

    # multimodality = 25
    # sampling_ratios = [0.5]
    # transform_types = ['euclidean']
    # dataset_names = ['clouds', 'cartoon']

    curvature_models = {}
    arclength_models = {}

    curvature_model, arclength_model = common_utils.load_models(transform_type='euclidean', device='cpu')
    curvature_model.share_memory()
    arclength_model.share_memory()
    curvature_models['euclidean'] = curvature_model
    arclength_models['euclidean'] = arclength_model

    curvature_model, arclength_model = common_utils.load_models(transform_type='equiaffine', device='cpu')
    curvature_model.share_memory()
    arclength_model.share_memory()
    curvature_models['equiaffine'] = curvature_model
    arclength_models['equiaffine'] = arclength_model

    curvature_model, arclength_model = common_utils.load_models(transform_type='affine', device='cpu')
    curvature_model.share_memory()
    arclength_model.share_memory()
    curvature_models['affine'] = curvature_model
    arclength_models['affine'] = arclength_model

    raw_curves_queues = []
    processes = []
    raw_curves_configs = []
    for transform_type in transform_types:
        curves_dir_path = os.path.normpath(os.path.join(args.curves_base_dir_path, transform_type, f'multimodality_{multimodality}'))
        for dataset_name in dataset_names:
            if len(processes) >= max_cpu_count:
                stop = False
                while not stop:
                    for i, raw_curves_queue in enumerate(raw_curves_queues):
                        try:
                            raw_curves_configs.append(raw_curves_queues[i].get_nowait())
                            processes[i].join()
                            del raw_curves_queues[i]
                            del processes[i]
                            stop = True
                            break
                        except queue.Empty:
                            continue

            raw_curves_queue = torch_mp.Queue()
            raw_curves_queues.append(raw_curves_queue)
            raw_curves_path = os.path.normpath(os.path.join(curves_dir_path, f'{dataset_name}{resolution}_1.npy'))
            raw_curves = numpy.load(raw_curves_path, allow_pickle=True)
            p = torch_mp.Process(target=evaluate_raw_curves_signatures_worker, args=(raw_curves_queue, curvature_models[transform_type], arclength_models[transform_type], raw_curves, transform_type, dataset_name,))
            processes.append(p)
            p.start()

    for raw_curves_queue in raw_curves_queues:
        raw_curves_configs.append(raw_curves_queue.get())

    for p in processes:
        p.join()

    processes = []
    results = []
    signature_comparison_queues = []
    for raw_curves_config in raw_curves_configs:
        transform_type = raw_curves_config['transform_type']
        dataset_name = raw_curves_config['dataset_name']
        signatures = raw_curves_config['signatures']
        for sampling_ratio in sampling_ratios:
            if len(processes) >= max_cpu_count:
                stop = False
                while not stop:
                    for i, signature_comparison_queue in enumerate(signature_comparison_queues):
                        try:
                            results.append(signature_comparison_queues[i].get_nowait())
                            processes[i].join()
                            del signature_comparison_queues[i]
                            del processes[i]
                            stop = True
                            save_excel_file(results=results, file_name='output.xlsx')
                            break
                        except queue.Empty:
                            continue

            signature_comparison_queue = torch_mp.Queue()
            signature_comparison_queues.append(signature_comparison_queue)
            curves_dir_path = os.path.normpath(os.path.join(args.curves_base_dir_path, transform_type, f'multimodality_{multimodality}'))
            downsampled_curves_path = os.path.normpath(os.path.join(curves_dir_path, f'{dataset_name}{resolution}_{str(sampling_ratio).replace(".", "_")}.npy'))
            downsampled_curves = numpy.load(downsampled_curves_path, allow_pickle=True)
            p = torch_mp.Process(target=compare_signatures_worker, args=(signature_comparison_queue, curvature_models[transform_type], arclength_models[transform_type], signatures, downsampled_curves, sampling_ratio, transform_type, dataset_name,))
            processes.append(p)
            p.start()

    for signature_comparison_queue in signature_comparison_queues:
        results.append(signature_comparison_queue.get())

    for p in processes:
        p.join()

    save_excel_file(results=results, file_name='output.xlsx')
