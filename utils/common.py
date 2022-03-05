# python peripherals
import os
from pathlib import Path
import multiprocessing

# numpy
import numpy

# common
from utils import settings

# pytorch
import torch

# sklearn

# scipy

# deep signature
from deep_signature.nn.networks import DeepSignatureArcLengthNet
from deep_signature.nn.networks import DeepSignatureCurvatureNet


def create_data_generator(dir_path, fine=False, limit=None):
    npy_file_paths = []
    base_dir_path = os.path.normpath(dir_path)
    for sub_dir_path, _, file_names in os.walk(base_dir_path):
        for file_name in file_names:
            npy_file_path = os.path.normpath(os.path.join(sub_dir_path, file_name))
            npy_file_paths.append(npy_file_path)

    if fine is False:
        for npy_file_path in npy_file_paths[:limit]:
            data_objects = numpy.load(file=npy_file_path, allow_pickle=True)
            yield data_objects
    else:
        i = 0
        for npy_file_path in npy_file_paths:
            data_objects = numpy.load(file=npy_file_path, allow_pickle=True)
            for data_object in data_objects:
                if i == limit:
                    return

                i += 1
                yield data_object


def par_proc(map_func, reduce_func, iterable, label, pool=None, chunksize=None):
    if pool is None:
        print('    - Creating pool... ', end='')
        pool = multiprocessing.Pool()
        print('Done.')

    if chunksize is None:
        chunksize = int(len(iterable) / multiprocessing.cpu_count())

    iterable_length = len(iterable)
    format_string = '\r    - Generating {0}... {1:.1%} Done.'

    print(f'    - Generating {label}...', end='')
    for i, processed_data in enumerate(pool.imap_unordered(func=map_func, iterable=iterable, chunksize=chunksize)):
        reduce_func(processed_data)
        print(format_string.format(label, (i + 1) / iterable_length), end='')
    print()


def insert_sorted(a, b):
    ii = numpy.searchsorted(a, b)
    return numpy.unique(numpy.insert(a, ii, b))


# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(input_list, chunks_count):
    return [input_list[i:i + chunks_count] for i in range(0, len(input_list), chunks_count)]


def list_subdirectories(base_dir='.'):
    result = []
    for current_sub_dir in os.listdir(base_dir):
        full_sub_dir_path = os.path.join(base_dir, current_sub_dir)
        if os.path.isdir(full_sub_dir_path):
            result.append(full_sub_dir_path)

    return result


def get_latest_subdirectory(base_dir='.'):
    subdirectories = list_subdirectories(base_dir)
    return os.path.normpath(max(subdirectories, key=os.path.getmtime))

def get_tuplets_dir_path(transform_type):
    if transform_type == 'euclidean':
        level_curves_curvature_tuplets_dir_path = settings.level_curves_euclidean_curvature_tuplets_dir_path
    elif transform_type == 'similarity':
        level_curves_curvature_tuplets_dir_path = settings.level_curves_similarity_curvature_tuplets_dir_path
    elif transform_type == 'equiaffine':
        level_curves_curvature_tuplets_dir_path = settings.level_curves_equiaffine_curvature_tuplets_dir_path
    elif transform_type == 'affine':
        level_curves_curvature_tuplets_dir_path = settings.level_curves_affine_curvature_tuplets_dir_path

    if transform_type == 'euclidean':
        level_curves_arclength_tuplets_dir_path = settings.level_curves_euclidean_arclength_tuplets_dir_path
    elif transform_type == 'similarity':
        level_curves_arclength_tuplets_dir_path = settings.level_curves_similarity_arclength_tuplets_dir_path
    elif transform_type == 'equiaffine':
        level_curves_arclength_tuplets_dir_path = settings.level_curves_equiaffine_arclength_tuplets_dir_path
    elif transform_type == 'affine':
        level_curves_arclength_tuplets_dir_path = settings.level_curves_affine_arclength_tuplets_dir_path

    return level_curves_curvature_tuplets_dir_path, level_curves_arclength_tuplets_dir_path

def get_results_dir_path(transform_type):
    if transform_type == 'euclidean':
        level_curves_arclength_tuplets_results_dir_path = settings.level_curves_euclidean_arclength_tuplets_results_dir_path
    elif transform_type == 'similarity':
        level_curves_arclength_tuplets_results_dir_path = settings.level_curves_similarity_arclength_tuplets_results_dir_path
    elif transform_type == 'equiaffine':
        level_curves_arclength_tuplets_results_dir_path = settings.level_curves_equiaffine_arclength_tuplets_results_dir_path
    elif transform_type == 'affine':
        level_curves_arclength_tuplets_results_dir_path = settings.level_curves_affine_arclength_tuplets_results_dir_path

    if transform_type == 'euclidean':
        level_curves_curvature_tuplets_results_dir_path = settings.level_curves_euclidean_curvature_tuplets_results_dir_path
    elif transform_type == 'similarity':
        level_curves_curvature_tuplets_results_dir_path = settings.level_curves_similarity_curvature_tuplets_results_dir_path
    elif transform_type == 'equiaffine':
        level_curves_curvature_tuplets_results_dir_path = settings.level_curves_equiaffine_curvature_tuplets_results_dir_path
    elif transform_type == 'affine':
        level_curves_curvature_tuplets_results_dir_path = settings.level_curves_affine_curvature_tuplets_results_dir_path

    return level_curves_curvature_tuplets_results_dir_path, level_curves_arclength_tuplets_results_dir_path

def load_models(transform_type, device='cuda'):
    level_curves_curvature_tuplets_results_dir_path, level_curves_arclength_tuplets_results_dir_path = get_results_dir_path(transform_type=transform_type)

    # if we're in the equiaffine case, snap 'step' to the closest mutiple of 3 (from above)
    # if transform_type == "equiaffine":
    #     step = int(3 * numpy.ceil(step / 3))

    # package settings
    torch.set_default_dtype(torch.float64)

    # create models
    curvature_model = DeepSignatureCurvatureNet(sample_points=settings.curvature_default_sample_points_count).to(device)
    arclength_model = DeepSignatureArcLengthNet(sample_points=settings.arclength_default_supporting_points_count, transformation_group_type=transform_type).to(device)

    # load curvature model state
    latest_subdir = get_latest_subdirectory(level_curves_curvature_tuplets_results_dir_path)
    results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    curvature_model.load_state_dict(torch.load(f"{latest_subdir}/{Path(results['model_file_path']).name}", map_location=device))
    curvature_model.eval()

    # load arclength model state
    latest_subdir = get_latest_subdirectory(level_curves_arclength_tuplets_results_dir_path)
    results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    arclength_model.load_state_dict(torch.load(f"{latest_subdir}/{Path(results['model_file_path']).name}", map_location=device))
    arclength_model.eval()

    return curvature_model, arclength_model