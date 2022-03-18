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
import torch.distributed as dist

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


def get_dataset_dir(data_dir, invariant, group, purpose):
    return f'{data_dir}/datasets/{invariant}/{group}/{purpose}'


def get_train_dataset_dir(data_dir, invariant, group):
    return get_dataset_dir(data_dir=data_dir, invariant=invariant, group=group, purpose='train')


def get_validation_dataset_dir(data_dir, invariant, group):
    return get_dataset_dir(data_dir=data_dir, invariant=invariant, group=group, purpose='validation')


def get_results_dir(data_dir, invariant, group):
    return f'{data_dir}/results/{invariant}/{group}'


def get_curves_dir(data_dir, purpose):
    return f'{data_dir}/curves/{purpose}'


def get_train_curves_dir(data_dir):
    return get_curves_dir(data_dir=data_dir, purpose='train')


def get_validation_curves_dir(data_dir):
    return get_curves_dir(data_dir=data_dir, purpose='validation')


def get_test_curves_dir(data_dir):
    return get_curves_dir(data_dir=data_dir, purpose='test')


def get_images_dir(data_dir, purpose):
    return f'{data_dir}/images/{purpose}'


def get_train_images_dir(data_dir):
    return get_images_dir(data_dir=data_dir, purpose='train')


def get_validation_images_dir(data_dir):
    return get_images_dir(data_dir=data_dir, purpose='validation')


def get_test_images_dir(data_dir):
    return get_images_dir(data_dir=data_dir, purpose='test')


def load_models(group, device=torch.device('cuda')):
    curvature_results_dir_path = get_results_dir(invariant='curvature', group=group)
    arclength_results_dir_path = get_results_dir(invariant='arclength', group=group)

    # create models
    curvature_model = DeepSignatureCurvatureNet(sample_points=settings.curvature_default_sample_points_count).cuda()

    dist.init_process_group(backend='gloo', init_method='env://', world_size=1, rank=0)
    arclength_model = DeepSignatureArcLengthNet(sample_points=settings.arclength_default_supporting_points_count).cuda()
    arclength_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(arclength_model)
    arclength_model = torch.nn.parallel.DistributedDataParallel(arclength_model)

    # load curvature model state
    latest_subdir = get_latest_subdirectory(curvature_results_dir_path)
    results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    curvature_model.load_state_dict(torch.load(f"{latest_subdir}/{Path(results['model_file_path']).name}", map_location=device))
    curvature_model.eval()

    # load arclength model state
    latest_subdir = get_latest_subdirectory(arclength_results_dir_path)
    results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    arclength_model.load_state_dict(torch.load(f"{latest_subdir}/{Path(results['model_file_path']).name}", map_location=device))
    arclength_model.eval()

    return curvature_model, arclength_model


def get_arclength_default_min_offset(supporting_points_count):
    return supporting_points_count - 1


def get_arclength_default_max_offset(supporting_points_count):
    return 3 * get_arclength_default_min_offset(supporting_points_count=supporting_points_count)


def save_object_dict(obj, file_path):
    dict = {key: value for key, value in obj.__dict__.items() if isinstance(value, str) or isinstance(value, int) or isinstance(value, float)}
    with open(file_path, "w") as text_file:
        for key, value in dict.items():
            text_file.write(f'{key}: {value}\n')
