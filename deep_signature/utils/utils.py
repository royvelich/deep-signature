# python peripherals
import os
import multiprocessing

# numpy
import numpy


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


def par_proc(map_func, reduce_func, iterable, label, chunksize=None):
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
