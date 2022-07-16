# python peripherals
import os
import numpy
import random
import queue
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from utils import common as common_utils

# torch
import torch
from torch.utils.data import Dataset

# deep_signature
from deep_signature.data_generation import curve_generation
from deep_signature.data_generation import dataset_generation
from deep_signature.data_manipulation import curve_processing
from deep_signature.data_manipulation import curve_sampling, curve_processing
from deep_signature.stats import discrete_distribution
from deep_signature.linalg import transformations


class DeepSignatureTupletsDataset(Dataset):
    def __init__(self, dataset_size, dir_path, replace, buffer_size, num_workers, args):
        self._online = True
        self._curves = curve_generation.CurvesGenerator.load_curves(dir_path)
        for i, _ in enumerate(self._curves):
            self._curves[i] = self._curves[i].astype('float32')
        self._dataset_size = dataset_size
        self._replace = replace
        self._buffer_size = buffer_size
        self._num_workers = num_workers
        self._q = Queue(maxsize=dataset_size)
        self._args = [self._curves, self._q, args]
        self._tuplets = []

    def __len__(self):
        if self._online is True:
            return self._dataset_size
        else:
            return len(self._tuplets)

    def __getitem__(self, index):
        if self._online is True:
            mod_index = numpy.mod(index, self._buffer_size)
            tuplet = self._tuplets[mod_index]

            if self._replace is True:
                try:
                    new_tuplet = self._q.get_nowait()
                    rand_index = int(numpy.random.randint(self._buffer_size, size=1))
                    self._tuplets[rand_index] = new_tuplet
                except queue.Empty:
                    pass
        else:
            tuplet = self._tuplets[index]

        return numpy.array(tuplet)

    def start(self):
        self._workers = [Process(target=self._map_func, args=self._args) for i in range(self._num_workers)]

        print('')
        for i, worker in enumerate(self._workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {self._num_workers}', end='')

        print(f'\nItem {len(self._tuplets)} / {self._buffer_size}', end='')
        if len(self._tuplets) == 0:
            while True:
                if self._q.empty() is False:
                    self._tuplets.append(self._q.get())
                    print(f'\rItem {len(self._tuplets)} / {self._buffer_size}', end='')
                    if len(self._tuplets) == self._buffer_size:
                        break
        print('')

    def stop(self):
        for i, worker in enumerate(self._workers):
            worker.terminate()

    def save(self, dataset_dir_path):
        Path(dataset_dir_path).mkdir(parents=True, exist_ok=True)
        numpy.save(file=os.path.join(dataset_dir_path, 'tuplets.npy'), arr=self._tuplets, allow_pickle=True)
        common_utils.save_object_dict(obj=self, file_path=os.path.join(dataset_dir_path, 'dataset_settings.txt'))

    def load(self, dataset_dir_path):
        self._tuplets = numpy.load(file=os.path.join(dataset_dir_path, 'tuplets.npy'), allow_pickle=True)
        # self._tuplets = [self._tuplets[i, :, :, :] for i in range(self._tuplets.shape[0])]
        # j = 6

    @classmethod
    def _map_func(cls, curves, q, args):
        while True:
            tuplet = cls._generate_tuplet(curves=curves, args=args)
            q.put(tuplet)


class ArclengthTupletsDataset(DeepSignatureTupletsDataset):
    def __init__(self, dataset_size, dir_path, replace, buffer_size, num_workers, args):
        DeepSignatureTupletsDataset.__init__(
            self,
            dataset_size=dataset_size,
            dir_path=dir_path,
            replace=replace,
            buffer_size=buffer_size,
            num_workers=num_workers,
            args=args)

    @classmethod
    def _generate_tuplet(cls, curves, args):
        h = 5


class CurvatureTupletsDataset(DeepSignatureTupletsDataset):
    def __init__(self, dataset_size, dir_path, replace, buffer_size, num_workers, args):
        DeepSignatureTupletsDataset.__init__(
            self,
            dataset_size=dataset_size,
            dir_path=dir_path,
            replace=replace,
            buffer_size=buffer_size,
            num_workers=num_workers,
            args=args)

    @classmethod
    def _generate_tuplet(cls, curves, args):
        tuplet = []
        dist_index = 0
        curve_index = int(numpy.random.randint(curves.shape[0], size=1))
        curve = curve_processing.center_curve(curve=curves[curve_index])
        curve_points_count = curve.shape[0]
        sampling_points_count = int(args.sampling_ratio * curve_points_count)
        discrete_distribution_pack = discrete_distribution.random_discrete_dist(bins=curve_points_count, multimodality=args.multimodality, max_density=1, count=args.negative_examples_count+2)
        center_point_index = int(numpy.random.randint(curve.shape[0], size=1))
        for i in range(2):
            transform = transformations.generate_random_transform_2d_training(transform_type=args.group)
            transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)

            indices_pool = discrete_distribution.sample_discrete_dist(dist=discrete_distribution_pack[dist_index], sampling_points_count=sampling_points_count)
            sample = curve_sampling.sample_curve_neighborhood(
                curve=transformed_curve,
                center_point_index=center_point_index,
                indices_pool=indices_pool,
                supporting_points_count=args.supporting_points_count)

            sample = curve_processing.normalize_curve(curve=sample)
            tuplet.append(sample)
            dist_index = dist_index + 1

        flipped_anchor = numpy.flip(m=tuplet[0], axis=0).copy()
        sample = curve_processing.normalize_curve(curve=flipped_anchor)
        tuplet.append(sample)

        for i in range(args.negative_examples_count):
            while True:
                center_point_index_offset = int(numpy.random.randint(args.offset_length, size=1)) - int(args.offset_length/2)
                if center_point_index_offset != 0:
                    break

            transform = transformations.generate_random_transform_2d_training(transform_type=args.group)
            transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)

            negative_center_point_index = numpy.mod(center_point_index + center_point_index_offset, transformed_curve.shape[0])
            # negative_center_point_index = int(numpy.random.randint(transformed_curve.shape[0], size=1))
            indices_pool = discrete_distribution.sample_discrete_dist(dist=discrete_distribution_pack[dist_index], sampling_points_count=sampling_points_count)
            sample = curve_sampling.sample_curve_neighborhood(
                curve=transformed_curve,
                center_point_index=negative_center_point_index,
                indices_pool=indices_pool,
                supporting_points_count=args.supporting_points_count)

            sample = curve_processing.normalize_curve(curve=sample)
            tuplet.append(sample)
            dist_index = dist_index + 1

        return tuplet


class DifferentialInvariantsTupletsDataset(DeepSignatureTupletsDataset):
    def __init__(self, dataset_size, dir_path, replace, buffer_size, num_workers, args):
        DeepSignatureTupletsDataset.__init__(
            self,
            dataset_size=dataset_size,
            dir_path=dir_path,
            replace=replace,
            buffer_size=buffer_size,
            num_workers=num_workers,
            args=args)

    @classmethod
    def _generate_tuplet(cls, curves, args):
        tuplet = []
        dist_index = 0
        curve_index = int(numpy.random.randint(curves.shape[0], size=1))
        curve = curve_processing.center_curve(curve=curves[curve_index])
        curve_points_count = curve.shape[0]
        sampling_points_count = int(args.sampling_ratio * curve_points_count)
        discrete_distribution_pack = discrete_distribution.random_discrete_dist(bins=curve_points_count, multimodality=args.multimodality, max_density=1, count=2)
        center_point_index = int(numpy.random.randint(curve.shape[0], size=1))
        for i in range(2):
            transform = transformations.generate_random_transform_2d_training(transform_type=args.group)
            if i == 0:
                transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
            else:
                transformed_curve = curve

            indices_pool = discrete_distribution.sample_discrete_dist(dist=discrete_distribution_pack[dist_index], sampling_points_count=sampling_points_count)
            sample = curve_sampling.sample_curve_neighborhood(
                curve=transformed_curve,
                center_point_index=center_point_index,
                indices_pool=indices_pool,
                supporting_points_count=args.supporting_points_count)

            sample = curve_processing.normalize_curve(curve=sample)
            tuplet.append(sample)
            dist_index = dist_index + 1

        # flipped_anchor = numpy.flip(m=tuplet[0], axis=0).copy()
        # sample = curve_processing.normalize_curve(curve=flipped_anchor)
        # tuplet.append(sample)
        #
        # for i in range(args.negative_examples_count):
        #     while True:
        #         center_point_index_offset = int(numpy.random.randint(args.offset_length, size=1)) - int(args.offset_length/2)
        #         if center_point_index_offset != 0:
        #             break
        #
        #     transform = transformations.generate_random_transform_2d_training(transform_type=args.group)
        #     transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
        #
        #     negative_center_point_index = numpy.mod(center_point_index + center_point_index_offset, transformed_curve.shape[0])
        #     # negative_center_point_index = int(numpy.random.randint(transformed_curve.shape[0], size=1))
        #     indices_pool = discrete_distribution.sample_discrete_dist(dist=discrete_distribution_pack[dist_index], sampling_points_count=sampling_points_count)
        #     sample = curve_sampling.sample_curve_neighborhood(
        #         curve=transformed_curve,
        #         center_point_index=negative_center_point_index,
        #         indices_pool=indices_pool,
        #         supporting_points_count=args.supporting_points_count)
        #
        #     sample = curve_processing.normalize_curve(curve=sample)
        #     tuplet.append(sample)
        #     dist_index = dist_index + 1

        return tuplet