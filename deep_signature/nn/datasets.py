# python peripherals
import os
import numpy
import random
import queue
from multiprocessing import Process, Queue, cpu_count

# torch
import torch
from torch.utils.data import Dataset

# deep_signature
from deep_signature.data_generation import curve_generation
from deep_signature.data_generation import dataset_generation
from deep_signature.data_manipulation import curve_processing


class DeepSignaturePairsDataset(Dataset):
    def __init__(self):
        self._pairs = None
        self._labels = None

    def load_dataset(self, negative_pairs_dir_path, positive_pairs_dir_path):
        negative_pairs = numpy.load(file=os.path.normpath(os.path.join(negative_pairs_dir_path, 'negative_pairs.npy')), allow_pickle=True)
        positive_pairs = numpy.load(file=os.path.normpath(os.path.join(positive_pairs_dir_path, 'positive_pairs.npy')), allow_pickle=True)
        # pairs_count = numpy.minimum(negative_pairs.shape[0], positive_pairs.shape[0])
        # full_pairs_count = 2 * pairs_count
        full_pairs_count = negative_pairs.shape[0] + positive_pairs.shape[0]

        random.shuffle(negative_pairs)
        random.shuffle(positive_pairs)
        # negative_pairs = negative_pairs[:pairs_count]
        # positive_pairs = positive_pairs[:pairs_count]

        self._pairs = numpy.empty((full_pairs_count, negative_pairs.shape[1], negative_pairs.shape[2], negative_pairs.shape[3]))
        self._pairs[:negative_pairs.shape[0], :] = negative_pairs
        self._pairs[negative_pairs.shape[0]:, :] = positive_pairs
        # del negative_pairs
        # del positive_pairs

        negaitve_labels = numpy.zeros(negative_pairs.shape[0])
        positive_labels = numpy.ones(positive_pairs.shape[0])
        self._labels = numpy.empty(full_pairs_count)
        self._labels[:negative_pairs.shape[0]] = negaitve_labels
        self._labels[negative_pairs.shape[0]:] = positive_labels
        # self._labels[::2] = negaitve_labels
        # self._labels[1::2] = positive_labels

    def __len__(self):
        return self._labels.shape[0]

    def __getitem__(self, idx):
        pair = self._pairs[idx, :]

        for i in range(2):
            if not curve_processing.is_ccw(curve=pair[i]):
                pair[i] = numpy.flip(pair[i], axis=0)

        for i in range(2):
            radians = curve_processing.calculate_secant_angle(curve=pair[i])
            pair[i] = curve_processing.rotate_curve(curve=pair[i], radians=radians)

        pair_torch = torch.from_numpy(pair).cuda().double()
        label_torch = torch.from_numpy(numpy.array([self._labels[idx]])).cuda().double()

        return {
            'input': pair_torch,
            'labels': label_torch
        }


class DeepSignatureTupletsDataset(Dataset):
    def __init__(self):
        self._tuplets = None

    def load_dataset(self, dir_path):
        self._tuplets = numpy.load(file=os.path.normpath(os.path.join(dir_path, 'tuplets.npy')), allow_pickle=True)

    def __len__(self):
        return self._tuplets.shape[0]

    def __getitem__(self, index):
        item = {}
        tuplet = self._tuplets[index]
        for key in tuplet.keys():
            item[key] = torch.from_numpy(numpy.array(self._tuplets[index][key]).astype('float64')).cuda().double()

        return item


class EuclideanTuple:
    @staticmethod
    def _generate_curvature_tuple(curves, sampling_ratio, multimodality, supporting_points_count, offset_length):
        return dataset_generation.EuclideanCurvatureTupletsDatasetGenerator.generate_tuple(
                curves=curves,
                sampling_ratio=sampling_ratio,
                multimodality=multimodality,
                supporting_points_count=supporting_points_count,
                offset_length=offset_length)

    @staticmethod
    def _generate_arclength_tuple(curves, sampling_ratio, multimodality, section_points_count):
        return dataset_generation.EuclideanArclengthTupletsDatasetGenerator.generate_tuple(
            curves=curves,
            sampling_ratio=sampling_ratio,
            multimodality=multimodality,
            section_points_count=section_points_count)


class EquiaffineTuple:
    @staticmethod
    def _generate_curvature_tuple(curves, sampling_ratio, multimodality, supporting_points_count, offset_length):
        return dataset_generation.EquiaffineCurvatureTupletsDatasetGenerator.generate_tuple(
                curves=curves,
                sampling_ratio=sampling_ratio,
                multimodality=multimodality,
                supporting_points_count=supporting_points_count,
                offset_length=offset_length)

    @staticmethod
    def _generate_arclength_tuple(curves, sampling_ratio, multimodality, section_points_count):
        return dataset_generation.EquiaffineArclengthTupletsDatasetGenerator.generate_tuple(
            curves=curves,
            sampling_ratio=sampling_ratio,
            multimodality=multimodality,
            section_points_count=section_points_count)


class AffineTuple:
    @staticmethod
    def _generate_curvature_tuple(curves, sampling_ratio, multimodality, supporting_points_count, offset_length):
        return dataset_generation.AffineCurvatureTupletsDatasetGenerator.generate_tuple(
                curves=curves,
                sampling_ratio=sampling_ratio,
                multimodality=multimodality,
                supporting_points_count=supporting_points_count,
                offset_length=offset_length)

    @staticmethod
    def _generate_arclength_tuple(curves, sampling_ratio, multimodality, section_points_count):
        return dataset_generation.AffineArclengthTupletsDatasetGenerator.generate_tuple(
                curves=curves,
                sampling_ratio=sampling_ratio,
                multimodality=multimodality,
                section_points_count=section_points_count)


class DeepSignatureTupletsOnlineDataset(Dataset):
    def __init__(self, dataset_size, dir_path, sampling_ratio, multimodality, buffer_size, num_workers):
        self._curves = curve_generation.CurvesGenerator.load_curves(dir_path)
        self._dataset_size = dataset_size
        self._sampling_ratio = sampling_ratio
        self._multimodality = multimodality
        self._buffer_size = buffer_size
        self._num_workers = num_workers
        self._q = Queue()
        self._args = [self._curves, self._sampling_ratio, self._multimodality, self._q]
        self._items = []

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        item = {}
        mod_index = numpy.mod(index, self._buffer_size)
        tuplet = self._items[mod_index]
        for key in tuplet.keys():
            item[key] = torch.from_numpy(numpy.array(tuplet[key]).astype('float64')).cuda().double()

        try:
            new_tuplet = self._q.get_nowait()
            rand_index = int(numpy.random.randint(self._buffer_size, size=1))
            self._items[rand_index] = new_tuplet
        except queue.Empty:
            pass

        return item

    def start(self):
        self._workers = [Process(target=self._map_func, args=self._args) for i in range(self._num_workers)]

        for i, worker in enumerate(self._workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {self._num_workers}', end='')

        print(f'\nItem {len(self._items)} / {self._buffer_size}', end='')
        while True:
            if self._q.empty() is False:
                self._items.append(self._q.get())
                print(f'\rItem {len(self._items)} / {self._buffer_size}', end='')
                if len(self._items) == self._buffer_size:
                    break


class DeepSignatureCurvatureTupletsOnlineDataset(DeepSignatureTupletsOnlineDataset):
    def __init__(self, dataset_size, dir_path, sampling_ratio, multimodality, buffer_size, num_workers, supporting_points_count, offset_length):
        DeepSignatureTupletsOnlineDataset.__init__(
            self,
            dataset_size=dataset_size,
            dir_path=dir_path,
            sampling_ratio=sampling_ratio,
            multimodality=multimodality,
            buffer_size=buffer_size,
            num_workers=num_workers)

        self._supporting_points_count = supporting_points_count
        self._args.append(supporting_points_count)

        self._offset_length = offset_length
        self._args.append(offset_length)

    @classmethod
    def _map_func(cls, curves, sampling_ratio, multimodality, q, supporting_points_count, offset_length):
        while True:
            q.put(cls._generate_curvature_tuple(
                curves=curves,
                sampling_ratio=sampling_ratio,
                multimodality=multimodality,
                supporting_points_count=supporting_points_count,
                offset_length=offset_length))


class DeepSignatureEuclideanCurvatureTupletsOnlineDataset(DeepSignatureCurvatureTupletsOnlineDataset, EuclideanTuple):
    pass


class DeepSignatureEquiaffineCurvatureTupletsOnlineDataset(DeepSignatureCurvatureTupletsOnlineDataset, EquiaffineTuple):
    pass


class DeepSignatureAffineCurvatureTupletsOnlineDataset(DeepSignatureCurvatureTupletsOnlineDataset, AffineTuple):
    pass


class DeepSignatureArclengthTupletsOnlineDataset(DeepSignatureTupletsOnlineDataset):
    def __init__(self, dataset_size, dir_path, sampling_ratio, multimodality, buffer_size, num_workers, section_points_count):
        DeepSignatureTupletsOnlineDataset.__init__(
            self,
            dataset_size=dataset_size,
            dir_path=dir_path,
            sampling_ratio=sampling_ratio,
            multimodality=multimodality,
            buffer_size=buffer_size,
            num_workers=num_workers)

        self._section_points_count = section_points_count
        self._args.append(section_points_count)

    @classmethod
    def _map_func(cls, curves, sampling_ratio, multimodality, q, section_points_count):
        while True:
            q.put(cls._generate_arclength_tuple(
                curves=curves,
                sampling_ratio=sampling_ratio,
                multimodality=multimodality,
                section_points_count=section_points_count))


class DeepSignatureEuclideanArclengthTupletsOnlineDataset(DeepSignatureArclengthTupletsOnlineDataset, EuclideanTuple):
    pass


class DeepSignatureEquiaffineArclengthTupletsOnlineDataset(DeepSignatureArclengthTupletsOnlineDataset, EquiaffineTuple):
    pass


class DeepSignatureAffineArclengthTupletsOnlineDataset(DeepSignatureArclengthTupletsOnlineDataset, AffineTuple):
    pass