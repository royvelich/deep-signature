# python peripherals
import os
import numpy
import random

# torch
import torch
from torch.utils.data import Dataset

# deep_signature
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
        tuplet = torch.from_numpy(self._tuplets[index].astype('float64')).cuda().double()
        return {
            'input': tuplet,
        }
