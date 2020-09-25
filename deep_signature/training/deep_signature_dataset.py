import torch
from torch.utils.data import Dataset
import os
import numpy


class DeepSignatureDataset(Dataset):
    def __init__(self, dir_path):
        self._dir_path = dir_path
        self._metadata = numpy.load(file=os.path.normpath(os.path.join(dir_path, 'metadata.npy')), allow_pickle=True)
        self._metadata = self._metadata.item()
        self._pairs = self._metadata['pairs']

    def __len__(self):
        return self._pairs.shape[0]

    def __getitem__(self, idx):
        pair = self._pairs[idx]

        label = pair[0]
        curve1_descriptor = pair[1:4]
        curve2_descriptor = pair[4:8]

        curve1_sample = DeepSignatureDataset._load_curve_sample(dir_path=self._dir_path, curve_descriptor=curve1_descriptor)
        curve2_sample = DeepSignatureDataset._load_curve_sample(dir_path=self._dir_path, curve_descriptor=curve2_descriptor)

        return {
            'curves': [torch.from_numpy(curve1_sample), torch.from_numpy(curve2_sample)],
            'labels': label
        }

    @staticmethod
    def _build_curve_path(dir_path, curve_descriptor):
        return os.path.normpath(os.path.join(dir_path, f'{curve_descriptor[0]}/{curve_descriptor[1]}/{curve_descriptor[2]}', 'sample.npy'))

    @staticmethod
    def _load_curve_sample(dir_path, curve_descriptor):
        return numpy.load(file=DeepSignatureDataset._build_curve_path(dir_path, curve_descriptor), allow_pickle=True)

