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

        curve1 = numpy.load(file=DeepSignatureDataset.build_curve_path(self._dir_path, curve1_descriptor),
                            allow_pickle=True)

        curve2 = numpy.load(file=DeepSignatureDataset.build_curve_path(self._dir_path, curve2_descriptor),
                            allow_pickle=True)

        return {
            'pair': torch.from_numpy(numpy.vstack((numpy.transpose(curve1), numpy.transpose(curve2)))),
            'label': label
        }

    @staticmethod
    def build_curve_path(dir_path, curve_descriptor):
        return numpy.load(
            file=os.path.normpath(os.path.join(
                dir_path,
                f'{curve_descriptor[0]}/{curve_descriptor[1]}/{curve_descriptor[2]}',
                'sample.npy')),
            allow_pickle=True)
