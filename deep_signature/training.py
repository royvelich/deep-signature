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

        curve1_tensor = torch.unsqueeze(torch.from_numpy(curve1_sample), 0).cuda().float()
        curve2_tensor = torch.unsqueeze(torch.from_numpy(curve2_sample), 0).cuda().float()
        labels_tensor = torch.squeeze(torch.from_numpy(numpy.array([label])).cuda().float(), 0)

        return {
            'curves': [curve1_tensor, curve2_tensor],
            'labels': labels_tensor
        }

    @staticmethod
    def _build_curve_path(dir_path, curve_descriptor):
        return os.path.normpath(os.path.join(dir_path, f'{curve_descriptor[0]}/{curve_descriptor[1]}/{curve_descriptor[2]}', 'sample.npy'))

    @staticmethod
    def _load_curve_sample(dir_path, curve_descriptor):
        return numpy.load(file=DeepSignatureDataset._build_curve_path(dir_path, curve_descriptor), allow_pickle=True)


class DeepSignatureNet(torch.nn.Module):
    def __init__(self, sample_points, padding):
        super(DeepSignatureNet, self).__init__()

        self._cnn1 = DeepSignatureNet.cnn_block(in_channels=1, padding=padding)
        self._cnn2 = DeepSignatureNet.cnn_block(in_channels=64, padding=padding)
        self._cnn3 = DeepSignatureNet.cnn_block(in_channels=64, padding=padding)

        dim_test = torch.unsqueeze(torch.unsqueeze(torch.rand(sample_points, 2), 0), 0)
        cnn1_out = self._cnn1(dim_test)
        cnn2_out = self._cnn2(cnn1_out)
        cnn3_out = self._cnn3(cnn2_out)
        in_features = numpy.prod(cnn3_out.shape)

        linear_modules = []
        for _ in range(4):
            out_features = int(in_features / 2)
            linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            linear_modules.append(torch.nn.ReLU())
            in_features = out_features

        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=sample_points))

        self._linear = torch.nn.Sequential(*linear_modules)

    def forward(self, x):

        cnn1_out = self._cnn1(x)
        cnn2_out = self._cnn2(cnn1_out)
        cnn3_out = self._cnn3(cnn2_out)

        cnn_output = cnn3_out.reshape([x.shape[0], -1])
        output = self._linear(cnn_output)
        return output

    @staticmethod
    def cnn_block(in_channels, padding):
        cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(2*padding + 1, 2),
                padding=padding,
                padding_mode='circular'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=padding,
                padding_mode='circular'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=padding,
                padding_mode='circular'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=padding,
                padding_mode='circular'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=5, padding=2)
        )

        return cnn


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, mu=1):
        super(ContrastiveLoss, self).__init__()
        self._mu = mu

    def forward(self, x1, x2, y):
        diff = x1 - x2
        diff_norm = torch.norm(diff)

        positive_penalties = y * diff_norm
        negative_penalties = (1 - y) * torch.max(torch.zeros_like(diff_norm), self._mu - diff_norm)

        positive_penalty = torch.sum(positive_penalties)
        negative_penalty = torch.sum(negative_penalties)

        return (positive_penalty + negative_penalty) / x1.shape[0]
