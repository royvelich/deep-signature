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

    @property
    def sample_points(self):
        return self._metadata['sample_points']

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
    def __init__(self, sample_points):
        super(DeepSignatureNet, self).__init__()

        self._feature_extractor = DeepSignatureNet._create_feature_extractor(
            kernel_size=5)

        dim_test = torch.unsqueeze(torch.unsqueeze(torch.rand(sample_points, 2), 0), 0)

        features = self._feature_extractor(dim_test)
        in_features = numpy.prod(features.shape)

        self._regressor = DeepSignatureNet._create_regressor(
            layers=3,
            in_features=in_features,
            sample_points=sample_points)

    def forward(self, x):
        features = self._feature_extractor(x)
        features_reshaped = features.reshape([x.shape[0], -1])
        output = self._regressor(features_reshaped)
        return output

    @staticmethod
    def _create_feature_extractor(kernel_size):
        return torch.nn.Sequential(
            DeepSignatureNet._create_cnn_block(
                in_channels=1,
                out_channels=64,
                kernel_size=kernel_size,
                first_block=True),
            DeepSignatureNet._create_cnn_block(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel_size,
                first_block=False),
            DeepSignatureNet._create_cnn_block(
                in_channels=128,
                out_channels=256,
                kernel_size=kernel_size,
                first_block=False)
        )

    @staticmethod
    def _create_regressor(layers, in_features, sample_points):
        linear_modules = []
        for _ in range(layers):
            out_features = int(in_features / 2)
            linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            linear_modules.append(torch.nn.ReLU())
            in_features = out_features

        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=sample_points))

        return torch.nn.Sequential(*linear_modules)

    @staticmethod
    def _create_cnn_block(in_channels, out_channels, kernel_size, first_block):
        padding = int(kernel_size / 2)
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 2) if first_block is True else (kernel_size, 1),
                padding=(padding, 0),
                padding_mode='circular'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='circular'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='circular'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='circular'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(3, 1),
                padding=(1, 0))
        )


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, mu=1):
        super(ContrastiveLoss, self).__init__()
        self._mu = mu

    def forward(self, x1, x2, y):
        diff = x1 - x2
        diff_norm = torch.norm(diff, dim=1)

        positive_penalties = y * diff_norm
        negative_penalties = (1 - y) * torch.max(torch.zeros_like(diff_norm), self._mu - diff_norm)

        positive_penalty = torch.sum(positive_penalties)
        negative_penalty = torch.sum(negative_penalties)

        return (positive_penalty + negative_penalty) / x1.shape[0]
