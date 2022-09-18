# python peripherals
import numpy
import copy
import math
from typing import Optional, Tuple, Union, List, Callable

# torch
import torch

# lightly
from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum


# Taken from https://github.com/vsitzmann/siren
class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(input)


# Taken from https://github.com/vsitzmann/siren
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-numpy.sqrt(6 / num_input) / 30, numpy.sqrt(6 / num_input) / 30)


# Taken from https://github.com/vsitzmann/siren
def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class BYOL(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, projection_head: BYOLProjectionHead, prediction_head: BYOLPredictionHead):
        super().__init__()

        self.backbone = backbone
        self.projection_head = projection_head
        self.prediction_head = prediction_head

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


class DeepSignaturesNet(torch.nn.Module):
    def __init__(self, sample_points: int, in_features_size: int, out_features_size: int, hidden_layer_repetitions: int, create_activation_fn: Callable[[int], torch.nn.Module], create_batch_norm_fn: Callable[[int], torch.nn.Module], dropout_p: float = None):
        super().__init__()
        self._sample_points = sample_points
        self._in_features_size = in_features_size
        self._out_features_size = out_features_size
        self._hidden_layer_repetitions = hidden_layer_repetitions
        self._create_activation_fn = create_activation_fn
        self._create_batch_norm_fn = create_batch_norm_fn
        self._dropout_p = dropout_p
        self._regressor = self._create_regressor()

    def forward(self, in_features):
        x = in_features.reshape([in_features.shape[0] * in_features.shape[1], in_features.shape[2] * in_features.shape[3]])
        output = self._regressor(x).reshape([in_features.shape[0], in_features.shape[1], 2])
        return output

    def _create_regressor(self):
        linear_modules = []
        in_features_size = 2 * self._sample_points
        out_features_size = self._in_features_size
        while out_features_size > 2:
            for _ in range(self._hidden_layer_repetitions):
                linear_modules.extend(self._create_hidden_layer(in_features_size=in_features_size, out_features_size=out_features_size))
            in_features_size = out_features_size
            out_features_size = int(out_features_size / 2)

        linear_modules.append(torch.nn.Linear(in_features=in_features_size, out_features=2))
        return torch.nn.Sequential(*linear_modules)

    def _create_hidden_layer(self, in_features_size: int, out_features_size: int):
        linear_modules = []
        linear_module = torch.nn.Linear(in_features=in_features_size, out_features=out_features_size)
        linear_modules.append(linear_module)

        if self._create_batch_norm_fn is not None:
            linear_modules.append(self._create_batch_norm_fn(out_features_size))

        linear_modules.append(self._create_activation_fn(out_features_size))

        if self._dropout_p is not None:
            linear_modules.append(torch.nn.Dropout(p=self._dropout_p))

        return linear_modules


class DeepSignaturesNetBYOL(torch.nn.Module):
    def __init__(self, deep_signatures_backbone: DeepSignaturesNet, projection_head: BYOLProjectionHead, prediction_head: BYOLPredictionHead):
        super().__init__()
        self._deep_signatures_backbone = deep_signatures_backbone
        self._model = BYOL(backbone=self._deep_signatures_backbone, projection_head=projection_head, prediction_head=prediction_head)

    def forward(self, in_features):
        x0 = in_features[:, 0, :, :].unsqueeze(dim=1)
        x1 = in_features[:, 1, :, :].unsqueeze(dim=1)
        update_momentum(self._model.backbone, self._model.backbone_momentum, m=0.99)
        update_momentum(self._model.projection_head, self._model.projection_head_momentum, m=0.99)
        p0 = self._model(x0)
        z0 = self._model.forward_momentum(x0)
        p1 = self._model(x1)
        z1 = self._model.forward_momentum(x1)
        return {
            'p0': p0,
            'p1': p1,
            'z0': z0,
            'z1': z1
        }
