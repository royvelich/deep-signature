# python peripherals
import numpy
import copy

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


class DifferentialInvariantsNetBYOL(torch.nn.Module):
    def __init__(self, sample_points):
        super(DifferentialInvariantsNetBYOL, self).__init__()
        self._diff_invariants_net = DifferentialInvariantsNet(sample_points=sample_points)
        self._model = BYOL(self._diff_invariants_net)

    def forward(self, in_features):
        x0 = in_features[:, 0, :, :].unsqueeze(dim=1)
        x1 = in_features[:, 1, :, :].unsqueeze(dim=1)
        update_momentum(self._model.backbone, self._model.backbone_momentum, m=0.99)
        update_momentum(self._model.projection_head, self._model.projection_head_momentum, m=0.99)
        p0, z0 = self._model(x0)
        z0_momentum = self._model.forward_momentum(x0)
        p1, z1 = self._model(x1)
        z1_momentum = self._model.forward_momentum(x1)

        return p0, p1, z0, z1, z0_momentum, z1_momentum

        # features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
        # output = self._regressor1(features).reshape([input.shape[0], input.shape[1], 1])
        # return output


class BYOL(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(32, 512, 8)
        self.prediction_head = BYOLPredictionHead(8, 64, 8)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p, z

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


class DifferentialInvariantsNet(torch.nn.Module):
    def __init__(self, sample_points):
        super(DifferentialInvariantsNet, self).__init__()
        self._regressor = DifferentialInvariantsNet._create_regressor(in_features=2*sample_points)
        # self._regressor2 = DifferentialInvariantsNet._create_regressor(in_features=2*sample_points)
        # self._regressor.apply(sine_init)
        # self._regressor[0].apply(first_layer_sine_init)

    def forward(self, in_features):
        x = in_features.reshape([in_features.shape[0] * in_features.shape[1], in_features.shape[2] * in_features.shape[3]])
        # output = self._regressor(x).reshape([input.shape[0], input.shape[1], 1])
        output = self._regressor(x)
        return output
        # output2 = self._regressor2(features).reshape([input.shape[0], input.shape[1], 1])
        # return torch.cat((output1, output2), dim=2)

    @staticmethod
    def _create_regressor(in_features):
        linear_modules = []
        in_features = in_features
        out_features = 256
        p = None
        while out_features > 16:
            linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            # linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            in_features = out_features
            out_features = int(out_features / 2)

        # linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=1))

        return torch.nn.Sequential(*linear_modules)

    @staticmethod
    def _create_hidden_layer(in_features, out_features, p=None, use_batch_norm=False, weights_init=None):
        linear_modules = []
        linear_module = torch.nn.Linear(in_features=in_features, out_features=out_features)

        linear_modules.append(linear_module)

        if use_batch_norm:
            linear_modules.append(torch.nn.BatchNorm1d(out_features))

        linear_modules.append(Sine())
        # linear_modules.append(torch.nn.ReLU())
        # linear_modules.append(torch.nn.PReLU(num_parameters=out_features))

        if p is not None:
            linear_modules.append(torch.nn.Dropout(p))

        return linear_modules


class CurvatureNet(torch.nn.Module):
    def __init__(self, sample_points):
        super(CurvatureNet, self).__init__()
        self._regressor = CurvatureNet._create_regressor(in_features=2*sample_points)

    def forward(self, input):
        features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
        output = self._regressor(features).reshape([input.shape[0], input.shape[1], 1])
        return output

    @staticmethod
    def _create_regressor(in_features):
        linear_modules = []
        in_features = in_features
        out_features = 100
        p = None
        while out_features > 10:
            linear_modules.extend(CurvatureNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            linear_modules.extend(CurvatureNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            in_features = out_features
            out_features = int(out_features / 2)

        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=1))

        return torch.nn.Sequential(*linear_modules)

    @staticmethod
    def _create_hidden_layer(in_features, out_features, p=None, use_batch_norm=False, weights_init=None):
        linear_modules = []
        linear_module = torch.nn.Linear(in_features=in_features, out_features=out_features)

        linear_modules.append(linear_module)

        if use_batch_norm:
            linear_modules.append(torch.nn.BatchNorm1d(out_features))

        # linear_modules.append(Sine())
        # linear_modules.append(torch.nn.ReLU())
        linear_modules.append(torch.nn.PReLU(num_parameters=out_features))

        if p is not None:
            linear_modules.append(torch.nn.Dropout(p))

        return linear_modules


class ArcLengthNet(torch.nn.Module):
    def __init__(self, sample_points):
        super(ArcLengthNet, self).__init__()
        self._regressor = ArcLengthNet._create_regressor(in_features=2 * sample_points)

    def forward(self, input):
        features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
        output = self._regressor(features).reshape([input.shape[0], input.shape[1], 1])
        return output.abs()

    @staticmethod
    def _create_regressor(in_features):
        linear_modules = []
        in_features = in_features
        out_features = 32

        p = None
        while out_features > 2:
            linear_modules.extend(ArcLengthNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True))
            linear_modules.extend(ArcLengthNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True))
            # linear_modules.extend(DeepSignatureArcLengthNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True))
            in_features = out_features
            out_features = int(out_features / 2)

        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=1))
        return torch.nn.Sequential(*linear_modules)

    @staticmethod
    def _create_hidden_layer(in_features, out_features, p=None, use_batch_norm=False):
        linear_modules = []
        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
        if use_batch_norm:
            linear_modules.append(torch.nn.BatchNorm1d(out_features))

        # linear_modules.append(torch.nn.PReLU(num_parameters=out_features))
        # linear_modules.append(torch.nn.LeakyReLU())
        # linear_modules.append(torch.nn.ReLU())
        linear_modules.append(torch.nn.GELU())
        # linear_modules.append(Sine())

        if p is not None:
            linear_modules.append(torch.nn.Dropout(p))
        return linear_modules
