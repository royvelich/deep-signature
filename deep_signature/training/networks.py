# python peripherals
import copy
from typing import Callable

# torch
import torch

# lightly
# from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
# from lightly.models.utils import deactivate_requires_grad
# from lightly.models.utils import update_momentum


# class BYOL(torch.nn.Module):
#     def __init__(self, backbone: torch.nn.Module, projection_head: BYOLProjectionHead, prediction_head: BYOLPredictionHead):
#         super().__init__()
#
#         self.backbone = backbone
#         self.projection_head = projection_head
#         self.prediction_head = prediction_head
#
#         self.backbone_momentum = copy.deepcopy(self.backbone)
#         self.projection_head_momentum = copy.deepcopy(self.projection_head)
#
#         deactivate_requires_grad(self.backbone_momentum)
#         deactivate_requires_grad(self.projection_head_momentum)
#
#     def forward(self, x):
#         y = self.backbone(x).flatten(start_dim=1)
#         z = self.projection_head(y)
#         p = self.prediction_head(z)
#         return p
#
#     def forward_momentum(self, x):
#         y = self.backbone_momentum(x).flatten(start_dim=1)
#         z = self.projection_head_momentum(y)
#         z = z.detach()
#         return z


# class DeepSignaturesNet(torch.nn.Module):
#     def __init__(self, sample_points: int, in_features_size: int, out_features_size: int, hidden_layer_repetitions: int, create_activation_fn: Callable[[int], torch.nn.Module], create_batch_norm_fn: Callable[[int], torch.nn.Module], dropout_p: float = None):
#         super(DeepSignaturesNet, self).__init__()
#         self._L = 4
#         # self._regressor = DifferentialInvariantsNet._create_regressor(in_features=2*sample_points*2*self._L)
#         self._regressor = DeepSignaturesNet._create_regressor(in_features=2*sample_points)
#         # self._regressor = DifferentialInvariantsNet._create_regressor(in_features=9*sample_points)
#
#     def forward(self, in_features):
#         # x = in_features.reshape([in_features.shape[0] * in_features.shape[1], in_features.shape[2] * in_features.shape[3]])
#         # coeffs = torch.pow(2, torch.linspace(0, self._L-1, steps=self._L)).repeat_interleave(2).cuda()
#         # x2 = x.unsqueeze(-1).expand(x.shape[0], x.shape[1], 2*self._L)
#         # x3 = x2 * coeffs * numpy.pi
#         # for i in range(2*self._L):
#         #     if i % 2 == 0:
#         #         x3[:, :, i] = torch.sin(x3[:, :, i])
#         #     else:
#         #         x3[:, :, i] = torch.cos(x3[:, :, i])
#         #
#         # x4 = x3.reshape([x3.shape[0], x3.shape[1] * x3.shape[2]])
#         #
#         # regressor_output = self._regressor(x4)
#         # output = regressor_output.reshape([in_features.shape[0], in_features.shape[1], 2])
#
#         # x = in_features[:, :, :, 0]
#         # y = in_features[:, :, :, 1]
#         # x2 = x * x
#         # y2 = y * y
#         # xy = x * y
#         # x3 = x * x * x
#         # y3 = y * y * y
#         # xxy = x * x * y
#         # xyy = x * y * y
#         #
#         # extended_in_features = torch.stack((x, y, x2, y2, xy, x3, y3, xxy, xyy), dim=-1)
#         # z = extended_in_features.reshape([extended_in_features.shape[0] * extended_in_features.shape[1], extended_in_features.shape[2] * extended_in_features.shape[3]])
#         # output = self._regressor(z).reshape([in_features.shape[0], in_features.shape[1], 2])
#
#
#         z = in_features.reshape([in_features.shape[0] * in_features.shape[1], in_features.shape[2] * in_features.shape[3]])
#         output = self._regressor(z).reshape([in_features.shape[0], in_features.shape[1], 2])
#
#         return output
#
#     @staticmethod
#     def _create_regressor(in_features):
#         linear_modules = []
#         in_features = in_features
#         out_features = 64
#         p = None
#         while out_features > 2:
#             linear_modules.extend(DeepSignaturesNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
#             linear_modules.extend(DeepSignaturesNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
#             # linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
#             in_features = out_features
#             out_features = int(out_features / 2)
#
#         # for _ in range(7):
#         #     linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
#         #     in_features = out_features
#
#         linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=2))
#
#         return torch.nn.Sequential(*linear_modules)
#
#     @staticmethod
#     def _create_hidden_layer(in_features, out_features, p=None, use_batch_norm=False, weights_init=None):
#         linear_modules = []
#         linear_module = torch.nn.Linear(in_features=in_features, out_features=out_features)
#
#         linear_modules.append(linear_module)
#
#         if use_batch_norm:
#             # linear_modules.append(DBN(num_features=out_features, num_groups=1))
#             linear_modules.append(torch.nn.BatchNorm1d(out_features))
#
#         # linear_modules.append(Sine())
#         linear_modules.append(torch.nn.ReLU())
#         # linear_modules.append(torch.nn.PReLU(num_parameters=out_features))
#
#         if p is not None:
#             linear_modules.append(torch.nn.Dropout(p))
#
#         return linear_modules







class DeepSignaturesNet(torch.nn.Module):
    def __init__(self, sample_points: int, in_features_size: int, out_features_size: int, hidden_layer_repetitions: int, create_activation_fn: Callable[[int], torch.nn.Module], create_batch_norm_fn: Callable[[int], torch.nn.Module], dropout_p: float = None):
        super().__init__()
        self._sample_points = sample_points
        self._in_features_size = in_features_size
        self._out_features_size = out_features_size
        self._hidden_layer_repetitions = hidden_layer_repetitions
        # self._create_activation_fn = create_activation_fn
        # self._create_batch_norm_fn = create_batch_norm_fn
        self._dropout_p = dropout_p
        self._regressor = self._create_regressor(create_activation_fn=create_activation_fn, create_batch_norm_fn=create_batch_norm_fn)

    # def __getstate__(self):
    #     d = dict(self.__dict__)
    #     del d['_create_activation_fn']
    #     del d['_create_batch_norm_fn']
    #     return d

    def forward(self, in_features):
        x = in_features.reshape([in_features.shape[0] * in_features.shape[1], in_features.shape[2] * in_features.shape[3]])
        output = self._regressor(x).reshape([in_features.shape[0], in_features.shape[1], 2])
        return output

    def _create_regressor(self, create_activation_fn: Callable[[int], torch.nn.Module], create_batch_norm_fn: Callable[[int], torch.nn.Module]):
        linear_modules = []
        in_features_size = 2 * self._sample_points
        out_features_size = self._in_features_size
        while out_features_size > 2:
            for i in range(self._hidden_layer_repetitions):
                linear_modules.extend(self._create_hidden_layer(in_features_size=in_features_size if i == 0 else out_features_size, out_features_size=out_features_size, create_activation_fn=create_activation_fn, create_batch_norm_fn=create_batch_norm_fn))

            in_features_size = out_features_size
            out_features_size = int(out_features_size / 2)

        linear_modules.append(torch.nn.Linear(in_features=in_features_size, out_features=2))
        return torch.nn.Sequential(*linear_modules)

    def _create_hidden_layer(self, in_features_size: int, out_features_size: int, create_activation_fn: Callable[[int], torch.nn.Module], create_batch_norm_fn: Callable[[int], torch.nn.Module]):
        linear_modules = []
        linear_module = torch.nn.Linear(in_features=in_features_size, out_features=out_features_size)
        linear_modules.append(linear_module)
        linear_modules.append(create_batch_norm_fn(out_features_size))
        linear_modules.append(create_activation_fn(out_features_size))

        if self._dropout_p is not None:
            linear_modules.append(torch.nn.Dropout(p=self._dropout_p))

        return linear_modules


# class DeepSignaturesNetBYOL(torch.nn.Module):
#     def __init__(self, deep_signatures_backbone: DeepSignaturesNet, projection_head: BYOLProjectionHead, prediction_head: BYOLPredictionHead):
#         super().__init__()
#         self._deep_signatures_backbone = deep_signatures_backbone
#         self._model = BYOL(backbone=self._deep_signatures_backbone, projection_head=projection_head, prediction_head=prediction_head)
#
#     def forward(self, in_features):
#         x0 = in_features[:, 0, :, :].unsqueeze(dim=1)
#         x1 = in_features[:, 1, :, :].unsqueeze(dim=1)
#         update_momentum(self._model.backbone, self._model.backbone_momentum, m=0.99)
#         update_momentum(self._model.projection_head, self._model.projection_head_momentum, m=0.99)
#         p0 = self._model(x0)
#         z0 = self._model.forward_momentum(x0)
#         p1 = self._model(x1)
#         z1 = self._model.forward_momentum(x1)
#         return {
#             'p0': p0,
#             'p1': p1,
#             'z0': z0,
#             'z1': z1
#         }
