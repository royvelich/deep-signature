# python peripherals
import numpy
import copy
import math
from typing import Optional, Tuple, Union, List

# torch
import torch

# lightly
from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum



class DBN(torch.nn.Module):
    def __init__(self, num_features, num_groups=32, num_channels=0, dim=2, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(DBN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(*self.shape))
            self.bias = torch.nn.Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, 1))
        self.register_buffer('running_projection', torch.eye(num_groups))
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.uniform_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor, shuffle=False):

        if isinstance(shuffle, list):
            input = input[shuffle[0]]


        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        # breakpoint()
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        # print(x.size())
        if training:
            mean = x.mean(1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            x_mean = x - mean
            # sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            # print(sigma.mean())
            # with open('/root/input.pkl', 'wb+') as f:
            #     pickle.dump(x.detach().cpu().numpy(), f)
            # with open('/root/sigma.pkl', 'wb+') as f:
            #     pickle.dump(sigma.detach().cpu().numpy(), f)
            # breakpoint()
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) #+ self.eps * torch.eye(self.num_groups, device=input.device)
            # print('sigma size {}'.format(sigma.size()))
            # print(sigma.shape)
            try:
                u, eig, tmp = sigma.cpu().svd()
                # import pdb
                # pdb.set_trace()
            except RuntimeError:
                import pdb
                pdb.set_trace()
                print(sigma[:5,:5])
                exit()
            # with open('/root/input.pkl', 'wb+') as f:
            #     pickle.dump(x.detach().cpu().numpy(), f)


            u = u.to(input.device)
            eig = eig.to(input.device)
            self.eig = eig[-1]
            scale = eig.rsqrt()
            # scale = 1 / eig
            wm = u.matmul(scale.diag()).matmul(u.t())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm
            y = wm.matmul(x_mean)

            # x_mean = x - self.running_mean.detach()
            # y = self.running_projection.detach().matmul(x_mean)
        else:
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        # if self.affine:
        #     output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)


class DBN2(DBN):
    """
    when evaluation phase, sigma using running average.
    """

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        mean = x.mean(1, keepdim=True) if training else self.running_mean
        x_mean = x - mean
        if training:
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * sigma
        else:
            sigma = self.running_projection
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = u.matmul(scale.diag()).matmul(u.t())
        y = wm.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output



linalg_device = 'cpu'

def transformation(covs, device, engine='svd'):
    covs = covs.to(linalg_device)
    if engine == 'cholesky':
        C = torch.cholesky(covs.to(linalg_device))
        W = torch.triangular_solve(torch.eye(C.size(-1)).expand_as(C).to(C), C, upper=False)[0].transpose(1, 2).to(x.device)
    else:
        if engine == 'symeig':

            S, U = torch.symeig(covs.to(linalg_device), eigenvectors=True, upper=True)
        elif engine == 'svd':
            U, S, _ = torch.svd(covs.to(linalg_device))
        elif engine == 'svd_lowrank':
            U, S, _ = torch.svd_lowrank(covs.to(linalg_device))
        elif engine == 'pca_lowrank':
            U, S, _ = torch.pca_lowrank(covs.to(linalg_device), center=False)
        S, U = S.to(device), U.to(device)
        W = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1, 2))
    return W


class ShuffledGroupWhitening(torch.nn.Module):
    def __init__(self, num_features, num_groups=None, shuffle=True, engine='svd'):
        super(ShuffledGroupWhitening, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        if self.num_groups is not None:
            assert self.num_features % self.num_groups == 0
        # self.momentum = momentum
        self.register_buffer('running_mean', None)
        self.register_buffer('running_covariance', None)
        self.shuffle = shuffle if self.num_groups != 1 else False
        self.engine = engine

    def forward(self, x):
        N, D = x.shape
        if self.num_groups is None:
            G = math.ceil(2 * D / N)  # automatic, the grouped dimension 'D/G' should be half of the batch size N
            # print(G, D, N)
        else:
            G = self.num_groups
        if self.shuffle:
            new_idx = torch.randperm(D)
            x = x.t()[new_idx].t()
        x = x.view(N, G, D // G)
        x = (x - x.mean(dim=0, keepdim=True)).transpose(0, 1)  # G, N, D//G
        # covs = x.transpose(1,2).bmm(x) / (N-1) #  G, D//G, N @ G, N, D//G -> G, D//G, D//G
        covs = x.transpose(1, 2).bmm(x) / N
        W = transformation(covs, x.device, engine=self.engine)
        x = x.bmm(W)
        if self.shuffle:
            return x.transpose(1, 2).flatten(0, 1)[torch.argsort(new_idx)].t()
        else:
            return x.transpose(0, 1).flatten(1)


# class ProjectionHead(torch.nn.Module):
#     """Base class for all projection and prediction heads.
#
#     Args:
#         blocks:
#             List of tuples, each denoting one block of the projection head MLP.
#             Each tuple reads (in_features, out_features, batch_norm_layer,
#             non_linearity_layer).
#
#     Examples:
#         >>> # the following projection head has two blocks
#         >>> # the first block uses batch norm an a ReLU non-linearity
#         >>> # the second block is a simple linear layer
#         >>> projection_head = ProjectionHead([
#         >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
#         >>>     (256, 128, None, None)
#         >>> ])
#
#     """
#
#     def __init__(
#         self,
#         blocks: List[Tuple[int, int, Optional[torch.nn.Module], Optional[torch.nn.Module]]]
#     ):
#         super(ProjectionHead, self).__init__()
#
#         layers = []
#         for input_dim, output_dim, batch_norm, non_linearity in blocks:
#             use_bias = not bool(batch_norm)
#             layers.append(torch.nn.Linear(input_dim, output_dim, bias=use_bias))
#             if batch_norm:
#                 layers.append(batch_norm)
#             if non_linearity:
#                 layers.append(non_linearity)
#         self.layers = torch.nn.Sequential(*layers)
#
#     def forward(self, x: torch.Tensor):
#         """Computes one forward pass through the projection head.
#
#         Args:
#             x:
#                 Input of shape bsz x num_ftrs.
#
#         """
#         return self.layers(x)
#
#
# class BYOLProjectionHead(ProjectionHead):
#     """Projection head used for BYOL.
#
#     "This MLP consists in a linear layer with output size 4096 followed by
#     batch normalization, rectified linear units (ReLU), and a final
#     linear layer with output dimension 256." [0]
#
#     [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733
#
#     """
#     def __init__(self,
#                  input_dim: int = 2048,
#                  hidden_dim: int = 4096,
#                  output_dim: int = 256):
#         super(BYOLProjectionHead, self).__init__([
#             (input_dim, hidden_dim, torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU()),
#             (hidden_dim, output_dim, DBN(output_dim, num_groups=1), None),
#         ])
#
#
# class BYOLPredictionHead(ProjectionHead):
#     """Prediction head used for BYOL.
#
#     "This MLP consists in a linear layer with output size 4096 followed by
#     batch normalization, rectified linear units (ReLU), and a final
#     linear layer with output dimension 256." [0]
#
#     [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733
#
#     """
#     def __init__(self,
#                  input_dim: int = 256,
#                  hidden_dim: int = 4096,
#                  output_dim: int = 256):
#         super(BYOLPredictionHead, self).__init__([
#             (input_dim, hidden_dim, torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU()),
#             (hidden_dim, output_dim, DBN(output_dim, num_groups=1), None),
#             # (hidden_dim, output_dim, None, None),
#         ])


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
        self.projection_head = BYOLProjectionHead(32, 16, 2)
        self.prediction_head = BYOLPredictionHead(2, 16, 2)

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
        self._L = 8
        # self._regressor = DifferentialInvariantsNet._create_regressor(in_features=2*sample_points*2*self._L)
        self._regressor = DifferentialInvariantsNet._create_regressor(in_features=2*sample_points)
        # self._regressor.apply(sine_init)
        # self._regressor[0].apply(first_layer_sine_init)

    def forward(self, in_features):
        x = in_features.reshape([in_features.shape[0] * in_features.shape[1], in_features.shape[2] * in_features.shape[3]])

        # coeffs = torch.pow(2, torch.linspace(0, self._L-1, steps=self._L)).repeat_interleave(2).cuda()
        # x2 = x.unsqueeze(-1).expand(x.shape[0], x.shape[1], 2*self._L)
        # x3 = x2 * coeffs * numpy.pi
        # for i in range(2*self._L):
        #     if i % 2 == 0:
        #         x3[:, :, i] = torch.sin(x3[:, :, i])
        #     else:
        #         x3[:, :, i] = torch.cos(x3[:, :, i])
        #
        # x4 = x3.reshape([x3.shape[0], x3.shape[1] * x3.shape[2]])
        #
        # output = self._regressor(x4)

        output = self._regressor(x)
        return output

    @staticmethod
    def _create_regressor(in_features):
        linear_modules = []
        in_features = in_features
        out_features = 128
        p = None
        # while out_features > 16:
        #     linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
        #     linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
        #     # linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
        #     in_features = out_features
        #     out_features = int(out_features / 2)

        for i in range(10):
            linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            # linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            # linear_modules.extend(DifferentialInvariantsNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            in_features = out_features

        linear_modules.append(torch.nn.Linear(in_features=out_features, out_features=32))

        return torch.nn.Sequential(*linear_modules)

    @staticmethod
    def _create_hidden_layer(in_features, out_features, p=None, use_batch_norm=False, weights_init=None):
        linear_modules = []
        linear_module = torch.nn.Linear(in_features=in_features, out_features=out_features)

        linear_modules.append(linear_module)

        if use_batch_norm:
            # linear_modules.append(DBN(num_features=out_features))
            # linear_modules.append(ShuffledGroupWhitening(out_features, num_groups=None, shuffle=False))
            linear_modules.append(torch.nn.BatchNorm1d(out_features))

        # linear_modules.append(Sine())
        # linear_modules.append(torch.nn.ReLU())
        linear_modules.append(torch.nn.PReLU(num_parameters=out_features))

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
