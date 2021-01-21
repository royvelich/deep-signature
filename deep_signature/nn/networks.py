# torch
import torch


class DeepSignatureCurvatureNet(torch.nn.Module):
    def __init__(self, sample_points):
        super(DeepSignatureCurvatureNet, self).__init__()
        self._regressor = DeepSignatureCurvatureNet._create_regressor(in_features=2 * sample_points)

    def forward(self, input):
        features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
        output = self._regressor(features).reshape([input.shape[0], input.shape[1], 1])
        return output

    @staticmethod
    def _create_regressor(in_features):
        linear_modules = []
        in_features = in_features
        out_features = 80
        p = None
        while out_features > 10:
            linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True))
            linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True))
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

        linear_modules.append(torch.nn.GELU())

        if p is not None:
            linear_modules.append(torch.nn.Dropout(p))
        return linear_modules


class DeepSignatureArcLengthNet(torch.nn.Module):
    def __init__(self, sample_points):
        super(DeepSignatureArcLengthNet, self).__init__()
        self._regressor = DeepSignatureArcLengthNet._create_regressor(in_features=2 * sample_points)

    def forward(self, input):
        features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
        output = self._regressor(features).reshape([input.shape[0], input.shape[1], 1])
        v1 = output[:, 0, :].unsqueeze(dim=1)
        v2 = output[:, 1:, :]
        v3 = v2[:, 0::2, :]
        v4 = v2[:, 1::2, :]
        v5 = v3 + v4
        v6 = torch.cat((v1, v5), dim=1)
        return v6

    @staticmethod
    def _create_regressor(in_features):
        linear_modules = []
        in_features = in_features
        out_features = 140
        p = None
        while out_features > 10:
            linear_modules.extend(DeepSignatureArcLengthNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True))
            linear_modules.extend(DeepSignatureArcLengthNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True))
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

        linear_modules.append(torch.nn.GELU())

        if p is not None:
            linear_modules.append(torch.nn.Dropout(p))
        return linear_modules
