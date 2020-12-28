# torch
import torch


class DeepSignatureNet(torch.nn.Module):
    def __init__(self, sample_points):
        super(DeepSignatureNet, self).__init__()
        self._regressor = DeepSignatureNet._create_regressor(in_features=2 * sample_points)

    def forward(self, input):
        features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
        output = self._regressor(features).reshape([input.shape[0], input.shape[1], 1])
        return output

    @staticmethod
    def _create_feature_extractor(kernel_size):
        return torch.nn.Sequential(
            DeepSignatureNet._create_cnn_block(
                in_channels=1,
                out_channels=64,
                kernel_size=kernel_size,
                first_block=True,
                last_block=False),
            DeepSignatureNet._create_cnn_block(
                in_channels=64,
                out_channels=32,
                kernel_size=kernel_size,
                first_block=False,
                last_block=False),
            DeepSignatureNet._create_cnn_block(
                in_channels=32,
                out_channels=16,
                kernel_size=kernel_size,
                first_block=False,
                last_block=True)
        )

    @staticmethod
    def _create_regressor(in_features):
        linear_modules = []
        in_features = in_features
        out_features = 80
        p = None
        while out_features > 10:
            linear_modules.extend(DeepSignatureNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True))
            linear_modules.extend(DeepSignatureNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True))
            in_features = out_features
            out_features = int(out_features / 2)

        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=1))

        # linear_modules.extend(SimpleDeepSignatureNet._create_hidden_layer(in_features=6, out_features=10, p=None, use_batch_norm=False))
        # linear_modules.extend(SimpleDeepSignatureNet._create_hidden_layer(in_features=10, out_features=5, p=None, use_batch_norm=False))
        # linear_modules.extend(SimpleDeepSignatureNet._create_hidden_layer(in_features=5, out_features=1, p=None, use_batch_norm=False))
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

    @staticmethod
    def _create_cnn_block(in_channels, out_channels, kernel_size, first_block, last_block):
        padding = int(kernel_size / 2)

        layers = [
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 2) if first_block is True else (kernel_size, 1),
                padding=(padding, 0),
                padding_mode='zeros'),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.GELU(),
            # torch.nn.Dropout2d(0.05),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='zeros'),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.GELU(),
            # torch.nn.Dropout2d(0.05),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='zeros'),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.GELU(),
            # torch.nn.Dropout2d(0.05),
        ]

        if not last_block:
            layers.append(torch.nn.MaxPool2d(
                kernel_size=(3, 1),
                padding=(1, 0)))

        return torch.nn.Sequential(*layers)
