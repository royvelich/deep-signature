import torch
import numpy


class DeepSignatureNet(torch.nn.Module):
    def __init__(self, sample_points, padding):
        super(DeepSignatureNet, self).__init__()
        self._cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(2*padding + 1, 2),
                padding=padding,
                padding_mode='circular'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(5, 5),
                padding=padding,
                padding_mode='circular'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
                padding=padding,
                padding_mode='circular'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=(3, 3),
                padding=padding,
                padding_mode='circular'),
            torch.nn.ReLU(),
        )

        dim_test = torch.unsqueeze(torch.unsqueeze(torch.rand(sample_points, 2), 0), 0)
        in_features = numpy.prod(self._cnn(dim_test).shape)

        linear_modules = []
        for _ in range(4):
            out_features = int(in_features / 2)
            linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            linear_modules.append(torch.nn.ReLU())
            in_features = out_features

        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=500))

        self._linear = torch.nn.Sequential(*linear_modules)

    def forward(self, x):
        cnn_output = self._cnn(x)
        output = self._linear(cnn_output)
        return output
