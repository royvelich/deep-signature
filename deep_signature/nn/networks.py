# python peripherals
import numpy

# torch
import torch


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
        out_features = 100
        p = None
        first_layer = True
        while out_features > 10:
            # if first_layer is True:
            #     weights_init = first_layer_sine_init
            #     first_layer = False
            # else:
            #     weights_init = sine_init

            linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
            # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=sine_init))
            in_features = out_features
            out_features = int(out_features / 2)

        # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=in_features, out_features=20, p=p, use_batch_norm=True, weights_init=None))
        # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=20, out_features=20, p=p, use_batch_norm=True, weights_init=None))
        # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=20, out_features=20, p=p, use_batch_norm=True, weights_init=None))
        # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=20, out_features=20, p=p, use_batch_norm=True, weights_init=None))
        # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=20, out_features=20, p=p, use_batch_norm=True, weights_init=None))
        # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=20, out_features=20, p=p, use_batch_norm=True, weights_init=None))
        # linear_modules.append(torch.nn.Linear(in_features=20, out_features=1))

        # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True, weights_init=None))
        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=1))

        # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=in_features, out_features=20, p=p, use_batch_norm=False))
        # # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=20, out_features=20, p=p, use_batch_norm=False))
        # # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=20, out_features=20, p=p, use_batch_norm=False))
        # linear_modules.extend(DeepSignatureCurvatureNet._create_hidden_layer(in_features=20, out_features=20, p=p, use_batch_norm=False))
        # linear_modules.append(torch.nn.Linear(in_features=20, out_features=1))

        return torch.nn.Sequential(*linear_modules)

    @staticmethod
    def _create_hidden_layer(in_features, out_features, p=None, use_batch_norm=False, weights_init=None):
        linear_modules = []
        linear_module = torch.nn.Linear(in_features=in_features, out_features=out_features)

        # if weights_init is not None:
        #     weights_init(linear_module)

        linear_modules.append(linear_module)

        if use_batch_norm:
            linear_modules.append(torch.nn.BatchNorm1d(out_features))

        linear_modules.append(Sine())

        if p is not None:
            linear_modules.append(torch.nn.Dropout(p))

        return linear_modules


class DeepSignatureArcLengthNet(torch.nn.Module):
    def __init__(self, sample_points):
        super(DeepSignatureArcLengthNet, self).__init__()
        self._regressor1 = DeepSignatureArcLengthNet._create_regressor(in_features=2*sample_points)
        self._regressor2 = DeepSignatureArcLengthNet._create_regressor(in_features=2*(sample_points + 1))

    @staticmethod
    def _reshape_features(inputs):
        features_list = []
        for input in inputs:
            features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
            features_list.append(features)
        return features_list

    @staticmethod
    def _process_features(regressor, inputs, features_list):
        outputs = []
        for input, features in zip(inputs, features_list):
            output = regressor(features).abs().reshape([input.shape[0], input.shape[1], 1])
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def _process_input(regressor, input):
        for [short, long] in input:
            features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
            output = regressor(features).abs().reshape([input.shape[0], input.shape[1], 1])
            return output

    def forward(self, input):
        output = []
        keys = list(input.keys())
        for orig_short_key, orig_long_key, trans_short_key, trans_long_key in zip(keys[0::4], keys[1::4], keys[2::4], keys[3::4]):
            orig_short = input[orig_short_key]
            orig_long = input[orig_long_key]
            trans_short = input[trans_short_key]
            trans_long = input[trans_long_key]

            features_orig_short = orig_short.reshape([orig_short.shape[0] * orig_short.shape[1], orig_short.shape[2] * orig_short.shape[3]])
            features_orig_long = orig_long.reshape([orig_long.shape[0] * orig_long.shape[1], orig_long.shape[2] * orig_long.shape[3]])
            output_orig_short = self._regressor1(features_orig_short).abs().reshape([orig_short.shape[0], orig_short.shape[1], 1])
            output_orig_long = self._regressor2(features_orig_long).abs().reshape([orig_long.shape[0], orig_long.shape[1], 1])

            features_trans_short = trans_short.reshape([trans_short.shape[0] * trans_short.shape[1], trans_short.shape[2] * trans_short.shape[3]])
            features_trans_long = trans_long.reshape([trans_long.shape[0] * trans_long.shape[1], trans_long.shape[2] * trans_long.shape[3]])
            output_trans_short = self._regressor1(features_trans_short).abs().reshape([trans_short.shape[0], trans_short.shape[1], 1])
            output_trans_long = self._regressor2(features_trans_long).abs().reshape([trans_long.shape[0], trans_long.shape[1], 1])

            output.append([output_orig_short, output_orig_long, output_trans_short, output_trans_long])
        return output

    def evaluate_regressor1(self, input):
        features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
        output = self._regressor1(features).abs().reshape([input.shape[0], input.shape[1], 1])
        return output

    def evaluate_regressor2(self, input):
        features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
        output = self._regressor2(features).abs().reshape([input.shape[0], input.shape[1], 1])
        return output

    @staticmethod
    def _create_regressor(in_features):
        linear_modules = []
        in_features = in_features
        out_features = 100
        p = None
        while out_features > 10:
            linear_modules.extend(DeepSignatureArcLengthNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True))
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

        # linear_modules.append(torch.nn.GELU())
        linear_modules.append(Sine())

        if p is not None:
            linear_modules.append(torch.nn.Dropout(p))
        return linear_modules
