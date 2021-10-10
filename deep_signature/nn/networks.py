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
        self._regressor = DeepSignatureArcLengthNet._create_regressor(in_features=2 * sample_points)

    def forward(self, input):
        features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
        output = self._regressor(features).reshape([input.shape[0], input.shape[1], 1])
        return output.abs()

    @staticmethod
    def _create_regressor(in_features):
        linear_modules = []
        in_features = in_features
        out_features = 250
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
        # linear_modules.append(Sine())

        if p is not None:
            linear_modules.append(torch.nn.Dropout(p))
        return linear_modules

# class DeepSignatureArcLengthNet(torch.nn.Module):
#     def __init__(self, sample_points):
#         super(DeepSignatureArcLengthNet, self).__init__()
#         self._regressor1 = DeepSignatureArcLengthNet._create_regressor(in_features=2*sample_points)
#         self._regressor2 = DeepSignatureArcLengthNet._create_regressor(in_features=2*(sample_points + 1))
#
#     @staticmethod
#     def _reshape_features(inputs):
#         features_list = []
#         for input in inputs:
#             features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
#             features_list.append(features)
#         return features_list
#
#     @staticmethod
#     def _process_features(regressor, inputs, features_list):
#         outputs = []
#         for input, features in zip(inputs, features_list):
#             output = regressor(features).reshape([input.shape[0], input.shape[1], 1]).abs()
#             outputs.append(output)
#         return torch.cat(outputs, dim=0)
#
#     @staticmethod
#     def _process_input(regressor, input):
#         features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
#         output = regressor(features).reshape([input.shape[0], input.shape[1], 1]).abs()
#         return output
#
#     def forward(self, input1, input2, input3, input4):
#         # inputs1 = [input1, input2]
#         # inputs2 = [input3, input4]
#         # features_list1 = DeepSignatureArcLengthNet._reshape_features(inputs=inputs1)
#         # features_list2 = DeepSignatureArcLengthNet._reshape_features(inputs=inputs2)
#         # output1 = DeepSignatureArcLengthNet._process_features(regressor=self._regressor1, inputs=inputs1, features_list=features_list1)
#         # output2 = DeepSignatureArcLengthNet._process_features(regressor=self._regressor2, inputs=inputs2, features_list=features_list2)
#         #
#         # return torch.cat([output1, output2], dim=1)
#         output1 = DeepSignatureArcLengthNet._process_input(regressor=self._regressor1, input=input1)
#         output2 = DeepSignatureArcLengthNet._process_input(regressor=self._regressor1, input=input2)
#         output3 = DeepSignatureArcLengthNet._process_input(regressor=self._regressor2, input=input3)
#         output4 = DeepSignatureArcLengthNet._process_input(regressor=self._regressor2, input=input4)
#         return [output1, output2, output3, output4]
#
#     def evaluate_regressor1(self, input):
#         features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
#         output = self._regressor1(features).reshape([input.shape[0], input.shape[1], 1])
#         return output
#
#     def evaluate_regressor2(self, input):
#         features = input.reshape([input.shape[0] * input.shape[1], input.shape[2] * input.shape[3]])
#         output = self._regressor2(features).reshape([input.shape[0], input.shape[1], 1])
#         return output
#
#     @staticmethod
#     def _create_regressor(in_features):
#         linear_modules = []
#         in_features = in_features
#         out_features = 100
#         p = None
#         while out_features > 10:
#             linear_modules.extend(DeepSignatureArcLengthNet._create_hidden_layer(in_features=in_features, out_features=out_features, p=p, use_batch_norm=True))
#             # linear_modules.extend(DeepSignatureArcLengthNet._create_hidden_layer(in_features=out_features, out_features=out_features, p=p, use_batch_norm=True))
#             in_features = out_features
#             out_features = int(out_features / 2)
#
#         linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=1))
#         return torch.nn.Sequential(*linear_modules)
#
#     @staticmethod
#     def _create_hidden_layer(in_features, out_features, p=None, use_batch_norm=False):
#         linear_modules = []
#         linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
#         if use_batch_norm:
#             linear_modules.append(torch.nn.BatchNorm1d(out_features))
#
#         # linear_modules.append(torch.nn.GELU())
#         linear_modules.append(Sine())
#
#         if p is not None:
#             linear_modules.append(torch.nn.Dropout(p))
#         return linear_modules
