# python peripherals
import os
import numpy
import random
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

# torch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler

# deep_signature
from deep_signature import curve_processing


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


class TupletLoss(torch.nn.Module):
    def __init__(self):
        super(TupletLoss, self).__init__()

    def forward(self, tuplet_embeddings):
        v = tuplet_embeddings[:, 0, :]
        v2 = v.unsqueeze(dim=1)
        v3 = v2 - tuplet_embeddings
        v4 = v3.abs().squeeze(dim=2)
        v5 = v4[:, 1:]
        v6 = v5[:, 0]
        v7 = v6.unsqueeze(dim=1)
        v8 = v7 - v5
        v9 = v8[:, 1:]
        v10 = v9.exp()
        v11 = v10.sum(dim=1)
        v12 = v11 + 1
        v13 = v12.log()
        return v13.mean(dim=0)


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, device='cuda'):
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._device = device
        self._model.to(device)

    def fit(self, dataset, epochs, batch_size, results_base_dir_path, epoch_handler=None, validation_split=0.2, shuffle_dataset=True):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(numpy.floor(validation_split * dataset_size))

        if shuffle_dataset is True:
            numpy.random.shuffle(indices)

        train_indices, validation_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(validation_indices)

        # train_sampler = SequentialSampler(train_indices)
        # validation_sampler = SequentialSampler(validation_indices)

        train_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        validation_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, drop_last=True)

        ModelTrainer._print_training_configuration('Epochs', epochs)
        ModelTrainer._print_training_configuration('Batch size', batch_size)
        ModelTrainer._print_training_configuration('Training dataset length', len(train_indices))
        ModelTrainer._print_training_configuration('Training batches per epoch', int(numpy.ceil(len(train_indices) / batch_size)))
        ModelTrainer._print_training_configuration('Validation dataset length', len(validation_indices))
        ModelTrainer._print_training_configuration('Validation batches per epoch', int(numpy.ceil(len(validation_indices) / batch_size)))

        results_dir_path = os.path.normpath(os.path.join(results_base_dir_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        model_file_path = os.path.normpath(os.path.join(results_dir_path, 'model.pt'))
        results_file_path = os.path.normpath(os.path.join(results_dir_path, 'results.npy'))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)

        print(f' - Start Training:')
        best_validation_average_loss = None
        train_loss_array = numpy.array([])
        validation_loss_array = numpy.array([])
        for epoch_index in range(epochs):
            print(f'    - Training Epoch #{epoch_index}:')
            train_loss = self._train_epoch(epoch_index=epoch_index, data_loader=train_data_loader)
            train_loss_array = numpy.append(train_loss_array, [numpy.mean(train_loss)])
            print(f'    - Validation Epoch #{epoch_index}:')
            validation_loss = self._validation_epoch(epoch_index=epoch_index, data_loader=validation_data_loader)
            validation_loss_array = numpy.append(validation_loss_array, [numpy.mean(validation_loss)])

            if best_validation_average_loss is None:
                torch.save(self._model.state_dict(), model_file_path)
                best_validation_average_loss = numpy.mean(validation_loss)
            else:
                validation_average_loss = numpy.mean(validation_loss)
                if validation_average_loss < best_validation_average_loss:
                    torch.save(self._model.state_dict(), model_file_path)
                    best_validation_average_loss = validation_average_loss

            lastest_model_path = os.path.normpath(os.path.join(results_dir_path, f'model_{epoch_index}.pt'))
            torch.save(self._model.state_dict(), lastest_model_path)

            if epoch_handler is not None:
                epoch_handler(epoch_index)

            results = {
                'train_loss_array': train_loss_array,
                'validation_loss_array': validation_loss_array,
                'epochs': epochs,
                'batch_size': batch_size,
                'model_file_path': model_file_path,
                'results_file_path': results_file_path
            }

            numpy.save(file=results_file_path, arr=results, allow_pickle=True)

        return results

    def _train_epoch(self, epoch_index, data_loader):
        self._model.train()
        return ModelTrainer._epoch(epoch_index=epoch_index, data_loader=data_loader, process_batch_fn=self._train_batch)

    def _validation_epoch(self, epoch_index, data_loader):
        self._model.eval()
        with torch.no_grad():
            return ModelTrainer._epoch(epoch_index=epoch_index, data_loader=data_loader, process_batch_fn=self._validation_batch)

    def _train_batch(self, batch_data):
        self._optimizer.zero_grad()
        loss = self._evaluate_loss(batch_data)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def _validation_batch(self, batch_data):
        loss = self._evaluate_loss(batch_data)
        return loss.item()

    def _evaluate_loss(self, batch_data):
        tuplets = ModelTrainer._extract_batch_data_tuplets(batch_data)
        out = self._model(tuplets)
        return self._loss_fn(out)

    @staticmethod
    def _epoch(epoch_index, data_loader, process_batch_fn):
        loss_array = numpy.array([])
        start = timer()
        for batch_index, batch_data in enumerate(data_loader, 0):
            batch_loss = process_batch_fn(batch_data)
            loss_array = numpy.append(loss_array, [batch_loss])
            end = timer()
            ModelTrainer._print_batch_loss(
                epoch_index=epoch_index,
                batch_index=batch_index,
                batch_loss=batch_loss,
                average_batch_loss=numpy.mean(loss_array),
                fill=' ',
                align='<',
                index_width=8,
                loss_width=25,
                batch_count=len(data_loader),
                batch_duration=end-start)
            start = timer()

        return loss_array

    @staticmethod
    def _print_training_configuration(title, value):
        print(f' - {title:{" "}{"<"}{30}} {value:{" "}{">"}{10}}')

    @staticmethod
    def _print_batch_loss(epoch_index, batch_index, batch_loss, average_batch_loss, fill, align, index_width, loss_width, batch_count, batch_duration):
        print(f'        - [Epoch {epoch_index:{fill}{align}{index_width}} | Batch {batch_index:{fill}{align}{index_width}} / {batch_count}]: Batch Loss = {batch_loss:{fill}{align}{loss_width}}, Avg. Batch Loss = {average_batch_loss:{fill}{align}{loss_width}}, Batch Duration: {batch_duration} sec.')

    @staticmethod
    def _extract_batch_data(batch_data):
        return batch_data['curves_channel1'], batch_data['curves_channel2'], batch_data['labels']

    @staticmethod
    def _extract_batch_data_tuplets(batch_data):
        return batch_data['tuplets']


class DeepSignaturePairsDataset(Dataset):
    def __init__(self):
        self._pairs = None
        self._labels = None

    def load_dataset(self, negative_pairs_dir_path, positive_pairs_dir_path):
        negative_pairs = numpy.load(file=os.path.normpath(os.path.join(negative_pairs_dir_path, 'negative_pairs.npy')), allow_pickle=True)
        positive_pairs = numpy.load(file=os.path.normpath(os.path.join(positive_pairs_dir_path, 'positive_pairs.npy')), allow_pickle=True)
        # pairs_count = numpy.minimum(negative_pairs.shape[0], positive_pairs.shape[0])
        # full_pairs_count = 2 * pairs_count
        full_pairs_count = negative_pairs.shape[0] + positive_pairs.shape[0]

        random.shuffle(negative_pairs)
        random.shuffle(positive_pairs)
        # negative_pairs = negative_pairs[:pairs_count]
        # positive_pairs = positive_pairs[:pairs_count]

        self._pairs = numpy.empty((full_pairs_count, negative_pairs.shape[1], negative_pairs.shape[2], negative_pairs.shape[3]))
        self._pairs[:negative_pairs.shape[0], :] = negative_pairs
        self._pairs[negative_pairs.shape[0]:, :] = positive_pairs
        # del negative_pairs
        # del positive_pairs

        negaitve_labels = numpy.zeros(negative_pairs.shape[0])
        positive_labels = numpy.ones(positive_pairs.shape[0])
        self._labels = numpy.empty(full_pairs_count)
        self._labels[:negative_pairs.shape[0]] = negaitve_labels
        self._labels[negative_pairs.shape[0]:] = positive_labels
        # self._labels[::2] = negaitve_labels
        # self._labels[1::2] = positive_labels

    def __len__(self):
        return self._labels.shape[0]

    def __getitem__(self, idx):
        pairs = self._pairs[idx, :]

        for i in range(2):
            if not curve_processing.is_ccw(curve_sample=pairs[i]):
                pairs[i] = numpy.flip(pairs[i], axis=0)

        for i in range(2):
            radians = curve_processing.calculate_tangent_angle(curve_sample=pairs[i])
            pairs[i] = curve_processing.rotate_curve(curve=pairs[i], radians=radians)

        curves = torch.from_numpy(pairs).cuda().double()
        label = torch.from_numpy(numpy.array([self._labels[idx]])).cuda().double()

        first_curve = torch.unsqueeze(curves[0, :, :], dim=0).cuda().double()
        second_curve = torch.unsqueeze(curves[1, :, :], dim=0).cuda().double()

        return {
            'curves_channel1': first_curve,
            'curves_channel2': second_curve,
            'labels': label
        }


class DeepSignatureTupletsDataset(Dataset):
    def __init__(self):
        self._tuplets = None

    def load_dataset(self, tuplets_dir_path):
        self._tuplets = numpy.load(file=os.path.normpath(os.path.join(tuplets_dir_path, 'tuplets.npy')), allow_pickle=True)

    def __len__(self):
        return self._tuplets.shape[0]

    def __getitem__(self, index):
        # tuplet = torch.unsqueeze(torch.from_numpy(self._tuplets[index].astype('float64')), dim=0).cuda().double()
        tuplet = torch.from_numpy(self._tuplets[index].astype('float64')).cuda().double()

        return {
            'tuplets': tuplet,
        }


class DeepSignatureNet(torch.nn.Module):
    def __init__(self, sample_points, layers):
        super(DeepSignatureNet, self).__init__()
        self._regressor = DeepSignatureNet._create_regressor(layers=layers, in_features=2 * sample_points)

    def forward(self, x):
        features = x.reshape([x.shape[0] * x.shape[1], x.shape[2] * x.shape[3]])
        # features2 = x.reshape([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
        output = self._regressor(features).reshape([x.shape[0], x.shape[1], 1])
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
    def _create_regressor(layers, in_features):
        linear_modules = []
        in_features = 6
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