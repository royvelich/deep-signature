import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
import random


class DeepSignatureDataset(Dataset):
    def __init__(self):
        self._pairs = None
        self._labels = None

    def load_dataset(self, dir_path):
        negative_pairs = numpy.load(file=os.path.normpath(os.path.join(dir_path, 'negative_pairs.npy')), allow_pickle=True)
        positive_pairs = numpy.load(file=os.path.normpath(os.path.join(dir_path, 'positive_pairs.npy')), allow_pickle=True)
        pairs_count = negative_pairs.shape[0]
        full_pairs_count = 2 * pairs_count

        random.shuffle(negative_pairs)
        random.shuffle(positive_pairs)
        self._pairs = numpy.empty((full_pairs_count, negative_pairs.shape[1], negative_pairs.shape[2], negative_pairs.shape[3]))
        self._pairs[::2, :] = negative_pairs
        self._pairs[1::2, :] = positive_pairs
        del negative_pairs
        del positive_pairs

        negaitve_labels = numpy.zeros(pairs_count)
        positive_labels = numpy.ones(pairs_count)
        self._labels = numpy.empty(full_pairs_count)
        self._labels[::2] = negaitve_labels
        self._labels[1::2] = positive_labels

    def __len__(self):
        return self._labels.shape[0]

    def __getitem__(self, idx):
        curves = torch.from_numpy(self._pairs[idx, :]).cuda().double()
        label = torch.from_numpy(numpy.array([self._labels[idx]])).cuda().double()

        first_curve = torch.unsqueeze(curves[0, :, :], dim=0).cuda().double()
        second_curve = torch.unsqueeze(curves[1, :, :], dim=0).cuda().double()

        return {
            'curves_channel1': first_curve,
            'curves_channel2': second_curve,
            'labels': label
        }


class DeepSignatureNet(torch.nn.Module):
    def __init__(self, sample_points):
        super(DeepSignatureNet, self).__init__()

        self._feature_extractor = DeepSignatureNet._create_feature_extractor(
            kernel_size=3)

        dim_test = torch.unsqueeze(torch.unsqueeze(torch.rand(sample_points, 2), 0), 0)

        features = self._feature_extractor(dim_test)
        in_features = numpy.prod(features.shape)

        self._regressor = DeepSignatureNet._create_regressor(layers=4, in_features=in_features)

    def forward(self, x):
        features = self._feature_extractor(x)
        features_reshaped = features.reshape([x.shape[0], -1])
        output = self._regressor(features_reshaped)
        return output

    @staticmethod
    def _create_feature_extractor(kernel_size):
        return torch.nn.Sequential(
            DeepSignatureNet._create_cnn_block(
                in_channels=1,
                out_channels=8,
                kernel_size=kernel_size,
                first_block=True,
                last_block=True),
            # DeepSignatureNet._create_cnn_block(
            #     in_channels=8,
            #     out_channels=8,
            #     kernel_size=kernel_size,
            #     first_block=False,
            #     last_block=False),
            # DeepSignatureNet._create_cnn_block(
            #     in_channels=8,
            #     out_channels=8,
            #     kernel_size=kernel_size,
            #     first_block=False,
            #     last_block=False),
            # DeepSignatureNet._create_cnn_block(
            #     in_channels=8,
            #     out_channels=8,
            #     kernel_size=kernel_size,
            #     first_block=False,
            #     last_block=True)
        )

    @staticmethod
    def _create_regressor(layers, in_features):
        linear_modules = []
        for _ in range(layers):
            out_features = int(0.8 * in_features)
            linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            # linear_modules.append(torch.nn.Linear(in_features=out_features, out_features=out_features))
            linear_modules.append(torch.nn.ReLU())
            in_features = out_features

        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=1))
        return torch.nn.Sequential(*linear_modules)

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
            # torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='zeros'),
            # torch.nn.Dropout2d(),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='zeros'),
            # torch.nn.Dropout2d(0.2),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        ]

        if not last_block:
            layers.append(torch.nn.MaxPool2d(
                kernel_size=(3, 1),
                padding=(1, 0)))

        return torch.nn.Sequential(*layers)


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


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, device='cuda'):
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._device = device
        self._model.to(device)

    def fit(self, dataset, epochs, batch_size, results_base_dir_path, epoch_handler=None, validation_split=0.2, shuffle_dataset=True):
        random_seed = 42

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(numpy.floor(validation_split * dataset_size))

        if shuffle_dataset is True:
            numpy.random.seed(random_seed)
            numpy.random.shuffle(indices)

        train_indices, validation_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(validation_indices)

        train_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        validation_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)

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
        x1, x2, labels = ModelTrainer._extract_batch_data(batch_data)
        out1 = self._model(x1)
        out2 = self._model(x2)
        return self._loss_fn(out1, out2, torch.squeeze(labels))

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
