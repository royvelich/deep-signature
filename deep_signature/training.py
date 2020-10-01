import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy
from datetime import datetime
from pathlib import Path

class DeepSignatureDataset(Dataset):
    def __init__(self, dir_path):
        self._dir_path = dir_path
        self._metadata = numpy.load(file=os.path.normpath(os.path.join(dir_path, 'metadata.npy')), allow_pickle=True)
        self._metadata = self._metadata.item()
        self._pairs = self._metadata['pairs']

    @property
    def sample_points(self):
        return self._metadata['sample_points']

    def __len__(self):
        return self._pairs.shape[0]

    def __getitem__(self, idx):
        pair = self._pairs[idx]

        label = pair[0]
        curve1_descriptor = pair[1:4]
        curve2_descriptor = pair[4:8]

        curve1_sample = DeepSignatureDataset._load_curve_sample(dir_path=self._dir_path, curve_descriptor=curve1_descriptor)
        curve2_sample = DeepSignatureDataset._load_curve_sample(dir_path=self._dir_path, curve_descriptor=curve2_descriptor)

        curve1_tensor = torch.unsqueeze(torch.from_numpy(curve1_sample), 0).cuda().float()
        curve2_tensor = torch.unsqueeze(torch.from_numpy(curve2_sample), 0).cuda().float()
        labels_tensor = torch.squeeze(torch.from_numpy(numpy.array([label])).cuda().float(), 0)

        return {
            'curves': [curve1_tensor, curve2_tensor],
            'labels': labels_tensor
        }

    @staticmethod
    def _build_curve_path(dir_path, curve_descriptor):
        return os.path.normpath(os.path.join(dir_path, f'{curve_descriptor[0]}/{curve_descriptor[1]}/{curve_descriptor[2]}', 'sample.npy'))

    @staticmethod
    def _load_curve_sample(dir_path, curve_descriptor):
        return numpy.load(file=DeepSignatureDataset._build_curve_path(dir_path, curve_descriptor), allow_pickle=True)


class DeepSignatureNet(torch.nn.Module):
    def __init__(self, sample_points):
        super(DeepSignatureNet, self).__init__()

        self._feature_extractor = DeepSignatureNet._create_feature_extractor(
            kernel_size=5)

        dim_test = torch.unsqueeze(torch.unsqueeze(torch.rand(sample_points, 2), 0), 0)

        features = self._feature_extractor(dim_test)
        in_features = numpy.prod(features.shape)

        self._regressor = DeepSignatureNet._create_regressor(
            layers=3,
            in_features=in_features,
            sample_points=sample_points)

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
                out_channels=64,
                kernel_size=kernel_size,
                first_block=True),
            DeepSignatureNet._create_cnn_block(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel_size,
                first_block=False),
            DeepSignatureNet._create_cnn_block(
                in_channels=128,
                out_channels=256,
                kernel_size=kernel_size,
                first_block=False)
        )

    @staticmethod
    def _create_regressor(layers, in_features, sample_points):
        linear_modules = []
        for _ in range(layers):
            out_features = int(in_features / 2)
            linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            linear_modules.append(torch.nn.ReLU())
            in_features = out_features

        linear_modules.append(torch.nn.Linear(in_features=in_features, out_features=sample_points))

        return torch.nn.Sequential(*linear_modules)

    @staticmethod
    def _create_cnn_block(in_channels, out_channels, kernel_size, first_block):
        padding = int(kernel_size / 2)
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 2) if first_block is True else (kernel_size, 1),
                padding=(padding, 0),
                padding_mode='circular'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='circular'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='circular'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                padding_mode='circular'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(3, 1),
                padding=(1, 0))
        )


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

    def fit(self, dataset, epochs, batch_size, results_base_dir_path, validation_split=0.2, shuffle_dataset=True):
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
            train_loss_array = numpy.append(train_loss_array, [train_loss])
            print(f'    - Validation Epoch #{epoch_index}:')
            validation_loss = self._validation_epoch(epoch_index=epoch_index, data_loader=validation_data_loader)
            validation_loss_array = numpy.append(validation_loss_array, [validation_loss])

            if best_validation_average_loss is None:
                torch.save(self._model.state_dict(), model_file_path)
                best_validation_average_loss = numpy.mean(validation_loss)
            else:
                validation_average_loss = numpy.mean(validation_loss)
                if validation_average_loss < best_validation_average_loss:
                    torch.save(self._model.state_dict(), model_file_path)
                    best_validation_average_loss = validation_average_loss

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
        return self._loss_fn(out1, out2, labels)

    @staticmethod
    def _epoch(epoch_index, data_loader, process_batch_fn):
        loss_array = numpy.array([])
        for batch_index, batch_data in enumerate(data_loader, 0):
            batch_loss = process_batch_fn(batch_data)
            loss_array = numpy.append(loss_array, [batch_loss])
            ModelTrainer._print_batch_loss(
                epoch_index=epoch_index,
                batch_index=batch_index,
                batch_loss=batch_loss,
                average_batch_loss=numpy.mean(loss_array),
                fill=' ',
                align='<',
                index_width=8,
                loss_width=25)

        return loss_array

    @staticmethod
    def _print_training_configuration(title, value):
        print(f' - {title:{" "}{"<"}{30}} {value:{" "}{">"}{10}}')

    @staticmethod
    def _print_batch_loss(epoch_index, batch_index, batch_loss, average_batch_loss, fill, align, index_width, loss_width):
        print(f'        - [Epoch {epoch_index:{fill}{align}{index_width}} | Batch {batch_index:{fill}{align}{index_width}}]: Batch Loss = {batch_loss:{fill}{align}{loss_width}}, Avg. Batch Loss = {average_batch_loss:{fill}{align}{loss_width}}')

    @staticmethod
    def _extract_batch_data(batch_data):
        return batch_data['curves'][0], batch_data['curves'][1], batch_data['labels']
