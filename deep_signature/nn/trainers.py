# python peripherals
import os
import numpy
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

# torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


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
        loss = self._evaluate_loss(batch_data=batch_data)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def _validation_batch(self, batch_data):
        loss = self._evaluate_loss(batch_data=batch_data)
        return loss.item()

    def _evaluate_loss(self, batch_data):
        output = self._model(batch_data['input'])
        return self._loss_fn(output=output, batch_data=batch_data)

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
