# python peripherals
import os
import numpy
import itertools
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
import json
import pickle

# torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from lightly.loss import NegativeCosineSimilarity

# deep-signature
from utils import settings


class ModelTrainer:
    def __init__(self, model, loss_function, optimizer, world_size, rank, device):
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._world_size = world_size
        self._rank = rank
        self._device = device
        self._model = model
        self._criterion = NegativeCosineSimilarity()

    def fit(self, train_dataset, validation_dataset, epochs, train_batch_size, validation_batch_size, results_base_dir_path, epoch_handler=None, validation_split=None, shuffle_dataset=True):
        dataset_size = None
        train_dataset_size = None
        validation_dataset_size = None
        if validation_split is not None:
            dataset_size = len(train_dataset)
            indices = list(range(dataset_size))
            split = int(numpy.floor(validation_split * dataset_size))
            train_indices, validation_indices = indices[split:], indices[:split]
            actual_train_dataset = train_dataset
            actual_validation_dataset = train_dataset
        else:
            train_dataset_size = len(train_dataset)
            validation_dataset_size = len(validation_dataset)
            train_indices = list(range(train_dataset_size))
            validation_indices = list(range(validation_dataset_size))
            actual_train_dataset = train_dataset
            actual_validation_dataset = validation_dataset

        # if shuffle_dataset is True:
        train_sampler = DistributedSampler(actual_train_dataset, shuffle=True)
        # train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(validation_indices)
        # else:
        # train_sampler = DistributedSampler(dataset=actual_train_dataset, shuffle=True)
        # validation_sampler = SequentialSampler(validation_indices)


        # validation_sampler = SubsetRandomSampler(validation_indices)
        # validation_sampler = SubsetRandomSampler(validation_indices)

        batch_size_per_gpu = train_batch_size // self._world_size

        print(f'Rank: {self._rank}, train_batch_size: {train_batch_size}')
        print(f'Rank: {self._rank}, batch_size_per_gpu: {batch_size_per_gpu}')

        train_data_loader = DataLoader(actual_train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, pin_memory=True, drop_last=True, num_workers=0)
        validation_data_loader = DataLoader(actual_validation_dataset, batch_size=validation_batch_size, sampler=validation_sampler, drop_last=True, num_workers=0)

        epochs_text = epochs if epochs is not None else 'infinite'

        # if self._rank == 0:
        print('')
        ModelTrainer._print_training_configuration('Epochs', epochs_text)
        ModelTrainer._print_training_configuration('Train Batch size', train_batch_size)
        ModelTrainer._print_training_configuration('Validation Batch size', validation_batch_size)
        ModelTrainer._print_training_configuration('Training dataset length', len(train_indices))
        ModelTrainer._print_training_configuration('Training batches per epoch', int(numpy.ceil(len(train_indices) / train_batch_size)))
        ModelTrainer._print_training_configuration('Validation dataset length', len(validation_indices))
        ModelTrainer._print_training_configuration('Validation batches per epoch', int(numpy.ceil(len(validation_indices) / validation_batch_size)))

        results_dir_path = os.path.normpath(os.path.join(results_base_dir_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        model_file_path = os.path.normpath(os.path.join(results_dir_path, 'model.pt'))
        module_file_path = os.path.normpath(os.path.join(results_dir_path, 'module.pt'))
        results_file_path = os.path.normpath(os.path.join(results_dir_path, 'results.npy'))
        model_architecture_file_path = os.path.normpath(os.path.join(results_dir_path, 'model_arch.txt'))
        loss_functions_file_path = os.path.normpath(os.path.join(results_dir_path, 'loss_functions.txt'))
        optimizer_file_path = os.path.normpath(os.path.join(results_dir_path, 'optimizer.txt'))
        trainer_data_file_path = os.path.normpath(os.path.join(results_dir_path, 'trainer_data.txt'))
        settings_data_file_path = os.path.normpath(os.path.join(results_dir_path, 'settings.txt'))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)

        # if self._rank == 0:
        with open(model_architecture_file_path, "w") as text_file:
            text_file.write(str(self._model))

        with open(loss_functions_file_path, "w") as text_file:
            text_file.write(str(self._loss_function))

        with open(optimizer_file_path, "w") as text_file:
            text_file.write(str(self._optimizer))

        with open(trainer_data_file_path, "w") as text_file:
            text_file.write(f'train_batch_size: {train_batch_size}\n')
            text_file.write(f'validation_batch_size: {validation_batch_size}\n')
            text_file.write(f'epochs: {epochs_text}\n')
            text_file.write(f'results_dir_path: {results_dir_path}\n')
            if validation_split is not None:
                text_file.write(f'validation_split: {validation_split}\n')
                text_file.write(f'dataset_size: {dataset_size}\n')
            else:
                text_file.write(f'train_dataset_size: {train_dataset_size}\n')
                text_file.write(f'validation_dataset_size: {validation_dataset_size}\n')

        settings_dict = {key: value for key, value in settings.__dict__.items() if isinstance(value, str) or isinstance(value, int) or isinstance(value, float)}
        with open(settings_data_file_path, "w") as text_file:
            for key, value in settings_dict.items():
                text_file.write(f'{key}: {value}\n')

        print('')
        print(f' - [Rank {self._rank}] Start Training:')

        results = None
        best_validation_average_loss = None
        train_loss_array = numpy.array([])
        validation_loss_array = numpy.array([])
        for epoch_index in itertools.count():
            train_sampler.set_epoch(epoch_index)
            # if self._rank == 0:
            print(f'    - [Rank {self._rank}] Training Epoch #{epoch_index+1}:')
            train_loss = self._train_epoch(epoch_index=epoch_index, data_loader=train_data_loader)
            train_loss_array = numpy.append(train_loss_array, [numpy.mean(train_loss)])

            if self._rank == 0:
                print(f'    - Validation Epoch #{epoch_index+1}:')
                validation_loss = self._validation_epoch(epoch_index=epoch_index, data_loader=validation_data_loader)
                validation_loss_array = numpy.append(validation_loss_array, [numpy.mean(validation_loss)])

                if best_validation_average_loss is None:
                    torch.save(self._model.state_dict(), model_file_path)
                    best_validation_average_loss = numpy.mean(validation_loss)
                else:
                    validation_average_loss = numpy.mean(validation_loss)
                    if validation_average_loss < best_validation_average_loss:
                        torch.save(self._model.state_dict(), model_file_path)
                        if isinstance(self._model, torch.nn.parallel.DistributedDataParallel):
                            torch.save(self._model.module.state_dict(), module_file_path)
                        best_validation_average_loss = validation_average_loss

                lastest_model_path = os.path.normpath(os.path.join(results_dir_path, f'model_{epoch_index}.pt'))
                torch.save(self._model.state_dict(), lastest_model_path)

                if isinstance(self._model, torch.nn.parallel.DistributedDataParallel):
                    lastest_module_path = os.path.normpath(os.path.join(results_dir_path, f'module_{epoch_index}.pt'))
                    # lastest_module_path = os.path.normpath(os.path.join(results_dir_path, f'module.pt'))
                    torch.save(self._model.module.state_dict(), lastest_module_path)

                if epoch_handler is not None:
                    epoch_handler(epoch_index)

                results = {
                    'train_loss_array': train_loss_array,
                    'validation_loss_array': validation_loss_array,
                    'epochs': epochs_text,
                    'train_batch_size': train_batch_size,
                    'validation_batch_size': validation_batch_size,
                    'model_file_path': model_file_path,
                    'module_file_path': module_file_path,
                    'results_file_path': results_file_path
                }

                numpy.save(file=results_file_path, arr=results, allow_pickle=True)

                if (epochs is not None) and (epoch_index + 1 == epochs):
                    break

        return results

    def _train_epoch(self, epoch_index, data_loader):
        self._model.train()
        return ModelTrainer._epoch(epoch_index=epoch_index, data_loader=data_loader, process_batch_fn=self._train_batch, rank=self._rank)

    def _validation_epoch(self, epoch_index, data_loader):
        self._model.eval()
        with torch.no_grad():
            return ModelTrainer._epoch(epoch_index=epoch_index, data_loader=data_loader, process_batch_fn=self._validation_batch, rank=self._rank)

    def _train_batch(self, batch_data):
        def closure():
            self._optimizer.zero_grad()
            loss = self._evaluate_loss(batch_data=batch_data.to(self._device, non_blocking=True))
            loss.backward()
            return loss

        final_loss = self._optimizer.step(closure)
        return final_loss.item()

        # self._optimizer.zero_grad()
        # loss = self._evaluate_loss(batch_data=batch_data.to(self._device, non_blocking=True))
        # loss.backward()
        # self._optimizer.step()
        # return loss.item()

    def _validation_batch(self, batch_data):
        loss = self._evaluate_loss(batch_data=batch_data.to(self._device, non_blocking=True))
        return loss.item()

    def _evaluate_loss(self, batch_data):
        output = self._model(batch_data)
        return self._loss_function(output=output)

    @staticmethod
    def _epoch(epoch_index, data_loader, process_batch_fn, rank):
        loss_array = numpy.array([])
        start = timer()
        for batch_index, batch_data in enumerate(data_loader, 0):
            batch_loss = process_batch_fn(batch_data)
            loss_array = numpy.append(loss_array, [batch_loss])
            end = timer()
            # if rank == 0:
            # print(f'Rank {rank}: batch_data.shape: {batch_data.shape}')
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
        print(f'        - [Epoch {epoch_index+1:{fill}{align}{index_width}} | Batch {batch_index+1:{fill}{align}{index_width}} / {batch_count}]: Batch Loss = {batch_loss:{fill}{align}{loss_width}}, Avg. Batch Loss = {average_batch_loss:{fill}{align}{loss_width}}, Batch Duration: {batch_duration} sec.')