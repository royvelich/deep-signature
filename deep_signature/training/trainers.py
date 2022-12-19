# python peripherals
import os
import numpy
import itertools
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Protocol, Dict, List, Union

# torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from lightly.loss import NegativeCosineSimilarity

# deep-signature
from deep_signature.core.base import LoggerObject
from deep_signature.core import utils
from deep_signature.manifolds.planar_curves.evaluation import PlanarCurvesShapeMatchingEvaluator


class EpochProcessor(Protocol):
    def __call__(self, epoch_index: int, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        ...


class BatchProcessor(Protocol):
    def __call__(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...


class ModelTrainer(LoggerObject):
    def __init__(
            self,
            results_dir_path: Path,
            model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_dataset: torch.utils.data.Dataset,
            validation_dataset: torch.utils.data.Dataset,
            evaluator: PlanarCurvesShapeMatchingEvaluator,
            epochs: int,
            batch_size: int,
            num_workers: int,
            checkpoint_rate: int,
            device: torch.device):
        super().__init__(log_dir_path=results_dir_path)
        self._results_dir_path = results_dir_path
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._evaluator = evaluator
        self._epochs = epochs
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._checkpoint_rate = checkpoint_rate
        self._model_file_path = self._create_model_file_path(epoch_index=None)
        self._train_dataset_size = len(self._train_dataset)
        self._validation_dataset_size = len(self._validation_dataset)
        self._train_indices = list(range(self._train_dataset_size))
        self._validation_indices = list(range(self._validation_dataset_size))
        self._train_batches_per_epoch = utils.calculate_batches_per_epoch(dataset_size=self._train_dataset_size, batch_size=self._batch_size)
        self._validation_batches_per_epoch = utils.calculate_batches_per_epoch(dataset_size=self._validation_dataset_size, batch_size=self._batch_size)
        self._device = device
        self._model.to(device)

    def train(self):
        self._pre_train()
        self._train()
        self._post_train()

    def _pre_train(self):
        self._logger.info(msg=utils.generate_title_text(text=f'Model Trainer'))

        self._logger.info(msg=utils.generate_bullet_text(text='Training Parameters', indentation=1))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Epochs', value=self._epochs, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Batch Size', value=self._batch_size, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Training Dataset Size', value=self._train_dataset_size, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Validation Dataset Size', value=self._validation_dataset_size, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Training Batches per Epoch', value=self._train_batches_per_epoch, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Validation Batches per Epoch', value=self._validation_batches_per_epoch, indentation=2, padding=30))

        self._logger.info(msg=utils.generate_bullet_text(text='Training Objects', indentation=1))
        self._logger.info(msg=utils.generate_serialized_object_text(text='Model', obj=self._model))
        self._logger.info(msg=utils.generate_serialized_object_text(text='Optimizer', obj=self._optimizer))
        self._logger.info(msg=utils.generate_serialized_object_text(text='Loss Function', obj=self._loss_fn))

        self._results_dir_path.mkdir(parents=True, exist_ok=True)

    def _train(self):
        # best_validation_average_loss = None
        # train_loss_array = numpy.array([])
        # validation_loss_array = numpy.array([])

        train_sampler = SubsetRandomSampler(self._train_indices)
        validation_sampler = SubsetRandomSampler(self._validation_indices)
        train_data_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, sampler=train_sampler, pin_memory=True, drop_last=True, num_workers=self._num_workers)
        validation_data_loader = DataLoader(self._validation_dataset, batch_size=self._batch_size, sampler=validation_sampler, pin_memory=True, drop_last=True, num_workers=self._num_workers)

        # loss_file_path = os.path.normpath(os.path.join(results_dir_path, 'loss.npy'))
        for epoch_index in range(self._epochs):
            self._process_epoch(epoch_index=epoch_index, data_loader=train_data_loader, epoch_name='Train', epoch_processor=self._train_epoch)
            self._process_epoch(epoch_index=epoch_index, data_loader=validation_data_loader, epoch_name='Validation', epoch_processor=self._validation_epoch)

            # self._logger.info(msg=utils.generate_bullet_text(text=f'Training Epoch #{epoch_index}', indentation=1))
            # train_loss = self._train_epoch(epoch_index=epoch_index, data_loader=train_data_loader)
            # train_loss_array = numpy.append(train_loss_array, [numpy.mean(train_loss)])
            #
            # self._logger.info(msg=utils.generate_bullet_text(text=f'Validation Epoch #{epoch_index}', indentation=1))
            # validation_loss = self._validation_epoch(epoch_index=epoch_index, data_loader=validation_data_loader)
            # validation_loss_array = numpy.append(validation_loss_array, [numpy.mean(validation_loss)])
            #
            # if best_validation_average_loss is None:
            #     torch.save(self._model.state_dict(), self._model_file_path)
            #     best_validation_average_loss = numpy.mean(validation_loss)
            # else:
            #     validation_average_loss = numpy.mean(validation_loss)
            #     if validation_average_loss < best_validation_average_loss:
            #         torch.save(self._model.state_dict(), self._model_file_path)
            #         best_validation_average_loss = validation_average_loss
            #
            # if epoch_index % self._checkpoint_rate == 0:
            #     lastest_model_path = self._create_model_file_path(epoch_index=epoch_index)
            #     torch.save(self._model.state_dict(), lastest_model_path)
            #
            # loss_data = {
            #     'train_loss': train_loss_array,
            #     'validation_loss': validation_loss_array,
            # }
            #
            # numpy.save(file=loss_file_path, arr=loss_data, allow_pickle=True)
            #
            # if (self._epochs is not None) and (epoch_index + 1 == self._epochs):
            #     break

    def _post_train(self):
        pass

    # def plot_train_samples(self, batch_size: int, figsize: Tuple[int, int], fontsize: int):
    #     self._plot_samples(dataset=self._train_dataset, indices=self._train_indices, batch_size=batch_size, figsize=figsize, fontsize=fontsize)
    #
    # def plot_validation_samples(self, batch_size: int, figsize: Tuple[int, int], fontsize: int):
    #     self._plot_samples(dataset=self._validation_dataset, indices=self._validation_indices, batch_size=batch_size, figsize=figsize, fontsize=fontsize)

    # def _plot_samples(self, dataset: Dataset, indices: List[int], batch_size: int, figsize: Tuple[int, int], fontsize: int):
    #     sampler = SequentialSampler(indices)
    #     data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
    #
    #     for batch_index, batch_data in enumerate(data_loader, 0):
    #         batch_data_aug = self._preprocess_batch(batch_data)
    #         for sample_index in range(batch_size):
    #             image_count = batch_data_aug.shape[1]
    #             fig, axes = plt.subplots(nrows=1, ncols=image_count, figsize=figsize)
    #             for image_index in range(image_count):
    #                 x = batch_data_aug[sample_index, image_index, :, :, :]
    #                 image = transforms.ToPILImage()(x)
    #                 axes[image_index].imshow(X=image)
    #                 axes[image_index].axis('off')
    #                 axes[image_index].set_title(f'Image #{image_index}', fontsize=fontsize)
    #             plt.show()
    #
    #         if batch_index == 0:
    #             break

    def _process_epoch(self, epoch_index: int, data_loader: DataLoader, epoch_name: str, epoch_processor: EpochProcessor) -> Dict[str, torch.Tensor]:
        self._logger.info(msg=utils.generate_bullet_text(text=f'{epoch_name} Epoch #{epoch_index}', indentation=1))
        return epoch_processor(epoch_index=epoch_index, data_loader=data_loader)

    # def _train_epochs(self, results_dir_path: str):
    #     best_validation_average_loss = None
    #     train_loss_array = numpy.array([])
    #     validation_loss_array = numpy.array([])
    #     train_sampler = SubsetRandomSampler(self._train_indices)
    #     validation_sampler = SubsetRandomSampler(self._validation_indices)
    #     train_data_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, sampler=train_sampler, pin_memory=True, drop_last=False, num_workers=self._num_workers)
    #     validation_data_loader = DataLoader(self._validation_dataset, batch_size=self._batch_size, sampler=validation_sampler, pin_memory=True, drop_last=False, num_workers=self._num_workers)
    #     loss_file_path = os.path.normpath(os.path.join(results_dir_path, 'loss.npy'))
    #     for epoch_index in range(1, self._epochs + 1):
    #
    #         self._process_epoch(epoch_index=epoch_index, data_loader=train_data_loader, epoch_name='Train', epoch_processor=self._train_epoch)
    #         self._process_epoch(epoch_index=epoch_index, data_loader=validation_data_loader, epoch_name='Validation', epoch_processor=self._validation_epoch)
    #
    #         # self._logger.info(msg=utils.generate_bullet_text(text=f'Training Epoch #{epoch_index}', indentation=1))
    #         # train_loss = self._train_epoch(epoch_index=epoch_index, data_loader=train_data_loader)
    #         # train_loss_array = numpy.append(train_loss_array, [numpy.mean(train_loss)])
    #         #
    #         # self._logger.info(msg=utils.generate_bullet_text(text=f'Validation Epoch #{epoch_index}', indentation=1))
    #         # validation_loss = self._validation_epoch(epoch_index=epoch_index, data_loader=validation_data_loader)
    #         # validation_loss_array = numpy.append(validation_loss_array, [numpy.mean(validation_loss)])
    #         #
    #         # if best_validation_average_loss is None:
    #         #     torch.save(self._model.state_dict(), self._model_file_path)
    #         #     best_validation_average_loss = numpy.mean(validation_loss)
    #         # else:
    #         #     validation_average_loss = numpy.mean(validation_loss)
    #         #     if validation_average_loss < best_validation_average_loss:
    #         #         torch.save(self._model.state_dict(), self._model_file_path)
    #         #         best_validation_average_loss = validation_average_loss
    #         #
    #         # if epoch_index % self._checkpoint_rate == 0:
    #         #     lastest_model_path = self._create_model_file_path(epoch_index=epoch_index)
    #         #     torch.save(self._model.state_dict(), lastest_model_path)
    #         #
    #         # loss_data = {
    #         #     'train_loss': train_loss_array,
    #         #     'validation_loss': validation_loss_array,
    #         # }
    #         #
    #         # numpy.save(file=loss_file_path, arr=loss_data, allow_pickle=True)
    #         #
    #         # if (self._epochs is not None) and (epoch_index + 1 == self._epochs):
    #         #     break

    def _train_epoch(self, epoch_index: int, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        self._model.train()
        return self._epoch(epoch_index=epoch_index, data_loader=data_loader, batch_processor=self._train_batch)

    def _validation_epoch(self, epoch_index: int, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        self._model.eval()
        with torch.no_grad():
            return self._epoch(epoch_index=epoch_index, data_loader=data_loader, batch_processor=self._validation_batch)

    def _train_batch(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        def closure():
            self._optimizer.zero_grad()
            evaluated_loss = self._evaluate_loss(batch_data=batch_data)
            evaluated_loss.backward()
            return evaluated_loss

        loss = self._optimizer.step(closure).item()
        return loss

    def _validation_batch(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss = self._evaluate_loss(batch_data=batch_data)
        return loss.item()

    def _evaluate_loss(self, batch_data: Dict[str, torch.Tensor]):
        out_features = self._model(batch_data)
        return self._loss_fn(out_features)

    def _epoch(self, epoch_index: int, data_loader: DataLoader, batch_processor: BatchProcessor) -> Dict[str, torch.Tensor]:
        loss_array = numpy.array([])
        start = timer()
        for batch_index, batch_data in enumerate(data_loader, 0):
            batch_results = batch_processor(batch_data)
            batch_loss = float(batch_results['loss'])
            loss_array = numpy.append(loss_array, [batch_loss])
            end = timer()

            batch_loss_text = utils.generate_batch_loss_text(
                epoch_index=epoch_index,
                batch_index=batch_index,
                batch_loss=batch_loss,
                average_batch_loss=float(numpy.mean(loss_array)),
                index_padding=8,
                loss_padding=25,
                batch_count=len(data_loader),
                batch_duration=end-start,
                indentation=1)

            self._logger.info(msg=batch_loss_text)

            start = timer()

        return {
            'loss': torch.Tensor(loss_array)
        }

    def _create_model_file_path(self, epoch_index: Union[None, int]) -> str:
        if epoch_index is None:
            model_file_name = f'model_{epoch_index}.pt'
        else:
            model_file_name = f'model.pt'

        return os.path.normpath(os.path.join(self._log_dir_path, model_file_name))

    def _create_results_dir_path(self) -> str:
        results_dir_path = os.path.normpath(os.path.join(self._log_dir_path, self._name))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)
        return results_dir_path
