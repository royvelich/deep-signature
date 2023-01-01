# python peripherals
import numpy
from pathlib import Path
from timeit import default_timer as timer
from typing import Protocol, Dict, Union

import pandas
# wandb
import wandb

# torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# deep-signature
from deep_signature.core.base import LoggerObject
from deep_signature.core import utils
from deep_signature.manifolds.planar_curves.evaluation import PlanarCurvesShapeMatchingEvaluator, PlanarCurvesQualitativeEvaluator


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
            shape_matching_evaluator: PlanarCurvesShapeMatchingEvaluator,
            qualitative_evaluator: PlanarCurvesQualitativeEvaluator,
            epochs: int,
            training_batch_size: int,
            validation_batch_size: int,
            num_workers: int,
            checkpoint_rate: int,
            device: torch.device,
            log_wandb: bool):
        super().__init__(log_dir_path=results_dir_path)
        self._results_dir_path = results_dir_path
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._shape_matching_evaluator = shape_matching_evaluator
        self._qualitative_evaluator = qualitative_evaluator
        self._epochs = epochs
        self._training_batch_size = training_batch_size
        self._validation_batch_size = validation_batch_size
        self._num_workers = num_workers
        self._checkpoint_rate = checkpoint_rate
        self._models_dir_path = self._results_dir_path / 'models'
        self._models_dir_path.mkdir(parents=True, exist_ok=True)
        self._model_file_path = self._create_model_file_path(epoch_index=None)
        self._train_dataset_size = len(self._train_dataset)
        self._validation_dataset_size = len(self._validation_dataset)
        self._train_indices = list(range(self._train_dataset_size))
        self._validation_indices = list(range(self._validation_dataset_size))
        self._train_batches_per_epoch = utils.calculate_batches_per_epoch(dataset_size=self._train_dataset_size, batch_size=self._training_batch_size)
        self._validation_batches_per_epoch = utils.calculate_batches_per_epoch(dataset_size=self._validation_dataset_size, batch_size=self._validation_batch_size)
        self._device = device
        self._log_wandb = log_wandb

    def train(self):
        self._pre_train()
        self._train()
        self._post_train()

    def _pre_train(self):
        self._logger.info(msg=utils.generate_title_text(text=f'Model Trainer'))

        self._logger.info(msg=utils.generate_bullet_text(text='Training Parameters', indentation=1))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Epochs', value=self._epochs, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Training Batch Size', value=self._training_batch_size, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Validation Batch Size', value=self._validation_batch_size, indentation=2, padding=30))
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
        accumulated_shape_matching_df = pandas.DataFrame()
        best_shape_matching_df = pandas.DataFrame()
        best_evaluation_score = 0.0
        train_sampler = SubsetRandomSampler(self._train_indices)
        validation_sampler = SubsetRandomSampler(self._validation_indices)
        train_data_loader = DataLoader(self._train_dataset, batch_size=self._training_batch_size, sampler=train_sampler, pin_memory=True, drop_last=True, num_workers=self._num_workers)
        validation_data_loader = DataLoader(self._validation_dataset, batch_size=self._validation_batch_size, sampler=validation_sampler, pin_memory=True, drop_last=True, num_workers=self._num_workers)

        self._shape_matching_evaluator.start()
        self._shape_matching_evaluator.join()

        # for epoch_index in range(self._epochs):
        #     self._model.cuda()
        #     train_epoch_results = self._process_epoch(epoch_index=epoch_index, data_loader=train_data_loader, epoch_name='Train', epoch_processor=self._train_epoch)
        #     validation_epoch_results = self._process_epoch(epoch_index=epoch_index, data_loader=validation_data_loader, epoch_name='Validation', epoch_processor=self._validation_epoch)
        #     latest_model_file_path = self._create_model_file_path(epoch_index=epoch_index)
        #     torch.save(self._model.state_dict(), latest_model_file_path)
        #     images = self._qualitative_evaluator.evaluate_curves()
        #     log_dict = {
        #         'train_loss': train_epoch_results['loss'],
        #         'validation_loss': validation_epoch_results['loss'],
        #         'images': [wandb.Image(image) for image in images],
        #         'epoch': epoch_index
        #     }
        #
        #     if epoch_index % self._checkpoint_rate == 0:
        #         # self._model.cpu()
        #         self._shape_matching_evaluator.start()
        #         self._shape_matching_evaluator.join()
        #         evaluation_score = self._shape_matching_evaluator.get_evaluation_score()
        #         print(f'evaluation_score: {evaluation_score}')
        #         shape_matching_df = self._shape_matching_evaluator.shape_matching_df.copy()
        #         shape_matching_df['epoch'] = epoch_index
        #         accumulated_shape_matching_df = pandas.concat([accumulated_shape_matching_df, shape_matching_df])
        #
        #         if evaluation_score > best_evaluation_score:
        #             best_shape_matching_df = shape_matching_df
        #             best_evaluation_score = evaluation_score
        #             torch.save(self._model.state_dict(), self._model_file_path)
        #
        #         evaluation_log_dict = {
        #             'best_evaluation_score': best_evaluation_score,
        #             'evaluation_score': evaluation_score,
        #             'accumulated_shape_matching_df': wandb.Table(dataframe=accumulated_shape_matching_df),
        #             'best_shape_matching_df': wandb.Table(dataframe=best_shape_matching_df),
        #         }
        #
        #         log_dict = log_dict | evaluation_log_dict
        #
        #     if self._log_wandb is True:
        #         wandb.log(log_dict)

    def _post_train(self):
        pass

    def _process_epoch(self, epoch_index: int, data_loader: DataLoader, epoch_name: str, epoch_processor: EpochProcessor) -> Dict[str, torch.Tensor]:
        self._logger.info(msg=utils.generate_bullet_text(text=utils.generate_epoch_text(epoch_index=epoch_index, epoch_name=epoch_name), indentation=1))
        return epoch_processor(epoch_index=epoch_index, data_loader=data_loader)

    def _train_epoch(self, epoch_index: int, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        self._model.train()
        return self._epoch(epoch_index=epoch_index, data_loader=data_loader, batch_processor=self._train_batch)

    def _validation_epoch(self, epoch_index: int, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        self._model.eval()
        with torch.no_grad():
            return self._epoch(epoch_index=epoch_index, data_loader=data_loader, batch_processor=self._validation_batch)

    def _train_batch(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        def closure() -> float:
            self._optimizer.zero_grad()
            evaluated_loss = self._evaluate_loss(batch_data=batch_data)
            evaluated_loss.backward()
            return float(evaluated_loss)

        loss = self._optimizer.step(closure=closure)
        return {
            'loss': torch.tensor(loss)
        }

    def _validation_batch(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss = self._evaluate_loss(batch_data=batch_data)
        return {
            'loss': loss
        }

    def _evaluate_loss(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        out_features = self._model(batch_data)
        return self._loss_fn(out_features)

    def _epoch(self, epoch_index: int, data_loader: DataLoader, batch_processor: BatchProcessor) -> Dict[str, torch.Tensor]:
        loss_array = numpy.array([])
        start = timer()
        epoch_loss = 0
        for batch_index, batch_data in enumerate(data_loader, 0):
            batch_data = batch_data.cuda()
            batch_results = batch_processor(batch_data)
            batch_loss = float(batch_results['loss'])
            loss_array = numpy.append(loss_array, [batch_loss])
            end = timer()

            epoch_loss = float(numpy.mean(loss_array))
            batch_loss_text = utils.generate_batch_loss_text(
                epoch_index=epoch_index,
                batch_index=batch_index,
                batch_loss=batch_loss,
                average_batch_loss=epoch_loss,
                index_padding=8,
                loss_padding=25,
                batch_count=len(data_loader),
                batch_duration=end-start,
                indentation=1)
            self._logger.info(msg=batch_loss_text)

            start = timer()

        return {
            'loss': torch.tensor(epoch_loss)
        }

    def _create_model_file_path(self, epoch_index: Union[None, int]) -> Path:
        if epoch_index is None:
            model_file_name = f'model_{epoch_index}.pt'
        else:
            model_file_name = f'model.pt'

        return self._models_dir_path / model_file_name
