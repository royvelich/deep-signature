# python peripherals
import os
from pathlib import Path
from typing import Optional, Callable, List

# torch
import torch

# wandb
import wandb

# deep-signature
from applets.core.utils import init_app, AppArgumentParser
from deep_signature.manifolds.planar_curves.groups import Group
from deep_signature.core import utils
from deep_signature.manifolds.planar_curves.implementation import PlanarCurvesManager
from deep_signature.training.datasets import CurveNeighborhoodTupletsDataset
from deep_signature.training.networks import DeepSignaturesNet
from deep_signature.training.activations import Sine
from deep_signature.training.losses import DifferentialInvariantsLoss
from deep_signature.training.trainers import ModelTrainer
from deep_signature.manifolds.planar_curves.evaluation import PlanarCurvesSignatureHausdorffComparator, PlanarCurvesNeuralSignatureCalculator
from deep_signature.manifolds.planar_curves.evaluation import PlanarCurvesShapeMatchingEvaluator, PlanarCurvesSignatureQualitativeEvaluator
from deep_signature.core.parallel_processing import GetItemPolicy
from deep_signature.training.losses import InvarianceLoss


class TrainSignatureArgumentParser(AppArgumentParser):
    training_curves_file_path: Path
    validation_curves_file_path: Path
    test_curves_file_path: Path
    training_circles_file_path: Path
    test_circles_file_path: Path
    evaluation_benchmark_dir_path: Path
    wandb_dir: Path
    wandb_cache_dir: Path
    wandb_config_dir: Path
    training_num_workers: int
    validation_num_workers: int
    evaluation_num_workers: int
    count: Optional[int] = None
    sweep_id: Optional[str] = None
    group_name: Optional[str] = None
    supporting_points_count: Optional[int] = None
    negative_examples_count: Optional[int] = None
    min_sampling_ratio: Optional[float] = None
    max_sampling_ratio: Optional[float] = None
    min_multimodality: Optional[int] = None
    max_multimodality: Optional[int] = None
    min_negative_example_offset: Optional[int] = None
    max_negative_example_offset: Optional[int] = None
    training_dataset_size: Optional[int] = None
    validation_dataset_size: Optional[int] = None
    validation_items_queue_maxsize: Optional[int] = None
    training_items_queue_maxsize: Optional[int] = None
    training_items_buffer_size: Optional[int] = None
    validation_items_buffer_size: Optional[int] = None
    evaluation_curves_count_per_collection: Optional[int] = None
    evaluation_curve_collections_file_names: Optional[List[str]] = None
    evaluation_sampling_ratios: Optional[List[float]] = None
    evaluation_multimodality: Optional[int] = None
    epochs: Optional[int] = None
    training_batch_size: Optional[int] = None
    validation_batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    checkpoint_rate: Optional[int] = None
    history_size: Optional[int] = None
    activation_fn: Optional[str] = None
    batch_norm_fn: Optional[str] = None
    in_features_size: Optional[int] = None
    hidden_layer_repetitions: Optional[int] = None
    training_min_det: Optional[float] = None
    training_max_det: Optional[float] = None
    training_min_cond: Optional[float] = None
    training_max_cond: Optional[float] = None
    validation_min_det: Optional[float] = None
    validation_max_det: Optional[float] = None
    validation_min_cond: Optional[float] = None
    validation_max_cond: Optional[float] = None
    add_flip_as_negative_example: Optional[bool] = None
    invariance_loss_fn: Optional[str] = None
    margin: Optional[float] = None


parser = TrainSignatureArgumentParser().parse_args()
os.environ["WANDB_DIR"] = str(parser.wandb_dir)
os.environ["WANDB_CACHE_DIR"] = str(parser.wandb_cache_dir)
os.environ["WANDB_CONFIG_DIR"] = str(parser.wandb_config_dir)


def main():
    if parser.sweep_id is not None:
        wandb.init()
        config = wandb.config
    else:
        config = parser

    results_dir_path = init_app(parser=parser)

    training_planar_curves_manager = PlanarCurvesManager(curves_file_paths=[parser.training_curves_file_path])
    training_group = Group.from_group_name(name=config.group_name, min_det=config.training_min_det, max_det=config.training_max_det, min_cond=config.training_min_cond, max_cond=config.training_max_cond)

    validation_planar_curves_manager = PlanarCurvesManager(curves_file_paths=[parser.validation_curves_file_path])
    validation_group = Group.from_group_name(name=config.group_name, min_det=config.validation_min_det, max_det=config.validation_max_det, min_cond=config.validation_min_cond, max_cond=config.validation_max_cond)

    test_planar_curves_manager = PlanarCurvesManager(curves_file_paths=[parser.test_curves_file_path])
    test_group_curves = Group.from_group_name(name=config.group_name, min_det=config.validation_min_det, max_det=config.validation_max_det, min_cond=config.validation_min_cond, max_cond=config.validation_max_cond)

    test_circles_curves_manager = PlanarCurvesManager(curves_file_paths=[parser.test_circles_file_path])
    test_group_circles = Group.from_group_name(name=config.group_name, min_det=config.validation_min_det, max_det=config.validation_max_det, min_cond=config.validation_min_cond, max_cond=config.validation_max_cond)

    validation_dataset = CurveNeighborhoodTupletsDataset(
        planar_curves_manager=validation_planar_curves_manager,
        group=validation_group,
        dataset_size=config.validation_dataset_size,
        supporting_points_count=config.supporting_points_count,
        negative_examples_count=config.negative_examples_count,
        min_sampling_ratio=config.min_sampling_ratio,
        max_sampling_ratio=config.max_sampling_ratio,
        min_multimodality=config.min_multimodality,
        max_multimodality=config.max_multimodality,
        min_negative_example_offset=config.min_negative_example_offset,
        max_negative_example_offset=config.max_negative_example_offset,
        log_dir_path=results_dir_path,
        num_workers=parser.validation_num_workers,
        items_queue_maxsize=config.validation_items_queue_maxsize,
        items_buffer_size=config.validation_items_buffer_size,
        get_item_policy=GetItemPolicy.Keep,
        add_flip_as_negative_example=config.add_flip_as_negative_example)

    validation_dataset.start()
    validation_dataset.stop()
    validation_dataset.join()

    training_dataset = CurveNeighborhoodTupletsDataset(
        planar_curves_manager=training_planar_curves_manager,
        group=training_group,
        dataset_size=config.training_dataset_size,
        supporting_points_count=config.supporting_points_count,
        negative_examples_count=config.negative_examples_count,
        min_sampling_ratio=config.min_sampling_ratio,
        max_sampling_ratio=config.max_sampling_ratio,
        min_multimodality=config.min_multimodality,
        max_multimodality=config.max_multimodality,
        min_negative_example_offset=config.min_negative_example_offset,
        max_negative_example_offset=config.max_negative_example_offset,
        log_dir_path=results_dir_path,
        num_workers=parser.training_num_workers,
        items_queue_maxsize=config.training_items_queue_maxsize,
        items_buffer_size=config.training_items_buffer_size,
        get_item_policy=GetItemPolicy.TryReplace,
        add_flip_as_negative_example=config.add_flip_as_negative_example)

    training_dataset.start()

    activation_fns = {
        'sine': lambda out_features_size: Sine(),
        'relu': lambda out_features_size: torch.nn.ReLU(),
        'gelu': lambda out_features_size: torch.nn.GELU()
    }

    create_batch_norm_fn: Callable[[int], torch.nn.Module] = lambda out_features_size: torch.nn.BatchNorm1d(num_features=out_features_size)

    model = DeepSignaturesNet(
        sample_points=2*config.supporting_points_count + 1,
        in_features_size=config.in_features_size,
        out_features_size=2,
        hidden_layer_repetitions=config.hidden_layer_repetitions,
        create_activation_fn=activation_fns[config.activation_fn],
        create_batch_norm_fn=create_batch_norm_fn,
        dropout_p=None)

    model = torch.nn.DataParallel(model)
    model.share_memory()

    invariance_loss_fns = {
        'TripletMarginLoss': lambda: torch.nn.TripletMarginLoss(margin=config.margin),
        'InvarianceLoss': lambda: InvarianceLoss(),
    }

    loss_fn = DifferentialInvariantsLoss(invariance_loss_fn=invariance_loss_fns[config.invariance_loss_fn])
    optimizer = torch.optim.LBFGS(model.parameters(), lr=config.learning_rate, line_search_fn='strong_wolfe', history_size=config.history_size)

    comparator = PlanarCurvesSignatureHausdorffComparator()
    calculator = PlanarCurvesNeuralSignatureCalculator(
        model=model,
        supporting_points_count=parser.supporting_points_count,
        device=torch.device('cpu'))

    shape_matching_evaluator = PlanarCurvesShapeMatchingEvaluator(
        log_dir_path=results_dir_path,
        num_workers_calculation=parser.evaluation_num_workers,
        num_workers_comparison=parser.evaluation_num_workers,
        curves_count_per_collection=config.evaluation_curves_count_per_collection,
        curve_collections_file_names=config.evaluation_curve_collections_file_names,
        benchmark_dir_path=parser.evaluation_benchmark_dir_path,
        sampling_ratios=config.evaluation_sampling_ratios,
        multimodalities=[config.evaluation_multimodality],
        group_names=[config.group_name],
        planar_curves_signature_calculator=calculator,
        planar_curves_signature_comparator=comparator)

    qualitative_evaluator_curves = PlanarCurvesSignatureQualitativeEvaluator(
        curves_count=3,
        model=model,
        supporting_points_count=config.supporting_points_count,
        planar_curves_manager=test_planar_curves_manager,
        group=test_group_curves,
        multimodality=config.evaluation_multimodality,
        sampling_ratios=config.evaluation_sampling_ratios)

    qualitative_evaluator_circles = PlanarCurvesSignatureQualitativeEvaluator(
        curves_count=2,
        model=model,
        supporting_points_count=config.supporting_points_count,
        planar_curves_manager=test_circles_curves_manager,
        group=test_group_circles,
        multimodality=config.evaluation_multimodality,
        sampling_ratios=config.evaluation_sampling_ratios)

    model_trainer = ModelTrainer(
        results_dir_path=results_dir_path,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        shape_matching_evaluator=shape_matching_evaluator,
        qualitative_evaluators=[qualitative_evaluator_circles, qualitative_evaluator_curves],
        epochs=config.epochs,
        training_batch_size=config.training_batch_size,
        validation_batch_size=config.validation_batch_size,
        num_workers=0,
        checkpoint_rate=config.checkpoint_rate,
        device=torch.device('cuda'),
        log_wandb=True if parser.sweep_id is not None else False)

    model_trainer.train()
    training_dataset.stop()
    training_dataset.join()


if __name__ == '__main__':
    if parser.sweep_id is not None:
        wandb.agent(parser.sweep_id, function=main, count=parser.count)
    else:
        main()
