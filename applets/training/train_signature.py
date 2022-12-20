# python peripherals
import os
from pathlib import Path
from typing import Optional, Callable, List

# torch
import torch

# deep-signature
from deep_signature.manifolds.planar_curves.groups import Group
from deep_signature.core import utils
from deep_signature.manifolds.planar_curves.implementation import PlanarCurvesManager
from deep_signature.training.datasets import CurveNeighborhoodTupletsDataset
from deep_signature.training.networks import DeepSignaturesNet
from deep_signature.training.activations import Sine
from deep_signature.training.losses import DifferentialInvariantsLoss
from deep_signature.training.trainers import ModelTrainer
from deep_signature.manifolds.planar_curves.evaluation import PlanarCurvesApproximatedSignatureComparator
from deep_signature.manifolds.planar_curves.evaluation import PlanarCurvesShapeMatchingEvaluator


class TrainSignatureArgumentParser(utils.AppArgumentParser):
    training_curves_file_path: Path
    validation_curves_file_path: Path
    group_name: str
    supporting_points_count: int
    negative_examples_count: int
    min_sampling_ratio: float
    max_sampling_ratio: float
    min_multimodality: int
    max_multimodality: int
    min_negative_example_offset: int
    max_negative_example_offset: int
    training_dataset_size: int
    validation_dataset_size: int
    training_num_workers: int
    validation_num_workers: int
    evaluation_num_workers: int
    evaluation_benchmark_dir_path: Path
    evaluation_curves_count_per_collection: int
    evaluation_curve_collections_file_names: List[str]
    evaluation_sampling_ratios: List[float]
    evaluation_multimodality: int
    epochs: int
    batch_size: int
    learning_rate: float
    checkpoint_rate: int
    history_size: int = 800
    activation_fn: str
    batch_norm_fn: str
    in_features_size: int
    hidden_layer_repetitions: int
    training_min_det: Optional[float] = None
    training_max_det: Optional[float] = None
    training_min_cond: Optional[float] = None
    training_max_cond: Optional[float] = None
    validation_min_det: Optional[float] = None
    validation_max_det: Optional[float] = None
    validation_min_cond: Optional[float] = None
    validation_max_cond: Optional[float] = None


if __name__ == '__main__':



    # def main():
    #     wandb.init(dir=os.getenv("WANDB_DIR", 'C:/sweeps'))

    parser = utils.init_app(typed_argument_parser_class=TrainSignatureArgumentParser)
    training_planar_curves_manager = PlanarCurvesManager(curves_file_path=parser.training_curves_file_path)
    training_group = Group.from_group_name(name=parser.group_name, min_det=parser.training_min_det, max_det=parser.training_max_det, min_cond=parser.training_min_cond, max_cond=parser.training_max_cond)

    validation_planar_curves_manager = PlanarCurvesManager(curves_file_path=parser.validation_curves_file_path)
    validation_group = Group.from_group_name(name=parser.group_name, min_det=parser.validation_min_det, max_det=parser.validation_max_det, min_cond=parser.validation_min_cond, max_cond=parser.validation_max_cond)

    training_dataset = CurveNeighborhoodTupletsDataset(
        planar_curves_manager=training_planar_curves_manager,
        group=training_group,
        dataset_size=parser.training_dataset_size,
        supporting_points_count=parser.supporting_points_count,
        negative_examples_count=parser.negative_examples_count,
        min_sampling_ratio=parser.min_sampling_ratio,
        max_sampling_ratio=parser.max_sampling_ratio,
        min_multimodality=parser.min_multimodality,
        max_multimodality=parser.max_multimodality,
        min_negative_example_offset=parser.min_negative_example_offset,
        max_negative_example_offset=parser.max_negative_example_offset)

    validation_dataset = CurveNeighborhoodTupletsDataset(
        planar_curves_manager=validation_planar_curves_manager,
        group=validation_group,
        dataset_size=parser.validation_dataset_size,
        supporting_points_count=parser.supporting_points_count,
        negative_examples_count=parser.negative_examples_count,
        min_sampling_ratio=parser.min_sampling_ratio,
        max_sampling_ratio=parser.max_sampling_ratio,
        min_multimodality=parser.min_multimodality,
        max_multimodality=parser.max_multimodality,
        min_negative_example_offset=parser.min_negative_example_offset,
        max_negative_example_offset=parser.max_negative_example_offset)

    activation_fns = {
        'sine': lambda out_features_size: Sine(),
        'relu': lambda out_features_size: torch.nn.ReLU(),
        'gelu': lambda out_features_size: torch.nn.GELU()
    }

    create_batch_norm_fn: Callable[[int], torch.nn.Module] = lambda out_features_size: torch.nn.BatchNorm1d(num_features=out_features_size)

    model = DeepSignaturesNet(
        sample_points=2*parser.supporting_points_count + 1,
        in_features_size=parser.in_features_size,
        out_features_size=2,
        hidden_layer_repetitions=parser.hidden_layer_repetitions,
        create_activation_fn=activation_fns[parser.activation_fn],
        create_batch_norm_fn=create_batch_norm_fn,
        dropout_p=None)

    loss_fn = DifferentialInvariantsLoss()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=parser.learning_rate, line_search_fn='strong_wolfe', history_size=parser.history_size)

    comparator = PlanarCurvesApproximatedSignatureComparator(
        model=model,
        supporting_points_count=parser.supporting_points_count,
        device=torch.device('cpu'))

    evaluator = PlanarCurvesShapeMatchingEvaluator(
        log_dir_path=parser.results_base_dir_path,
        num_workers=parser.evaluation_num_workers,
        curves_count_per_collection=parser.evaluation_curves_count_per_collection,
        curve_collections_file_names=parser.evaluation_curve_collections_file_names,
        benchmark_dir_path=parser.evaluation_benchmark_dir_path,
        sampling_ratios=parser.evaluation_sampling_ratios,
        multimodalities=[parser.evaluation_multimodality],
        group_names=[parser.group_name],
        planar_curves_signature_comparator=comparator)

    model_trainer = ModelTrainer(
        results_dir_path=parser.results_base_dir_path,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        evaluator=evaluator,
        epochs=parser.epochs,
        batch_size=parser.batch_size,
        num_workers=parser.training_num_workers,
        checkpoint_rate=parser.checkpoint_rate,
        device=torch.device('cuda'))
