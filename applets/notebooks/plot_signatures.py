# python peripherals
import sys
from pathlib import Path
from typing import Callable

# torch
import torch

# matplotlib
import matplotlib
import matplotlib.pyplot
import matplotlib.axes

# deep-signature
from deep_signature.core.base import SeedableObject
from deep_signature.manifolds.planar_curves.implementation import PlanarCurvesManager, PlanarCurve
from deep_signature.manifolds.planar_curves.groups import EuclideanGroup, SimilarityGroup, EquiaffineGroup, AffineGroup
from deep_signature.core import discrete_distributions
from deep_signature.training import datasets
from deep_signature.training.networks import DeepSignaturesNet
from deep_signature.training.activations import Sine
from deep_signature.manifolds.planar_curves.evaluation import PlanarCurvesApproximatedSignatureComparator, PlanarCurvesShapeMatchingEvaluator


if __name__ == '__main__':
    supporting_points_count = 9
    sample_points = 2 * supporting_points_count + 1

    create_activation_fn: Callable[[int], torch.nn.Module] = lambda out_features_size: Sine()
    create_batch_norm_fn: Callable[[int], torch.nn.Module] = lambda out_features_size: torch.nn.BatchNorm1d(num_features=out_features_size)
    model_file_path = "C:/deep-signature-data-new/training/2023-01-02-17-11-16/models/model_150.pt"
    device = torch.device('cpu')
    model = DeepSignaturesNet(sample_points=sample_points, in_features_size=64, out_features_size=2, hidden_layer_repetitions=2, create_activation_fn=create_activation_fn, create_batch_norm_fn=create_batch_norm_fn, dropout_p=None)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    model.cpu()

    for p in model.parameters():
        if p.requires_grad:
            print(p.name, p.data)

    comparator = PlanarCurvesApproximatedSignatureComparator(
        model=model,
        supporting_points_count=supporting_points_count,
        device=device)

    shape_matching_evaluator = PlanarCurvesShapeMatchingEvaluator(
        log_dir_path=Path("C:/deep-signature-data-new/notebooks_output"),
        num_workers=5,
        curves_count_per_collection=10,
        curve_collections_file_names=['whales'],
        benchmark_dir_path=Path('C:/deep-signature-data-new/curves/benchmark/2023-01-02-00-24-28'),
        sampling_ratios=[0.8],
        multimodalities=[5],
        group_names=['equiaffine'],
        planar_curves_signature_comparator=comparator)

    shape_matching_evaluator.start()
    shape_matching_evaluator.join()

    print(shape_matching_evaluator.df)
    print(shape_matching_evaluator.shape_matching_df)
