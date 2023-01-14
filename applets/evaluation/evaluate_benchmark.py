# python peripherals
import sys
from pathlib import Path
from typing import Callable, Optional
sys.path.append('../../.')

# numpy
import numpy

# torch
import torch

# matplotlib
import matplotlib
import matplotlib.pyplot
import matplotlib.axes
matplotlib.pyplot.rcParams['text.usetex'] = True

# deep-signature
from deep_signature.core.base import SeedableObject
from deep_signature.manifolds.planar_curves.implementation import PlanarCurvesManager, PlanarCurve
from deep_signature.manifolds.planar_curves.groups import EuclideanGroup, SimilarityGroup, EquiaffineGroup, AffineGroup
from deep_signature.core import discrete_distributions
from deep_signature.training import datasets
from deep_signature.training.networks import DeepSignaturesNet
from deep_signature.training.activations import Sine
from deep_signature.manifolds.planar_curves.evaluation import PlanarCurvesApproximatedSignatureComparator, PlanarCurvesShapeMatchingEvaluator
from deep_signature.manifolds.planar_curves.groups import EquiaffineGroup

SeedableObject.set_seed(seed=42)

supporting_points_count = 14
sample_points = 2 * supporting_points_count + 1
sampling_ratio = 1
multimodality = 10
group_name = 'affine'

if __name__ == '__main__':
    create_activation_fn: Callable[[int], torch.nn.Module] = lambda out_features_size: Sine()
    create_batch_norm_fn: Callable[[int], torch.nn.Module] = lambda out_features_size: torch.nn.BatchNorm1d(num_features=out_features_size)
    # model_file_path = "C:/deep-signature-data-new/training/2023-01-03-13-40-21/models/model_375.pt"
    # model_file_path = "C:/deep-signature-data-new/training/2023-01-05-15-42-57/models/model_300.pt"
    # model_file_path = "C:/deep-signature-data-new/training/2023-01-06-19-05-56/models/model_725.pt"
    # model_file_path = "C:/deep-signature-data-new/training/2023-01-08-10-32-10/models/model_175.pt"

    # best equiaffine
    # model_file_path = "C:/deep-signature-data-new/training/2023-01-08-15-06-32/models/model_950.pt"

    # model_file_path = "C:/deep-signature-data-new/training/2023-01-10-05-20-03/models/model_950.pt"
    model_file_path = "/data/training/2023-01-12-08-13-54/models/model_2325.pt"

    device = torch.device('cpu')
    model = DeepSignaturesNet(sample_points=sample_points, in_features_size=128, out_features_size=2, hidden_layer_repetitions=3, create_activation_fn=create_activation_fn, create_batch_norm_fn=create_batch_norm_fn, dropout_p=None)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    model.cpu()

    comparator = PlanarCurvesApproximatedSignatureComparator(
        model=model,
        supporting_points_count=supporting_points_count,
        device=device)

    collections = ['basketball', 'bats', 'birds', 'branches', 'bunnies', 'butterflies', 'cacti', 'cats', 'chickens', 'clouds', 'deers', 'dogs', 'fishes', 'flames', 'flies', 'fruits', 'glasses', 'hearts', 'horses', 'insects', 'jogging', 'leaves', 'monkeys', 'mustaches', 'pieces', 'profiles', 'rats', 'shapes', 'shields', 'signs', 'spiders', 'trees', 'whales', 'wings']

    shape_matching_evaluator = PlanarCurvesShapeMatchingEvaluator(
        log_dir_path=Path("/data/output"),
        num_workers=19,
        curves_count_per_collection=30,
        curve_collections_file_names=collections,
        benchmark_dir_path=Path('/data/curves/benchmark/2023-01-07-10-12-19'),
        sampling_ratios=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        multimodalities=[10],
        group_names=[group_name],
        planar_curves_signature_comparator=comparator)

    shape_matching_evaluator.start()
    shape_matching_evaluator.join()

    print(shape_matching_evaluator.shape_matching_df)
    shape_matching_evaluator.shape_matching_df.to_csv("/data/output/output.csv")
