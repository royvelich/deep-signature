# python peripherals
import sys
from pathlib import Path
from typing import Callable, Optional, List
sys.path.append('../../.')

# numpy
import numpy

# torch
import torch

# matplotlib
# import matplotlib
# import matplotlib.pyplot
# import matplotlib.axes
# matplotlib.pyplot.rcParams['text.usetex'] = True

# deep-signature
from deep_signature.core.base import SeedableObject
# from deep_signature.manifolds.planar_curves.implementation import PlanarCurvesManager, PlanarCurve
# from deep_signature.manifolds.planar_curves.groups import EuclideanGroup, SimilarityGroup, EquiaffineGroup, AffineGroup
# from deep_signature.core import discrete_distributions
# from deep_signature.training import datasets
from deep_signature.training.networks import DeepSignaturesNet
from deep_signature.training.activations import Sine
from deep_signature.manifolds.planar_curves.evaluation import PlanarCurvesNeuralSignatureCalculator, PlanarCurvesAxiomaticEuclideanSignatureCalculator, PlanarCurvesShapeMatchingEvaluator, PlanarCurvesSignatureHausdorffComparator, PlanarCurvesSignatureCalculator, PlanarCurvesSignatureComparator
# from deep_signature.manifolds.planar_curves.groups import EquiaffineGroup

SeedableObject.set_seed(seed=42)

supporting_points_count = 14
sample_points = 2 * supporting_points_count + 1
sampling_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
multimodality = 10
curves_count_per_collection = 30
num_workers_calculation = 17
num_workers_comparison = 5
group_name = 'affine'
root_folder = 'C:/deep-signature-data-new'


def run_benchmark(benchmark_dir_path: Path, collections: List[str], output_folder: Path, group_name: str, calculator: PlanarCurvesSignatureCalculator, comparator: PlanarCurvesSignatureComparator):
    output_folder.mkdir(parents=True, exist_ok=True)
    shape_matching_evaluator = PlanarCurvesShapeMatchingEvaluator(
        log_dir_path=output_folder,
        num_workers_calculation=num_workers_calculation,
        num_workers_comparison=num_workers_comparison,
        curves_count_per_collection=curves_count_per_collection,
        curve_collections_file_names=collections,
        benchmark_dir_path=benchmark_dir_path,
        sampling_ratios=sampling_ratios,
        multimodalities=[multimodality],
        group_names=[group_name],
        planar_curves_signature_calculator=calculator,
        planar_curves_signature_comparator=comparator)

    shape_matching_evaluator.start()
    shape_matching_evaluator.join()

    print(shape_matching_evaluator.shape_matching_df)
    shape_matching_evaluator.shape_matching_df.to_csv(f"{str(output_folder)}/shape_matching_df.csv")
    shape_matching_evaluator.df.to_csv(f"{str(output_folder)}/df.csv")


if __name__ == '__main__':
    create_activation_fn: Callable[[int], torch.nn.Module] = lambda out_features_size: Sine()
    create_batch_norm_fn: Callable[[int], torch.nn.Module] = lambda out_features_size: torch.nn.BatchNorm1d(num_features=out_features_size)

    models = [
        {
            'model_date': '2023-01-20-04-18-14',
            'model_iteration': '2200'
        },
        {
            'model_date': '2023-01-12-08-13-54',
            'model_iteration': '2325'
        },
        {
            'model_date': '2023-01-18-00-26-44',
            'model_iteration': '1000'
        },
        {
            'model_date': '2023-01-10-05-20-03',
            'model_iteration': '950'
        }
    ]

    benchmark_dates = ['2023-01-19-21-49-56', '2023-01-19-21-52-07', '2023-01-19-21-55-02']
    collections = ['basketball', 'bats', 'birds', 'branches', 'bunnies', 'butterflies', 'cacti', 'cats', 'chickens', 'clouds', 'deers', 'dogs', 'fishes', 'flames', 'flies', 'fruits', 'glasses', 'hearts', 'horses', 'insects', 'jogging', 'leaves', 'monkeys', 'mustaches', 'pieces', 'profiles', 'rats', 'shapes', 'shields', 'signs', 'spiders', 'trees', 'whales', 'wings']
    for model in models:
        model_date = model['model_date']
        model_iteration = model['model_iteration']
        model_file_path = f"{root_folder}/training/{model_date}/models/model_{model_iteration}.pt"
        device = torch.device('cpu')
        model = DeepSignaturesNet(sample_points=sample_points, in_features_size=128, out_features_size=2, hidden_layer_repetitions=3, create_activation_fn=create_activation_fn, create_batch_norm_fn=create_batch_norm_fn, dropout_p=None)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_file_path))
        model.eval()
        model.cpu()

        calculator = PlanarCurvesNeuralSignatureCalculator(
            model=model,
            supporting_points_count=supporting_points_count,
            device=device)

        comparator = PlanarCurvesSignatureHausdorffComparator()

        for benchmark_date in benchmark_dates:
            output_folder = Path(f"{root_folder}/output/MODEL_{model_date}_{model_iteration}_BENCHMARK_{benchmark_date}")
            benchmark_dir_path = Path(f"{root_folder}/curves/benchmark/{benchmark_date}")
            run_benchmark(benchmark_dir_path=benchmark_dir_path, output_folder=output_folder, group_name=group_name, collections=collections, calculator=calculator, comparator=comparator)

    calculator = PlanarCurvesAxiomaticEuclideanSignatureCalculator()
    comparator = PlanarCurvesSignatureHausdorffComparator()
    for benchmark_date in benchmark_dates:
        output_folder = Path(f"{root_folder}/output/AXIOMATIC_EUCLIDEAN_BENCHMARK_{benchmark_date}")
        benchmark_dir_path = Path(f"{root_folder}/curves/benchmark/{benchmark_date}")
        run_benchmark(benchmark_dir_path=benchmark_dir_path, output_folder=output_folder, group_name=group_name, collections=collections, calculator=calculator, comparator=comparator)