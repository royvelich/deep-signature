# python peripherals
from __future__ import annotations
from typing import List, cast, Optional
import os
import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum, auto
from multiprocessing import Manager

# numpy
import numpy

# pandas
import pandas

# pytorch
import torch
import torch.nn

# matplotlib
import matplotlib
import matplotlib.axes
import matplotlib.figure

# PIL
import PIL
from PIL import Image

# deep_signature
from deep_signature.manifolds.planar_curves.implementation import PlanarCurve
from deep_signature.core.parallel_processing import TaskParallelProcessor, ParallelProcessingTask
from deep_signature.manifolds.planar_curves.generation import ShapeMatchingBenchmarkCurvesGeneratorTask
from deep_signature.metrics import hausdorff
from deep_signature.manifolds.planar_curves.groups import Group
from deep_signature.manifolds.planar_curves.implementation import PlanarCurvesManager
from deep_signature.core import discrete_distributions


# =================================================
# PlanarCurvesSignatureCalculator Class
# =================================================
class PlanarCurvesSignatureCalculator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calculate_signature(self, curve: PlanarCurve) -> numpy.ndarray:
        pass


# =================================================
# PlanarCurvesNeuralSignatureCalculator Class
# =================================================
class PlanarCurvesNeuralSignatureCalculator(PlanarCurvesSignatureCalculator):
    def __init__(self, model: torch.nn.Module, supporting_points_count: int, device: torch.device):
        self._model = model
        self._supporting_points_count = supporting_points_count
        self._device = device
        super().__init__()

    def calculate_signature(self, curve: PlanarCurve) -> numpy.ndarray:
        return curve.approximate_curve_signature(model=self._model, supporting_points_count=self._supporting_points_count, device=self._device)


# =================================================
# PlanarCurvesAxiomaticEuclideanSignatureCalculator Class
# =================================================
class PlanarCurvesAxiomaticEuclideanSignatureCalculator(PlanarCurvesSignatureCalculator):
    def __init__(self):
        super().__init__()

    def calculate_signature(self, curve: PlanarCurve) -> numpy.ndarray:
        return curve.approximate_euclidean_signature()


# =================================================
# PlanarCurvesSignatureComparator Class
# =================================================
class PlanarCurvesSignatureComparator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compare_signatures(self, signature1: numpy.ndarray, signature2: numpy.ndarray) -> float:
        pass


# =================================================
# PlanarCurvesSignatureHausdorffComparator Class
# =================================================
class PlanarCurvesSignatureHausdorffComparator(PlanarCurvesSignatureComparator):
    def __init__(self):
        super().__init__()

    def compare_signatures(self, signature1: numpy.ndarray, signature2: numpy.ndarray) -> float:
        return hausdorff.avg_hausdorff_distance(subset1=signature1, subset2=signature2, distance_function=hausdorff.euclidean)


# =================================================
# PlanarCurvesShapeMatchingSignatureCalculationTask Class
# =================================================
class PlanarCurvesShapeMatchingSignatureCalculationTask(ParallelProcessingTask):
    def __init__(
            self,
            curves_file_name: str,
            curve_id: int,
            benchmark_dir_path: Path,
            sampling_ratio: Optional[float],
            multimodality: int,
            group_name: str,
            planar_curves_signature_calculator: PlanarCurvesSignatureCalculator):
        super().__init__()
        self._curves_file_name = curves_file_name
        self._curve_id = curve_id
        self._benchmark_dir_path = benchmark_dir_path
        self._sampling_ratio = sampling_ratio
        self._multimodality = multimodality
        self._group_name = group_name
        self._planar_curves_signature_calculator = planar_curves_signature_calculator
        self._df = pandas.DataFrame()

    @property
    def df(self) -> pandas.DataFrame:
        return self._df

    def _pre_process(self):
        pass

    def _process(self, *argv):
        if self._sampling_ratio is not None:
            curves_file_path = self._benchmark_dir_path / ShapeMatchingBenchmarkCurvesGeneratorTask.get_query_relative_file_path(curves_file_name=self._curves_file_name, sampling_ratio=self._sampling_ratio, multimodality=self._multimodality, group_name=self._group_name)
        else:
            curves_file_path = self._benchmark_dir_path / ShapeMatchingBenchmarkCurvesGeneratorTask.get_dataset_relative_file_path(curves_file_name=self._curves_file_name)

        curves = numpy.load(file=str(curves_file_path), allow_pickle=True)
        curve = PlanarCurve(points=curves[self._curve_id])
        signature = self._planar_curves_signature_calculator.calculate_signature(curve=curve)

        data = {
            'curves_file_name': [self._curves_file_name],
            'sampling_ratio': [self._sampling_ratio],
            'multimodality': [self._multimodality],
            'group_name': [self._group_name],
            'curve_id': [self._curve_id],
            'signature': [signature]
        }

        self._df = pandas.DataFrame(data=data)

    def _post_process(self):
        pass

    def post_process(self):
        pass


# =================================================
# PlanarCurvesShapeMatchingSignatureComparisonTask Class
# =================================================
class PlanarCurvesShapeMatchingSignatureComparisonTask(ParallelProcessingTask):
    def __init__(
            self,
            curves_file_name: str,
            query_curve_id: int,
            database_curve_id: int,
            sampling_ratio: Optional[float],
            multimodality: int,
            group_name: str,
            planar_curves_signature_comparator: PlanarCurvesSignatureComparator):
        super().__init__()
        self._curves_file_name = curves_file_name
        self._query_curve_id = query_curve_id
        self._database_curve_id = database_curve_id
        self._sampling_ratio = sampling_ratio
        self._multimodality = multimodality
        self._group_name = group_name
        self._planar_curves_signature_comparator = planar_curves_signature_comparator
        self._df = pandas.DataFrame()

    @property
    def df(self) -> pandas.DataFrame:
        return self._df

    def _pre_process(self):
        pass

    def _get_signature(self, df: pandas.DataFrame, curves_file_name: str, sampling_ratio: Optional[float], multimodality: int, group_name: str, curve_id: int) -> numpy.ndarray:
        if sampling_ratio is not None:
            return df.loc[
                    (df['curves_file_name'] == curves_file_name) &
                    (df['sampling_ratio'] == sampling_ratio) &
                    (df['multimodality'] == multimodality) &
                    (df['group_name'] == group_name) &
                    (df['curve_id'] == curve_id)]['signature'].item()
        else:
            return df.loc[
                (df['curves_file_name'] == curves_file_name) &
                (df['sampling_ratio'].isnull()) &
                (df['multimodality'] == multimodality) &
                (df['group_name'] == group_name) &
                (df['curve_id'] == curve_id)]['signature'].item()

    def _process(self, *argv):
        df = argv[0]
        signature1 = self._get_signature(df=df, curves_file_name=self._curves_file_name, sampling_ratio=self._sampling_ratio, multimodality=self._multimodality, group_name=self._group_name, curve_id=self._query_curve_id)
        signature2 = self._get_signature(df=df, curves_file_name=self._curves_file_name, sampling_ratio=None, multimodality=self._multimodality, group_name=self._group_name, curve_id=self._database_curve_id)
        score = self._planar_curves_signature_comparator.compare_signatures(signature1=signature1, signature2=signature2)
        data = {
            'curves_file_name': [self._curves_file_name],
            'sampling_ratio': [self._sampling_ratio],
            'multimodality': [self._multimodality],
            'group': [self._group_name],
            'query_curve_id': [self._query_curve_id],
            'database_curve_id': [self._database_curve_id],
            'score': [score]
        }

        self._df = pandas.DataFrame(data=data)

    def _post_process(self):
        pass

    def post_process(self):
        pass


# =================================================
# PlanarCurvesShapeMatchingEvaluatorPhase Enum
# =================================================
class PlanarCurvesShapeMatchingEvaluatorPhase(Enum):
    SignatureCalculation = auto()
    SignatureComparison = auto()


# =================================================
# PlanarCurvesShapeMatchingEvaluator Class
# =================================================
class PlanarCurvesShapeMatchingEvaluator(TaskParallelProcessor):
    def __init__(
            self,
            log_dir_path: Path,
            num_workers_calculation: int,
            num_workers_comparison: int,
            curves_count_per_collection: int,
            curve_collections_file_names: List[str],
            benchmark_dir_path: Path,
            sampling_ratios: List[Optional[float]],
            multimodalities: List[int],
            group_names: List[str],
            planar_curves_signature_calculator: PlanarCurvesSignatureCalculator,
            planar_curves_signature_comparator: PlanarCurvesSignatureComparator):
        self._num_workers_calculation = num_workers_calculation
        self._num_workers_comparison = num_workers_comparison
        self._curves_count_per_collection = curves_count_per_collection
        self._curve_ids = list(range(self._curves_count_per_collection))
        self._curve_collections_file_names = [os.path.normpath(curve_collection_file_name) for curve_collection_file_name in curve_collections_file_names]
        self._benchmark_dir_path = benchmark_dir_path
        self._sampling_ratios = sampling_ratios
        self._sampling_ratios_with_none = sampling_ratios.copy()
        self._sampling_ratios_with_none.append(None)
        self._multimodalities = multimodalities
        self._group_names = group_names
        self._planar_curves_signature_calculator = planar_curves_signature_calculator
        self._planar_curves_signature_comparator = planar_curves_signature_comparator
        self._signature_calculation_df = pandas.DataFrame()
        self._signature_comparison_df = pandas.DataFrame()
        self._max_scores_df = pandas.DataFrame()
        self._shape_matching_df = pandas.DataFrame()
        self._phase = PlanarCurvesShapeMatchingEvaluatorPhase.SignatureCalculation
        super().__init__(log_dir_path=log_dir_path, num_workers=self._num_workers_calculation)

    @property
    def df(self) -> pandas.DataFrame:
        return self._df

    @property
    def shape_matching_df(self) -> pandas.DataFrame:
        return self._shape_matching_df

    def get_evaluation_score(self) -> float:
        return self._shape_matching_df['match'].mean()

    def start(self, num_workers: Optional[int] = None):
        self._phase = PlanarCurvesShapeMatchingEvaluatorPhase.SignatureCalculation
        super().start(num_workers=self._num_workers_calculation)
        super().join()
        self._phase = PlanarCurvesShapeMatchingEvaluatorPhase.SignatureComparison
        super().start(num_workers=self._num_workers_comparison)

    def _get_states_to_remove(self) -> List[str]:
        return ['_manager']

    def _get_args(self) -> List[object]:
        if self._phase == PlanarCurvesShapeMatchingEvaluatorPhase.SignatureCalculation:
            return []
        else:
            return [self._namespace.signature_calculation_df]

    def _generate_tasks(self) -> List[ParallelProcessingTask]:
        tasks = []
        if self._phase == PlanarCurvesShapeMatchingEvaluatorPhase.SignatureCalculation:
            combinations = list(itertools.product(*[
                self._curve_collections_file_names,
                self._curve_ids,
                [self._benchmark_dir_path],
                self._sampling_ratios_with_none,
                self._multimodalities,
                self._group_names,
                [self._planar_curves_signature_calculator]]))

            for combination in combinations:
                tasks.append(PlanarCurvesShapeMatchingSignatureCalculationTask(
                    curves_file_name=combination[0],
                    curve_id=combination[1],
                    benchmark_dir_path=combination[2],
                    sampling_ratio=combination[3],
                    multimodality=combination[4],
                    group_name=combination[5],
                    planar_curves_signature_calculator=combination[6]))
        else:
            combinations = list(itertools.product(*[
                self._curve_collections_file_names,
                self._curve_ids,
                self._curve_ids,
                self._sampling_ratios,
                self._multimodalities,
                self._group_names,
                [self._planar_curves_signature_comparator]]))

            # for combination in combinations:
            #     curves_file_name = combination[0]
            #     query_curve_id = combination[1]
            #     database_curve_id = combination[2]
            #     sampling_ratio = combination[3]
            #     multimodality = combination[4]
            #     group_name = combination[5]
            #     planar_curves_signature_comparator = combination[6]
            #
            #     tasks.append(PlanarCurvesShapeMatchingSignatureComparisonTask(
            #         curves_file_name=curves_file_name,
            #         query_curve_id=query_curve_id,
            #         database_curve_id=database_curve_id,
            #         sampling_ratio=sampling_ratio,
            #         multimodality=multimodality,
            #         group_name=group_name,
            #         planar_curves_signature_comparator=planar_curves_signature_comparator))

            tasks = [PlanarCurvesShapeMatchingSignatureComparisonTask(
                    curves_file_name=combination[0],
                    query_curve_id=combination[1],
                    database_curve_id=combination[2],
                    sampling_ratio=combination[3],
                    multimodality=combination[4],
                    group_name=combination[5],
                    planar_curves_signature_comparator=combination[6]) for combination in combinations]

        return tasks

    def _post_start(self):
        super()._post_start()
        if self._phase == PlanarCurvesShapeMatchingEvaluatorPhase.SignatureCalculation:
            df_list = []
            for completed_task in self._completed_tasks:
                completed_task = cast(typ=PlanarCurvesShapeMatchingSignatureCalculationTask, val=completed_task)
                df_list.append(completed_task.df)
            self._signature_calculation_df = pandas.concat(df_list).reset_index()
            self._namespace.signature_calculation_df = self._signature_calculation_df
        else:
            df_list = []
            for completed_task in self._completed_tasks:
                completed_task = cast(typ=PlanarCurvesShapeMatchingSignatureComparisonTask, val=completed_task)
                df_list.append(completed_task.df)
            self._df = pandas.concat(df_list).reset_index()
            self._max_scores_df = self._df.loc[self._df.groupby(['curves_file_name', 'sampling_ratio', 'multimodality', 'group', 'query_curve_id'])['score'].idxmin()]
            self._max_scores_df['match'] = (self._max_scores_df['query_curve_id'] == self._max_scores_df['database_curve_id']).astype(int)
            self._shape_matching_df = self._max_scores_df.groupby(['curves_file_name', 'sampling_ratio', 'multimodality', 'group'])['match'].mean().reset_index()

    def _pre_join(self):
        pass

    def _post_join(self):
        pass


# # =================================================
# # PlanarCurvesEvaluationPlotter Class
# # =================================================
# class PlanarCurvesEvaluationPlotter(ABC):
#     def __init__(self):
#         super().__init__()
#
#     @abstractmethod
#     def plot_evaluation(self) -> List[Image]:
#         pass
#
#
# # =================================================
# # PlanarCurvesEvaluationPlotter Class
# # =================================================
# class PlanarCurvesSignatureEvaluationPlotter(PlanarCurvesEvaluationPlotter):
#     def __init__(self, model: torch.nn.Module, supporting_points_count: int, device: torch.device):
#         self._model = model
#         self._supporting_points_count = supporting_points_count
#         self._device = device
#         super().__init__()
#
#     def plot_evaluation(self) -> Image:


# =================================================
# PlanarCurvesQualitativeEvaluator Class
# =================================================
class PlanarCurvesQualitativeEvaluator(ABC):
    def __init__(self, curves_count: int):
        self._curves_count = curves_count

    @abstractmethod
    def evaluate_curves(self) -> List[Image]:
        pass


# =================================================
# PlanarCurvesQualitativeEvaluator Class
# =================================================
class PlanarCurvesSignatureQualitativeEvaluator(PlanarCurvesQualitativeEvaluator):
    def __init__(
            self,
            curves_count: int,
            model: torch.nn.Module,
            supporting_points_count: int,
            planar_curves_manager: PlanarCurvesManager,
            group: Group,
            sampling_ratios: List[float],
            multimodality: int):
        super().__init__(curves_count=curves_count)
        self._model = model
        self._supporting_points_count = supporting_points_count
        self._planar_curves_manager = planar_curves_manager
        self._group = group
        self._curves = []
        self._sampling_ratios = sampling_ratios
        self._multimodality = multimodality
        for i in range(self._curves_count):
            curve = self._planar_curves_manager.get_random_planar_curve()
            self._curves.append(curve)

    def _evaluate_curve(self, curve: PlanarCurve, sampling_ratio: float) -> Image:
        device = torch.device('cpu')
        fig, axes = matplotlib.pyplot.subplots(nrows=4, ncols=1, figsize=(10, 20))
        axes[0].set_title(label='Curve', fontsize=30)
        axes[1].set_title(label='K', fontsize=30)
        axes[2].set_title(label='dK/ds', fontsize=30)
        axes[3].set_title(label='K vs. dK/ds', fontsize=30)
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.tick_params(axis='both', which='minor', labelsize=20)

        discrete_distribution = discrete_distributions.MultimodalGaussianDiscreteDistribution(bins_count=curve.points_count, multimodality=self._multimodality)
        curve = curve.sample_curve(sampling_ratio=sampling_ratio, discrete_distribution=discrete_distribution)
        curve.plot_scattered_curve(ax=axes[0])
        curve.plot_signature(model=self._model, supporting_points_count=self._supporting_points_count, line_style='-', marker='', device=device, ax=axes[1:])
        fig.canvas.draw()
        image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        matplotlib.pyplot.close(fig)
        return image

    def evaluate_curves(self) -> List[Image]:
        images = []
        for curve in self._curves:
            image = self._evaluate_curve(curve=curve, sampling_ratio=1.0)
            images.append(image)

            for sampling_ratio in self._sampling_ratios:
                image = self._evaluate_curve(curve=curve, sampling_ratio=sampling_ratio)
                images.append(image)

        return images
