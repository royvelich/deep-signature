# python peripherals
from __future__ import annotations
from typing import List, cast
import os
import itertools
from abc import ABC, abstractmethod
from pathlib import Path

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


# =================================================
# PlanarCurvesSignatureComparator Class
# =================================================
class PlanarCurvesSignatureComparator(ABC):
    def __init__(self):
        pass

    def compare_signatures(self, curve1: PlanarCurve, curve2: PlanarCurve) -> float:
        curve1_signature = self._calculate_signature(curve=curve1)
        curve2_signature = self._calculate_signature(curve=curve2)
        return hausdorff.avg_hausdorff_distance(subset1=curve1_signature, subset2=curve2_signature, distance_function=hausdorff.euclidean)

    @abstractmethod
    def _calculate_signature(self, curve: PlanarCurve) -> numpy.ndarray:
        pass


# =================================================
# PlanarCurvesSignatureComparator Class
# =================================================
class PlanarCurvesApproximatedSignatureComparator(PlanarCurvesSignatureComparator):
    def __init__(self, model: torch.nn.Module, supporting_points_count: int, device: torch.device):
        self._model = model
        self._supporting_points_count = supporting_points_count
        self._device = device
        super().__init__()

    def _calculate_signature(self, curve: PlanarCurve) -> numpy.ndarray:
        return curve.approximate_curve_signature(model=self._model, supporting_points_count=self._supporting_points_count, device=self._device)


# =================================================
# ShapeMatchingBenchmarkCurvesGeneratorTask Class
# =================================================
class PlanarCurvesShapeMatchingEvaluatorTask(ParallelProcessingTask):
    def __init__(
            self,
            curves_file_name: str,
            query_curve_id: int,
            database_curve_id: int,
            benchmark_dir_path: Path,
            sampling_ratio: float,
            multimodality: int,
            group_name: str,
            planar_curves_signature_comparator: PlanarCurvesSignatureComparator):
        super().__init__()
        self._curves_file_name = curves_file_name
        self._query_curve_id = query_curve_id
        self._database_curve_id = database_curve_id
        self._benchmark_dir_path = benchmark_dir_path
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

    def _process(self, *argv):
        query_curves_file_path = self._benchmark_dir_path / ShapeMatchingBenchmarkCurvesGeneratorTask.get_relative_file_path(curves_file_name=self._curves_file_name, sampling_ratio=self._sampling_ratio, multimodality=self._multimodality, group_name=self._group_name)
        database_curves_file_path = self._benchmark_dir_path / ShapeMatchingBenchmarkCurvesGeneratorTask.get_relative_file_path(curves_file_name=self._curves_file_name, sampling_ratio=1.0, multimodality=self._multimodality, group_name=self._group_name)
        query_curves = numpy.load(file=str(query_curves_file_path), allow_pickle=True)
        query_curve = PlanarCurve(points=query_curves[self._query_curve_id])
        database_curves = numpy.load(file=str(database_curves_file_path), allow_pickle=True)
        database_curve = PlanarCurve(points=database_curves[self._database_curve_id])
        score = self._planar_curves_signature_comparator.compare_signatures(curve1=query_curve, curve2=database_curve)
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
# PlanarCurvesShapeMatchingEvaluator Class
# =================================================
class PlanarCurvesShapeMatchingEvaluator(TaskParallelProcessor):
    def __init__(
            self,
            log_dir_path: Path,
            num_workers: int,
            curves_count_per_collection: int,
            curve_collections_file_names: List[str],
            benchmark_dir_path: Path,
            sampling_ratios: List[float],
            multimodalities: List[int],
            group_names: List[str],
            planar_curves_signature_comparator: PlanarCurvesSignatureComparator):
        self._num_workers = num_workers
        self._curves_count_per_collection = curves_count_per_collection
        self._curve_collections_file_names = [os.path.normpath(curve_collection_file_name) for curve_collection_file_name in curve_collections_file_names]
        self._benchmark_dir_path = benchmark_dir_path
        self._sampling_ratios = sampling_ratios
        self._multimodalities = multimodalities
        self._group_names = group_names
        self._planar_curves_signature_comparator = planar_curves_signature_comparator
        self._df = pandas.DataFrame()
        self._max_scores_df = pandas.DataFrame()
        self._shape_matching_df = pandas.DataFrame()
        super().__init__(log_dir_path=log_dir_path, num_workers=num_workers)

    @property
    def df(self) -> pandas.DataFrame:
        return self._df

    @property
    def shape_matching_df(self) -> pandas.DataFrame:
        return self._shape_matching_df

    def get_evaluation_score(self) -> float:
        return self._shape_matching_df['match'].mean()
    #
    # def _get_states_to_remove(self) -> List[str]:
    #     return ['_planar_curves_signature_comparator']

    # def _get_args(self) -> List[object]:
    #     return [self._planar_curves_signature_comparator]

    def _generate_tasks(self) -> List[ParallelProcessingTask]:
        tasks = []
        curve_ids = list(range(self._curves_count_per_collection))
        combinations = list(itertools.product(*[
            self._curve_collections_file_names,
            curve_ids,
            curve_ids,
            [self._benchmark_dir_path],
            self._sampling_ratios,
            self._multimodalities,
            self._group_names],
            [self._planar_curves_signature_comparator]))

        for combination in combinations:
            tasks.append(PlanarCurvesShapeMatchingEvaluatorTask(
                curves_file_name=combination[0],
                query_curve_id=combination[1],
                database_curve_id=combination[2],
                benchmark_dir_path=combination[3],
                sampling_ratio=combination[4],
                multimodality=combination[5],
                group_name=combination[6],
                planar_curves_signature_comparator=combination[7]))

        return tasks

    def _post_start(self):
        super()._post_start()
        df_list = []
        for completed_task in self._completed_tasks:
            completed_task = cast(typ=PlanarCurvesShapeMatchingEvaluatorTask, val=completed_task)
            df_list.append(completed_task.df)
        self._df = pandas.concat(df_list).reset_index()
        self._max_scores_df = self._df.loc[self._df.groupby(['curves_file_name', 'sampling_ratio', 'multimodality', 'group', 'query_curve_id'])['score'].idxmax()]
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
            group: Group):
        super().__init__(curves_count=curves_count)
        self._model = model
        self._supporting_points_count = supporting_points_count
        self._planar_curves_manager = planar_curves_manager
        self._group = group
        self._curves = []
        for i in range(self._curves_count):
            curve = self._planar_curves_manager.get_random_planar_curve()
            self._curves.append(curve)

    def evaluate_curves(self) -> List[Image]:
        device = torch.device('cpu')
        images = []
        for curve in self._curves:
            fig, axes = matplotlib.pyplot.subplots(nrows=4, ncols=1, figsize=(10, 20))
            axes[0].set_title(label='Curve', fontsize=30)
            axes[1].set_title(label='K', fontsize=30)
            axes[2].set_title(label='dK/ds', fontsize=30)
            axes[3].set_title(label='K vs. dK/ds', fontsize=30)
            for ax in axes:
                ax.tick_params(axis='both', which='major', labelsize=20)
                ax.tick_params(axis='both', which='minor', labelsize=20)
            curve.plot_scattered_curve(ax=axes[0])
            curve.plot_scattered_signature(model=self._model, supporting_points_count=self._supporting_points_count, device=device, ax=axes[1:])

            fig.canvas.draw()
            image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            images.append(image)
        # matplotlib.pyplot.show()
        return images
