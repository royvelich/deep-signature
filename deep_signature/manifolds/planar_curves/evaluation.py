# python peripherals
from __future__ import annotations
from typing import Tuple, List, cast
import os
import pathlib
import itertools
from abc import ABC, abstractmethod

# numpy
import numpy

# pandas
import pandas

# deep_signature
from deep_signature.core import discrete_distributions

# skimage
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure

# opencv
import cv2

# skimage
import skimage.io
import skimage.color
import skimage.measure

# matplotlib
import matplotlib
import matplotlib.axes
import matplotlib.figure

# pytorch
import torch
import torch.nn

# deep_signature
from deep_signature.manifolds.planar_curves.implementation import PlanarCurve
from deep_signature.core.parallel_processing import TaskParallelProcessor, ParallelProcessingTask
from deep_signature.manifolds.planar_curves.groups import Group
from deep_signature.manifolds.planar_curves.generation import ShapeMatchingBenchmarkCurvesGeneratorTask
from deep_signature.metrics import hausdorff


# =================================================
# PlanarCurvesSignatureComparator Class
# =================================================
class PlanarCurvesSignatureComparator(ABC):
    def __init__(self):
        pass

    def compare_signatures(self, curve1: PlanarCurve, curve2: PlanarCurve) -> float:
        curve1_signature = self._calculate_signature(curve=curve1)
        curve2_signature = self._calculate_signature(curve=curve2)
        return hausdorff.hausdorff_distance(subset1=curve1_signature, curve2=curve2_signature)

    @abstractmethod
    def _calculate_signature(self, curve: PlanarCurve) -> numpy.ndarray:
        pass


# =================================================
# PlanarCurvesSignatureComparator Class
# =================================================
class PlanarCurvesApproximatedLocalSignatureComparator(PlanarCurvesSignatureComparator):
    def __init__(self, model: torch.nn.Module, supporting_points_count: int, device: torch.device):
        self._model = model
        self._supporting_points_count = supporting_points_count
        self._device = device
        super().__init__()

    def _calculate_signature(self, curve: PlanarCurve) -> numpy.ndarray:
        return curve.predict_curve_local_signature(model=self._model, supporting_points_count=self._supporting_points_count, device=self._device)


# =================================================
# ShapeMatchingBenchmarkCurvesGeneratorTask Class
# =================================================
class PlanarCurvesShapeMatchingEvaluatorTask(ParallelProcessingTask):
    def __init__(
            self,
            curves_file_name: str,
            query_curve_id: int,
            database_curve_id: int,
            benchmark_base_dir_path: str,
            sampling_ratio: float,
            multimodality: int,
            group: Group,
            planar_curves_signature_comparator: PlanarCurvesSignatureComparator):
        super().__init__()
        self._curves_file_name = curves_file_name
        self._query_curve_id = query_curve_id
        self._database_curve_id = database_curve_id
        self._benchmark_base_dir_path = benchmark_base_dir_path
        self._sampling_ratio = sampling_ratio
        self._multimodality = multimodality
        self._group = group
        self._planar_curves_signature_comparator = planar_curves_signature_comparator
        self._df = pandas.DataFrame()

    @property
    def df(self) -> pandas.DataFrame:
        return self._df

    def _pre_process(self):
        pass

    def _process(self):
        query_curves_file_path = os.path.join(
            self._benchmark_base_dir_path,
            ShapeMatchingBenchmarkCurvesGeneratorTask.get_relative_dir_path(curves_file_name=self._curves_file_name, sampling_ratio=self._sampling_ratio, multimodality=self._multimodality, group=self._group))

        database_curves_file_path = os.path.join(
            self._benchmark_base_dir_path,
            ShapeMatchingBenchmarkCurvesGeneratorTask.get_relative_dir_path(curves_file_name=self._curves_file_name, sampling_ratio=1.0, multimodality=self._multimodality, group=self._group))

        query_curves = numpy.load(file=query_curves_file_path, allow_pickle=True)
        query_curve = query_curves[self._query_curve_id]

        database_curves = numpy.load(file=database_curves_file_path, allow_pickle=True)
        database_curve = database_curves[self._database_curve_id]

        signature_comparison = self._planar_curves_signature_comparator.compare_signatures(curve1=query_curve, curve2=database_curve)

        data = {
            'curves_file_name': [self._curves_file_name],
            'sampling_ratio': [self._sampling_ratio],
            'multimodality': [self._multimodality],
            'group': [self._group],
            'query_curve_id': [self._query_curve_id],
            'database_curve_id': [self._database_curve_id],
            'signature_comparison': [signature_comparison]
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
            num_workers: int,
            curves_count_per_collection: int,
            curve_collections_file_names: List[str],
            benchmark_base_dir_path: str,
            sampling_ratios: List[float],
            multimodalities: List[int],
            groups: List[Group],
            planar_curves_signature_comparator: PlanarCurvesSignatureComparator):
        self._num_workers = num_workers
        self._curves_count_per_collection = curves_count_per_collection
        self._curve_collections_file_names = [os.path.normpath(curve_collection_file_name) for curve_collection_file_name in curve_collections_file_names]
        self._benchmark_base_dir_path = os.path.normpath(benchmark_base_dir_path)
        self._sampling_ratios = sampling_ratios
        self._multimodalities = multimodalities
        self._groups = groups
        self._planar_curves_signature_comparator = planar_curves_signature_comparator
        self._df = pandas.DataFrame()
        super().__init__(num_workers=num_workers)

    @property
    def df(self) -> pandas.DataFrame:
        return self._df

    def _generate_tasks(self) -> List[ParallelProcessingTask]:
        tasks = []
        curve_ids = list(range(self._curves_count_per_collection))
        combinations = list(itertools.product(*[
            self._curve_collections_file_names,
            curve_ids,
            curve_ids,
            [self._benchmark_base_dir_path],
            self._sampling_ratios,
            self._multimodalities,
            self._groups,
            [self._planar_curves_signature_comparator]]))

        for combination in combinations:
            tasks.append(PlanarCurvesShapeMatchingEvaluatorTask(
                curves_file_name=combination[0],
                query_curve_id=combination[1],
                database_curve_id=combination[2],
                benchmark_base_dir_path=combination[3],
                sampling_ratio=combination[4],
                multimodality=combination[5],
                group=combination[6],
                planar_curves_signature_comparator=combination[7]))

        return tasks

    def _post_start(self):
        super()._post_start()

        df_list = []
        for completed_task in self._completed_tasks:
            completed_task = cast(typ=PlanarCurvesShapeMatchingEvaluatorTask, val=completed_task)
            df_list.append(completed_task.df)
        self._df = pandas.concat(df_list)
