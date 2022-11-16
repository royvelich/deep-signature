# python peripherals
from __future__ import annotations
from typing import Tuple, List
import os
import pathlib
import itertools
from abc import ABC, abstractmethod

# numpy
import numpy

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
from deep_signature.core.parallel_processing import ParallelProcessor, ParallelProcessorTask
from deep_signature.manifolds.planar_curves.groups import Group
from deep_signature.manifolds.planar_curves.generation import ShapeMatchingBenchmarkCurvesGeneratorTask


# =================================================
# PlanarCurvesSignatureComparator Class
# =================================================
class PlanarCurvesSignatureComparator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compare_signatures(self, curve1: PlanarCurve, curve2: PlanarCurve) -> float:
        pass


# =================================================
# PlanarCurvesSignatureComparator Class
# =================================================
class PlanarCurvesSignatureComparator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compare_signatures(self, curve1: PlanarCurve, curve2: PlanarCurve) -> float:
        pass


# =================================================
# ShapeMatchingBenchmarkCurvesGeneratorTask Class
# =================================================
class PlanarCurvesShapeMatchingEvaluatorTask(ParallelProcessorTask):
    def __init__(self, identifier: int, curves_file_name: str, benchmark_base_dir_path: str, sampling_ratio: float, multimodality: int, group: Group, model: torch.nn.Module):
        super().__init__(identifier=identifier)
        self._curves_file_name = curves_file_name
        self._benchmark_base_dir_path = benchmark_base_dir_path
        self._sampling_ratio = sampling_ratio
        self._multimodality = multimodality
        self._group = group
        self._model = model

    def process(self):
        query_curves_file_path = os.path.join(
            self._benchmark_base_dir_path,
            ShapeMatchingBenchmarkCurvesGeneratorTask.get_relative_dir_path(curves_file_name=self._curves_file_name, sampling_ratio=self._sampling_ratio, multimodality=self._multimodality, group=self._group))

        database_curves_file_path = os.path.join(
            self._benchmark_base_dir_path,
            ShapeMatchingBenchmarkCurvesGeneratorTask.get_relative_dir_path(curves_file_name=self._curves_file_name, sampling_ratio=1.0, multimodality=self._multimodality, group=self._group))

        query_curves = numpy.load(file=query_curves_file_path, allow_pickle=True)
        database_curves = numpy.load(file=database_curves_file_path, allow_pickle=True)


    def post_process(self):
        pass


# =================================================
# PlanarCurvesShapeMatchingEvaluator Class
# =================================================
class PlanarCurvesShapeMatchingEvaluator(ParallelProcessor):
    def __init__(
            self,
            num_workers: int,
            curves_file_names: List[str],
            benchmark_base_dir_path: str,
            sampling_ratios: List[float],
            multimodalities: List[int],
            groups: List[Group],
            model: torch.nn.Module):
        self._num_workers = num_workers
        self._curves_file_names = [os.path.normpath(curves_file_name) for curves_file_name in curves_file_names]
        self._benchmark_base_dir_path = os.path.normpath(benchmark_base_dir_path)
        self._sampling_ratios = sampling_ratios
        self._multimodalities = multimodalities
        self._groups = groups
        self._model = model
        super().__init__(num_workers=num_workers)

    def _generate_tasks(self) -> List[ParallelProcessorTask]:
        tasks = []
        combinations = list(itertools.product(*[
            self._curves_file_names,
            [self._benchmark_base_dir_path],
            self._sampling_ratios,
            self._multimodalities,
            self._groups,
            [self._model]]))

        for combination in combinations:
            tasks.append(PlanarCurvesShapeMatchingEvaluatorTask(
                curves_file_name=combination[0],
                benchmark_base_dir_path=combination[1],
                sampling_ratio=combination[2],
                multimodality=combination[3],
                group=combination[4],
                model=combination[5]))

        return tasks