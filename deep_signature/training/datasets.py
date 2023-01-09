# python peripherals
import numpy
from abc import abstractmethod
import random
from pathlib import Path
from typing import Optional

# torch
from torch.utils.data import Dataset

# deep_signature
from deep_signature.manifolds.planar_curves.implementation import PlanarCurvesManager, PlanarCurve
from deep_signature.core.discrete_distributions import MultimodalGaussianDiscreteDistribution
from deep_signature.manifolds.planar_curves.groups import Group
from deep_signature.core.base import SeedableObject
from deep_signature.core.parallel_processing import InfiniteOnlineParallelProcessor, GetItemPolicy


class TupletsDataset(Dataset, SeedableObject):
    def __init__(self, planar_curves_manager: PlanarCurvesManager, group: Group, dataset_size: int, **kw: object):
        super().__init__(**kw)
        self._planar_curves_manager = planar_curves_manager
        self._group = group
        self._dataset_size = dataset_size

    def __len__(self):
        return self._dataset_size

    @abstractmethod
    def __getitem__(self, index: int):
        pass


class CurveNeighborhoodTupletsDataset(InfiniteOnlineParallelProcessor, TupletsDataset):
    def __init__(
            self,
            planar_curves_manager: PlanarCurvesManager,
            group: Group,
            dataset_size: int,
            supporting_points_count: int,
            negative_examples_count: int,
            min_sampling_ratio: float,
            max_sampling_ratio: float,
            min_multimodality: int,
            max_multimodality: int,
            min_negative_example_offset: int,
            max_negative_example_offset: int,
            log_dir_path: Path,
            num_workers: int,
            items_queue_maxsize: int,
            items_buffer_size: int,
            get_item_policy: GetItemPolicy,
            add_flip_as_negative_example: bool):
        super().__init__(
            planar_curves_manager=planar_curves_manager,
            group=group,
            dataset_size=dataset_size,
            log_dir_path=log_dir_path,
            num_workers=num_workers,
            items_queue_maxsize=items_queue_maxsize,
            items_buffer_size=items_buffer_size,
            get_item_policy=get_item_policy)
        self._supporting_points_count = supporting_points_count
        self._negative_examples_count = negative_examples_count
        self._min_sampling_ratio = min_sampling_ratio
        self._max_sampling_ratio = max_sampling_ratio
        self._min_multimodality = min_multimodality
        self._max_multimodality = max_multimodality
        self._min_negative_example_offset = min_negative_example_offset
        self._max_negative_example_offset = max_negative_example_offset
        self._add_flip_as_negative_example = add_flip_as_negative_example

    def _generate_item(self, item_id: Optional[int]) -> object:
        tuplet = []
        planar_curve = self._planar_curves_manager.get_random_planar_curve()
        center_point_index = planar_curve.get_random_point_index()
        for i in range(2):
            sampled_planar_curve = self._sample_planar_curve(planar_curve=planar_curve)
            curve_neighborhood_points = self._extract_curve_neighborhood_points(planar_curve=sampled_planar_curve, center_point_index=center_point_index)
            tuplet.append(curve_neighborhood_points)

        # if self._add_flip_as_negative_example:
        #     flipped_anchor_example = self._flip_curve_neighborhood_points(points=tuplet[0])
        #     tuplet.append(flipped_anchor_example)

        # for i in range(self._negative_examples_count):
        #     sampled_planar_curve = self._sample_planar_curve(planar_curve=planar_curve)
        #     negative_example_offset = self._rng.integers(low=self._min_negative_example_offset, high=self._max_negative_example_offset)
        #     if bool(random.getrandbits(1)) is True:
        #         negative_example_offset = -negative_example_offset
        #     negative_example_center_point_index = numpy.mod(center_point_index + negative_example_offset, planar_curve.points_count)
        #     curve_neighborhood_points = self._extract_curve_neighborhood_points(planar_curve=sampled_planar_curve, center_point_index=negative_example_center_point_index)
        #     tuplet.append(curve_neighborhood_points)

        for i in range(self._negative_examples_count):
            planar_curve = self._planar_curves_manager.get_random_planar_curve()
            center_point_index = planar_curve.get_random_point_index()
            sampled_planar_curve = self._sample_planar_curve(planar_curve=planar_curve)
            curve_neighborhood_points = self._extract_curve_neighborhood_points(planar_curve=sampled_planar_curve, center_point_index=center_point_index)
            tuplet.append(curve_neighborhood_points)

        return numpy.array(tuplet)

    def _sample_planar_curve(self, planar_curve: PlanarCurve) -> PlanarCurve:
        multimodality = self._rng.integers(low=self._min_multimodality, high=self._max_multimodality)
        sampling_ratio = self._rng.uniform(low=self._min_sampling_ratio, high=self._max_sampling_ratio)
        discrete_distribution = MultimodalGaussianDiscreteDistribution(bins_count=planar_curve.points_count, multimodality=multimodality)
        return planar_curve.sample_curve(sampling_ratio=sampling_ratio, discrete_distribution=discrete_distribution)

    def _extract_curve_neighborhood_points(self, planar_curve: PlanarCurve, center_point_index: int) -> numpy.ndarray:
        group_action = self._group.generate_random_group_action()
        curve_neighborhood = planar_curve.extract_curve_neighborhood(center_point_index=center_point_index, supporting_points_count=self._supporting_points_count)
        curve_neighborhood = curve_neighborhood.transform_curve(transform=group_action)
        curve_neighborhood = curve_neighborhood.normalize_curve(force_ccw=False, force_endpoint=False)
        return curve_neighborhood.points

    def _flip_curve_neighborhood_points(self, points: numpy.ndarray) -> numpy.ndarray:
        curve = PlanarCurve(points=points, closed=False)
        curve = curve.normalize_curve(force_ccw=False, force_endpoint=False)
        curve = curve.reflect_curve_horizontally()
        return curve.points
