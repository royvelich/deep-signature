# python peripherals
from __future__ import annotations
import random
from typing import Tuple, List, Optional
from abc import abstractmethod
import os
from pathlib import Path
import itertools

# numpy
import numpy

# tap
from tap import Tap

# deep_signature
from deep_signature.core import discrete_distributions
from deep_signature.manifolds.planar_curves.groups import EuclideanGroup, SimilarityGroup, EquiaffineGroup, AffineGroup

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

# deep_signature
from deep_signature.manifolds.planar_curves.implementation import PlanarCurve
from deep_signature.core.parallel_processing import TaskParallelProcessor, ParallelProcessingTask
from deep_signature.manifolds.planar_curves.groups import Group
from deep_signature.core.base import SeedableObject


# =================================================
# CirclesManager Class
# =================================================
# class CirclesManager(PlanarCurvesManager):
#     def __init__(self, curves_count: int, min_radius: float, max_radius: float, sampling_density: float):
#         self._min_radius = min_radius
#         self._max_radius = max_radius
#         self._sampling_density = sampling_density
#         super().__init__(curves_count=curves_count)
#
#     def _generate_curve(self, min_radius: float, max_radius: float, sampling_density: float):
#         radius = float(numpy.random.uniform(low=min_radius, high=max_radius, size=1))
#         circumference = 2 * radius * numpy.pi
#         points_count = int(numpy.round(sampling_density * circumference))
#         radians_delta = 2 * numpy.pi / points_count
#         pointer = numpy.array([radius, 0])
#         circle = numpy.empty((points_count, 2))
#         for i in range(points_count):
#             circle[i] = curve_processing.rotate_curve(curve=pointer, radians=i * radians_delta)
#
#         return circle
#
#     def _zip_iterable(self):
#         return [self._min_radius, self._max_radius, self._sampling_density] * self._curves_count


# =================================================
# CirclesGeneratorTask Class
# =================================================
class CirclesGeneratorTask(ParallelProcessingTask, SeedableObject):
    def __init__(self, min_radius: float, max_radius: float, min_sampling_density: float, max_sampling_density: float):
        super().__init__()
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._min_sampling_density = min_sampling_density
        self._max_sampling_density = max_sampling_density
        self._curves = []

    @property
    def curves(self) -> List[numpy.ndarray]:
        return self._curves

    def _pre_process(self):
        pass

    def _process(self):
        radius = float(self._rng.uniform(low=self._min_radius, high=self._max_radius, size=1))
        sampling_density = float(self._rng.uniform(low=self._min_sampling_density, high=self._max_sampling_density, size=1))
        circumference = 2 * radius * numpy.pi
        points_count = int(numpy.round(sampling_density * circumference))
        radians_delta = 2 * numpy.pi / points_count
        pointer = numpy.array([radius, 0])
        circle = numpy.empty((points_count, 2))
        pointer_curve = PlanarCurve(points=pointer, closed=False)
        for i in range(points_count):
            circle[i] = pointer_curve.rotate_curve(radians=i * radians_delta).points

        self._curves.append(circle)

    def _post_process(self):
        pass


class CirclesGenerator(TaskParallelProcessor):
    def __init__(
            self,
            log_dir_path: Path,
            num_workers: int,
            curves_base_dir_path: Path,
            circles_count: int,
            min_radius: float,
            max_radius: float,
            min_sampling_density: float,
            max_sampling_density: float):
        self._curves_base_dir_path = os.path.normpath(curves_base_dir_path)
        self._circles_count = circles_count
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._min_sampling_density = min_sampling_density
        self._max_sampling_density = max_sampling_density
        super().__init__(log_dir_path=log_dir_path, num_workers=num_workers, max_tasks=None)
        self._logger.info(msg=f'curves_base_dir_path: {self._curves_base_dir_path}')

    def _pre_join(self):
        pass

    def _post_join(self):
        curves = []
        for task in self._completed_tasks:
            curves.extend(task.curves)

        curves_file_path = os.path.normpath(os.path.join(self._curves_base_dir_path, 'curves.npy'))
        Path(self._curves_base_dir_path).mkdir(parents=True, exist_ok=True)
        numpy.save(file=curves_file_path, arr=curves, allow_pickle=True)

    def _generate_tasks(self) -> List[ParallelProcessingTask]:
        tasks = []
        combinations = list(itertools.product(*[
            list(range(self._circles_count)),
            [self._min_radius],
            [self._max_radius],
            [self._min_sampling_density],
            [self._max_sampling_density]]))

        for combination in combinations:
            tasks.append(CirclesGeneratorTask(
                min_radius=combination[1],
                max_radius=combination[2],
                min_sampling_density=combination[3],
                max_sampling_density=combination[4]))

        return tasks


# =================================================
# LevelCurvesGenerator Class
# =================================================
class LevelCurvesGeneratorTask(ParallelProcessingTask):
    def __init__(self, image_file_path: str, contour_level: float, min_points_count: int, max_points_count: int):
        super().__init__()
        self._image_file_path = image_file_path
        self._contour_level = contour_level
        self._min_points_count = min_points_count
        self._max_points_count = max_points_count
        self._curves = []

    @property
    def image_file_path(self) -> str:
        return self._image_file_path

    @property
    def contour_level(self) -> float:
        return self._contour_level

    @property
    def image_name(self) -> str:
        return Path(self._image_file_path).stem

    @property
    def curves(self) -> List[numpy.ndarray]:
        return self._curves

    def _process(self):
        image = self._load_image()
        preprocessed_image = self._preprocess_image(image=image)
        contours = self._extract_contours(image=preprocessed_image)
        self._curves = self._get_valid_contours(contours=contours)

    @abstractmethod
    def _get_valid_contours(self, contours: List[numpy.ndarray]) -> List[numpy.ndarray]:
        pass

    @abstractmethod
    def _preprocess_image(self, image: numpy.ndarray) -> numpy.ndarray:
        pass

    def _preprocess_contour(self, contour: numpy.ndarray) -> numpy.ndarray:
        curve = PlanarCurve(points=contour)
        curve = curve.center_curve()
        return curve.points

    def _load_image(self) -> numpy.ndarray:
        image = skimage.io.imread(self._image_file_path)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = skimage.color.rgba2rgb(image)
        elif len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)

        gray_image = skimage.color.rgb2gray(image)
        return gray_image

    def _extract_contours(self, image: numpy.ndarray) -> List[numpy.ndarray]:
        contours = skimage.measure.find_contours(image=image, level=self._contour_level)
        contours = [contour for contour in contours if self._min_points_count <= contour.shape[0] <= self._max_points_count]
        contours = [self._preprocess_contour(contour=contour) for contour in contours]
        return contours


class LevelCurvesGenerator(TaskParallelProcessor):
    def __init__(
            self,
            log_dir_path: Path,
            num_workers: int,
            images_base_dir_path: Path,
            curves_base_dir_path: Path,
            min_points_count: int,
            max_points_count: int,
            max_tasks: Optional[int] = None):
        self._images_base_dir_path = os.path.normpath(images_base_dir_path)
        self._curves_base_dir_path = os.path.normpath(curves_base_dir_path)
        self._min_points_count = min_points_count
        self._max_points_count = max_points_count
        self._image_file_paths = self._get_image_file_paths()
        super().__init__(log_dir_path=log_dir_path, num_workers=num_workers, max_tasks=max_tasks)
        self._logger.info(msg=f'images_base_dir_path: {self._images_base_dir_path}')
        self._logger.info(msg=f'curves_base_dir_path: {self._curves_base_dir_path}')

    @abstractmethod
    def _generate_tasks(self) -> List[ParallelProcessingTask]:
        pass

    def _get_image_file_paths(self) -> List[str]:
        image_file_paths = []
        for sub_dir_path, _, file_names in os.walk(self._images_base_dir_path):
            for file_name in file_names:
                image_file_path = os.path.normpath(os.path.join(sub_dir_path, file_name))
                image_file_paths.append(image_file_path)

        return image_file_paths


# =================================================
# ImageLevelCurvesGenerator Class
# =================================================
class ImageLevelCurvesGeneratorTask(LevelCurvesGeneratorTask):
    def __init__(self, image_file_path: str, contour_level: float, min_points_count: int, max_points_count: int, kernel_size: int, smoothing_iterations: int, smoothing_window_length: int, smoothing_poly_order: int, flat_point_threshold: float, max_flat_points_ratio: float, min_equiaffine_std: float):
        super().__init__(image_file_path=image_file_path, contour_level=contour_level, min_points_count=min_points_count, max_points_count=max_points_count)
        self._image_file_path = image_file_path
        self._kernel_size = kernel_size
        self._smoothing_iterations = smoothing_iterations
        self._smoothing_window_length = smoothing_window_length
        self._smoothing_poly_order = smoothing_poly_order
        self._flat_point_threshold = flat_point_threshold
        self._max_flat_points_ratio = max_flat_points_ratio
        self._min_equiaffine_std = min_equiaffine_std

    def _pre_process(self):
        pass

    def _post_process(self):
        pass

    def _preprocess_image(self, image: numpy.ndarray) -> numpy.ndarray:
        return cv2.GaussianBlur(src=image, ksize=(self._kernel_size, self._kernel_size), sigmaX=0)

    def _preprocess_contour(self, contour: numpy.ndarray) -> numpy.ndarray:
        contour = super()._preprocess_contour(contour=contour)
        curve = PlanarCurve(points=contour)
        curve = curve.smooth_curve(iterations=self._smoothing_iterations, window_length=self._smoothing_window_length, poly_order=self._smoothing_poly_order)
        return curve.points

    def _get_valid_contours(self, contours: List[numpy.ndarray]) -> List[numpy.ndarray]:
        valid_contours = []
        for contour in contours:
            # contour = contour[0:-1]
            planar_curve = PlanarCurve(points=contour)
            if planar_curve.closed is False:
                continue

            k_euclidean = planar_curve.calculate_euclidean_k()
            flat_points = numpy.sum(numpy.array([1 if x < self._flat_point_threshold else 0 for x in numpy.abs(k_euclidean)]))
            flat_points_ratio = flat_points / len(k_euclidean)
            if flat_points_ratio > self._max_flat_points_ratio:
                continue

            k_equiaffine = planar_curve.calculate_equiaffine_k()
            k_equiaffine_cleaned = k_equiaffine[~numpy.isnan(k_equiaffine)]
            equiaffine_std = numpy.std(k_equiaffine_cleaned)
            if equiaffine_std < self._min_equiaffine_std:
                continue

            valid_contours.append(planar_curve.points)
            break

        return valid_contours


class ImageLevelCurvesGenerator(LevelCurvesGenerator):
    def __init__(
            self,
            log_dir_path: Path,
            num_workers: int,
            images_base_dir_path: Path,
            curves_base_dir_path: Path,
            min_points_count: int,
            max_points_count: int,
            contour_levels: List[float],
            kernel_sizes: List[int],
            flat_point_threshold: float,
            max_flat_points_ratio: float,
            min_equiaffine_std: float,
            smoothing_iterations: int,
            smoothing_window_length: int,
            smoothing_poly_order: int,
            max_tasks: Optional[int] = None,
            curves_file_name: str = 'curves.npy'):
        self._kernel_sizes = kernel_sizes
        self._contour_levels = contour_levels
        self._flat_point_threshold = flat_point_threshold
        self._max_flat_points_ratio = max_flat_points_ratio
        self._min_equiaffine_std = min_equiaffine_std
        self._smoothing_iterations = smoothing_iterations
        self._smoothing_window_length = smoothing_window_length
        self._smoothing_poly_order = smoothing_poly_order
        self._curves_file_name = curves_file_name
        self._max_image_files = max_tasks
        super().__init__(log_dir_path=log_dir_path, num_workers=num_workers, images_base_dir_path=images_base_dir_path, curves_base_dir_path=curves_base_dir_path, min_points_count=min_points_count, max_points_count=max_points_count, max_tasks=max_tasks)

    def _pre_join(self):
        pass

    def _post_join(self):
        curves = []
        for task in self._completed_tasks:
            curves.extend(task.curves)

        curves_file_path = os.path.normpath(os.path.join(self._curves_base_dir_path, self._curves_file_name))
        Path(self._curves_base_dir_path).mkdir(parents=True, exist_ok=True)
        numpy.save(file=curves_file_path, arr=curves, allow_pickle=True)

    def _generate_tasks(self) -> List[ParallelProcessingTask]:
        tasks = []
        combinations = list(itertools.product(*[
            self._image_file_paths,
            self._contour_levels,
            [self._min_points_count],
            [self._max_points_count],
            self._kernel_sizes,
            [self._smoothing_iterations],
            [self._smoothing_window_length],
            [self._smoothing_poly_order],
            [self._flat_point_threshold],
            [self._max_flat_points_ratio],
            [self._min_equiaffine_std]]))

        for combination in combinations:
            tasks.append(ImageLevelCurvesGeneratorTask(
                image_file_path=combination[0],
                contour_level=combination[1],
                min_points_count=combination[2],
                max_points_count=combination[3],
                kernel_size=combination[4],
                smoothing_iterations=combination[5],
                smoothing_window_length=combination[6],
                smoothing_poly_order=combination[7],
                flat_point_threshold=combination[8],
                max_flat_points_ratio=combination[9],
                min_equiaffine_std=combination[10]))

        return tasks


# =================================================
# SilhouetteLevelCurvesGenerator Class
# =================================================
class SilhouetteLevelCurvesGeneratorTask(LevelCurvesGeneratorTask):
    def __init__(self, image_file_path: str, contour_level: float, min_points_count: int, max_points_count: int, curves_base_dir_path: str):
        super().__init__(image_file_path=image_file_path, contour_level=contour_level, min_points_count=min_points_count, max_points_count=max_points_count)
        self._curves_base_dir_path = curves_base_dir_path

    def _pre_process(self):
        pass

    def _post_process(self):
        curves_file_path = os.path.normpath(os.path.join(self._curves_base_dir_path, f'{self.image_name}.npy'))
        Path(self._curves_base_dir_path).mkdir(parents=True, exist_ok=True)
        if len(self._curves) >= 30:
            numpy.save(file=curves_file_path, arr=self._curves, allow_pickle=True)

    def _preprocess_image(self, image: numpy.ndarray) -> numpy.ndarray:
        image = cv2.copyMakeBorder(src=image, top=100, bottom=100, left=100, right=100, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return image

    def _get_valid_contours(self, contours: List[numpy.ndarray]) -> List[numpy.ndarray]:
        closed_contours = []
        for contour in contours:
            planar_curve = PlanarCurve(points=contour)
            if planar_curve.closed is True:
                closed_contours.append(contour)

        valid_contours = []
        for i, ref_contour in enumerate(closed_contours):
            ref_planar_curve = PlanarCurve(points=ref_contour, closed=True)
            valid = True
            for j, contour in enumerate(closed_contours):
                planar_curve = PlanarCurve(points=contour, closed=True)
                if (ref_planar_curve.is_within(planar_curve=planar_curve) is True) and (i != j):
                    valid = False

            if valid is True:
                valid_contours.append(ref_contour)

        return valid_contours


class SilhouetteLevelCurvesGenerator(LevelCurvesGenerator):
    def __init__(
            self,
            log_dir_path: Path,
            num_workers: int,
            images_base_dir_path: Path,
            curves_base_dir_path: Path,
            min_points_count: int,
            max_points_count: int,
            contour_level: float,
            max_tasks: Optional[int] = None):
        self._contour_level = contour_level
        super().__init__(
            log_dir_path=log_dir_path,
            num_workers=num_workers,
            images_base_dir_path=images_base_dir_path,
            curves_base_dir_path=curves_base_dir_path,
            min_points_count=min_points_count,
            max_points_count=max_points_count,
            max_tasks=max_tasks)

    def _generate_tasks(self) -> List[ParallelProcessingTask]:
        tasks = []
        for image_file_path in self._image_file_paths:
            tasks.append(SilhouetteLevelCurvesGeneratorTask(image_file_path=image_file_path, contour_level=self._contour_level, min_points_count=self._min_points_count, max_points_count=self._max_points_count, curves_base_dir_path=self._curves_base_dir_path))

        return tasks

    def _pre_join(self):
        pass

    def _post_join(self):
        pass


# =================================================
# ShapeMatchingBenchmarkCurvesGeneratorTask Class
# =================================================
class ShapeMatchingBenchmarkCurvesGeneratorTask(ParallelProcessingTask):
    def __init__(
            self,
            curves_file_path: str,
            benchmark_base_dir_path: str,
            sampling_ratio: float,
            multimodality: int,
            group_name: str,
            min_cond: float,
            max_cond: float,
            min_det: float,
            max_det: float,
            fig_size: Tuple[int, int],
            point_size: int):
        super().__init__()
        self._curves_file_path = curves_file_path
        self._benchmark_base_dir_path = benchmark_base_dir_path
        self._sampling_ratio = sampling_ratio
        self._multimodality = multimodality
        self._group_name = group_name
        self._min_cond = min_cond
        self._max_cond = max_cond
        self._min_det = min_det
        self._max_det = max_det
        self._group = self._create_group()
        self._sampled_planar_curves = []
        self._fig_size = fig_size
        self._point_size = point_size

    @property
    def sampled_planar_curves(self) -> List[numpy.ndarray]:
        return self._sampled_planar_curves

    @property
    def _curves_file_name(self) -> str:
        return Path(self._curves_file_path).stem

    def _create_group(self) -> Group:
        if self._group_name == 'euclidean':
            return EuclideanGroup(seed=None)
        if self._group_name == 'equiaffine':
            return EquiaffineGroup(min_cond=self._min_cond, max_cond=self._max_cond, seed=None)
        if self._group_name == 'similarity':
            return SimilarityGroup(min_det=self._min_det, max_det=self._max_det, seed=None)
        if self._group_name == 'affine':
            return AffineGroup(min_cond=self._min_cond, max_cond=self._max_cond, min_det=self._min_det, max_det=self._max_det, seed=None)

    def _pre_process(self):
        pass

    def _process(self):
        curves = numpy.load(file=self._curves_file_path, allow_pickle=True)
        for curve in curves:
            planar_curve = PlanarCurve(points=curve)
            discrete_distribution = discrete_distributions.MultimodalGaussianDiscreteDistribution(bins_count=planar_curve.points_count, multimodality=self._multimodality)
            sampled_planar_curve = planar_curve.sample_curve(sampling_ratio=self._sampling_ratio, discrete_distribution=discrete_distribution)
            if self._sampling_ratio < 1.0:
                transform = self._group.generate_random_group_action()
                sampled_planar_curve = sampled_planar_curve.transform_curve(transform=transform)
            self._sampled_planar_curves.append(sampled_planar_curve)

    def _post_process(self):
        relative_file_path = ShapeMatchingBenchmarkCurvesGeneratorTask.get_relative_file_path(curves_file_name=self._curves_file_name, group_name=self._group.name, multimodality=self._multimodality, sampling_ratio=self._sampling_ratio)
        curves_file_path = os.path.normpath(os.path.join(self._benchmark_base_dir_path, relative_file_path))
        Path(os.path.dirname(curves_file_path)).mkdir(parents=True, exist_ok=True)
        curves = [sampled_planar_curve.points for sampled_planar_curve in self._sampled_planar_curves]
        numpy.save(file=curves_file_path, arr=curves, allow_pickle=True)
        for curve_index, sampled_planar_curve in enumerate(self._sampled_planar_curves):
            fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=self._fig_size)
            sampled_planar_curve.reference_curve.plot_scattered_curve(ax=ax, cmap=None, color='red', point_size=self._point_size)
            sampled_planar_curve.plot_scattered_curve(ax=ax, cmap=None, color='green')
            matplotlib.pyplot.axis('off')
            ax.axis('equal')
            plot_file_dir_path = os.path.normpath(os.path.join('plots', relative_file_path.parent))
            self._save_fig(fig=fig, plot_file_dir_path=plot_file_dir_path, file_format='png', curve_index=curve_index)
            self._save_fig(fig=fig, plot_file_dir_path=plot_file_dir_path, file_format='svg', curve_index=curve_index)
            matplotlib.pyplot.close(fig)

    def _save_fig(self, fig: matplotlib.figure.Figure, plot_file_dir_path: str, file_format: str, curve_index: int):
        plot_file_path = os.path.normpath(os.path.join(self._benchmark_base_dir_path, f'{plot_file_dir_path}/{file_format}/{self._curves_file_name}_{curve_index}.{file_format}'))
        Path(os.path.dirname(plot_file_path)).mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_file_path, format=file_format)

    @staticmethod
    def get_relative_dir_path(curves_file_name: str, group_name: str, multimodality: int, sampling_ratio: float) -> Path:
        sampling_ratio_str = str(sampling_ratio).replace(".", "_")
        return Path(f'{curves_file_name}/{group_name}/{multimodality}/{sampling_ratio_str}')

    @staticmethod
    def get_relative_file_path(curves_file_name: str, group_name: str, multimodality: int, sampling_ratio: float) -> Path:
        relative_dir_path = ShapeMatchingBenchmarkCurvesGeneratorTask.get_relative_dir_path(curves_file_name=curves_file_name, group_name=group_name, multimodality=multimodality, sampling_ratio=sampling_ratio)
        return relative_dir_path / 'curves.npy'


# =================================================
# ShapeMatchingBenchmarkCurvesGenerator Class
# =================================================
class ShapeMatchingBenchmarkCurvesGenerator(TaskParallelProcessor):
    def __init__(
            self,
            log_dir_path: Path,
            num_workers: int,
            curves_base_dir_path: Path,
            benchmark_base_dir_path: Path,
            sampling_ratios: List[float],
            multimodalities: List[int],
            group_name: List[str],
            min_cond: float,
            max_cond: float,
            min_det: float,
            max_det: float,
            fig_size: Tuple[int, int],
            point_size: float):
        self._curves_base_dir_path = os.path.normpath(curves_base_dir_path)
        self._benchmark_base_dir_path = os.path.normpath(benchmark_base_dir_path)
        self._sampling_ratios = sampling_ratios
        self._multimodalities = multimodalities
        self._group_names = group_name
        self._min_cond = min_cond
        self._max_cond = max_cond
        self._min_det = min_det
        self._max_det = max_det
        self._fig_size = fig_size
        self._point_size = point_size
        self._curves_file_paths = self._get_curve_file_paths()
        super().__init__(log_dir_path=log_dir_path, num_workers=num_workers)

    def _generate_tasks(self) -> List[ParallelProcessingTask]:
        tasks = []
        combinations = list(itertools.product(*[
            self._curves_file_paths,
            [self._benchmark_base_dir_path],
            self._sampling_ratios,
            self._multimodalities,
            self._group_names,
            [self._min_cond],
            [self._max_cond],
            [self._min_det],
            [self._max_det],
            [self._fig_size],
            [self._point_size]]))

        for combination in combinations:
            tasks.append(ShapeMatchingBenchmarkCurvesGeneratorTask(
                curves_file_path=combination[0],
                benchmark_base_dir_path=combination[1],
                sampling_ratio=combination[2],
                multimodality=combination[3],
                group_name=combination[4],
                min_cond=combination[5],
                max_cond=combination[6],
                min_det=combination[7],
                max_det=combination[8],
                fig_size=combination[9],
                point_size=combination[10]))

        return tasks

    def _pre_join(self):
        pass

    def _post_join(self):
        pass

    def _get_curve_file_paths(self) -> List[Path]:
        curve_file_paths = []
        for sub_dir_path, _, file_names in os.walk(self._curves_base_dir_path):
            for file_name in file_names:
                image_file_path = Path(os.path.join(sub_dir_path, file_name))
                if image_file_path.suffix == '.npy':
                    curve_file_paths.append(image_file_path)

        return curve_file_paths
