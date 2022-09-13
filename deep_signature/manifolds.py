# python peripherals
from __future__ import annotations
from typing import Dict, Union, Tuple, List, Callable
from abc import ABC, abstractmethod
import os
import pathlib
import inspect
import random

# numpy
import numpy

# scipy
import scipy.signal

# sklearn
import sklearn.preprocessing

# itertools
from itertools import combinations

# deep_signature
from deep_signature import discrete_distributions
from deep_signature.linalg import transformations
from deep_signature.discrete_distributions import DiscreteDistribution

# skimage
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure

# matplotlib
import matplotlib

# deep_signature
from deep_signature import utils
from deep_signature.parallel_processing import ParallelProcessor
import deep_signature.visualization


# =================================================
# PlanarCurve Class
# =================================================
class PlanarCurve:
    def __init__(self, points: numpy.ndarray, closed: Union[None, bool] = None):
        self._points = points

        if closed is None:
            self._closed = self.is_closed()
        else:
            self._closed = closed

        self._reference_curve = self
        self._reference_indices = numpy.array(list(range(self._points.shape[0])))
        self._discrete_distribution = None

    # -------------------------------------------------
    # properties
    # -------------------------------------------------

    @property
    def points(self) -> numpy.ndarray:
        return self._points

    @property
    def points_count(self) -> int:
        return self._points.shape[0]

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def reference_curve(self) -> PlanarCurve:
        return self._reference_curve

    @property
    def reference_indices(self) -> numpy.ndarray:
        return self._reference_indices

    @property
    def discrete_distribution(self) -> Union[None, DiscreteDistribution]:
        return self._discrete_distribution

    # -------------------------------------------------
    # padding and unpadding
    # -------------------------------------------------

    @staticmethod
    def _pad_array(array: numpy.ndarray, padding: int = 1, mode: str = 'wrap') -> numpy.ndarray:
        if array.ndim == 2:
            return numpy.pad(array=array, pad_width=((padding, padding), (0, 0)), mode=mode)
        else:
            return numpy.pad(array=array, pad_width=(padding, padding), mode=mode)

    @staticmethod
    def _unpad_array(array: numpy.ndarray, padding: int = 1) -> numpy.ndarray:
        return array[padding:-padding]

    # -------------------------------------------------
    # gradients
    # -------------------------------------------------

    @staticmethod
    def _calculate_gradient(array: numpy.ndarray, closed: bool, normalize: bool = True) -> numpy.ndarray:
        if closed:
            array = PlanarCurve._pad_array(array=array)

        array_diff = numpy.gradient(f=array, axis=0)
        if not normalize:
            array_diff = 2 * array_diff
            if not closed:
                array_diff[0] = array_diff[0] / 2
                array_diff[-1] = array_diff[-1] / 2

        if closed:
            array_diff = PlanarCurve._unpad_array(array=array_diff)

        return array_diff

    # -------------------------------------------------
    # tangents and normals
    # -------------------------------------------------

    def calculate_tangents(self) -> numpy.ndarray:
        return PlanarCurve._calculate_gradient(array=self._points, closed=self._closed)

    def calculate_normals(self) -> numpy.ndarray:
        tangents = PlanarCurve._calculate_gradient(array=self._points, closed=self._closed)
        normals = numpy.transpose(a=tangents, axes=(1, 0))
        normals[:, 1] = -normals[:, 1]
        return normals

    # -------------------------------------------------
    # euclidean curvature and arclength approximation
    # -------------------------------------------------

    def calculate_euclidean_k(self) -> numpy.ndarray:
        tangents = PlanarCurve._calculate_gradient(array=self._points, closed=self._closed)
        dtangents_dt = PlanarCurve._calculate_gradient(array=tangents, closed=self._closed)
        dx_dt = tangents[:, 0]
        dy_dt = tangents[:, 1]
        d2x_dt2 = dtangents_dt[:, 0]
        d2y_dt2 = dtangents_dt[:, 1]
        k = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        return k

    def calculate_euclidean_s(self) -> numpy.ndarray:
        secants = numpy.diff(a=self._points, n=1, axis=0)
        secants_norm = numpy.linalg.norm(x=secants, ord=2, axis=1)
        s = numpy.cumsum(a=numpy.concatenate((numpy.array([0]), secants_norm), axis=0))
        return s

    def calculate_euclidean_dk_ds(self) -> numpy.ndarray:
        return self._calculate_dk_ds(calculate_k=self.calculate_euclidean_k, calculate_s=self.calculate_euclidean_s)

    # -------------------------------------------------
    # equiaffine curvature and arclength approximation
    # https://link.springer.com/article/10.1023/A:1007992709392
    # -------------------------------------------------
    def _calculate_signed_parallelogram_area(self, indices: numpy.ndarray) -> float:
        if indices.shape[0] == 4:
            return self._calculate_signed_parallelogram_area(indices=indices[[0, 1, 3]]) - self._calculate_signed_parallelogram_area(indices=indices[[0, 1, 2]])

        points = self._points[indices]
        points = numpy.hstack((points, numpy.ones((points.shape[0], 1), dtype=points.dtype)))
        return numpy.linalg.det(points)

    def _calculate_T(self, indices: numpy.ndarray) -> float:
        T = 1
        for item in combinations(indices, 3):
            sorted_indices = sorted(item)
            T = T * self._calculate_signed_parallelogram_area(indices=numpy.array(sorted_indices))
        return T / 4

    def _calculate_S(self, indices: numpy.ndarray) -> float:
        c013 = self._calculate_signed_parallelogram_area(indices=indices[[0, 1, 3]])
        c024 = self._calculate_signed_parallelogram_area(indices=indices[[0, 2, 4]])
        c1234 = self._calculate_signed_parallelogram_area(indices=indices[[1, 2, 3, 4]])
        c012 = self._calculate_signed_parallelogram_area(indices=indices[[0, 1, 2]])
        c034 = self._calculate_signed_parallelogram_area(indices=indices[[0, 3, 4]])
        c1324 = self._calculate_signed_parallelogram_area(indices=indices[[1, 3, 2, 4]])
        c123 = self._calculate_signed_parallelogram_area(indices=indices[[1, 2, 3]])
        c234 = self._calculate_signed_parallelogram_area(indices=indices[[2, 3, 4]])
        c124 = self._calculate_signed_parallelogram_area(indices=indices[[1, 2, 4]])
        c134 = self._calculate_signed_parallelogram_area(indices=indices[[1, 3, 4]])

        t1 = numpy.square(c013) * numpy.square(c024) * numpy.square(c1234)
        t2 = numpy.square(c012) * numpy.square(c034) * numpy.square(c1324)
        t3 = 2 * c012 * c034 * c013 * c024 * ((c123 * c234) + (c124 * c134))
        return (t1 + t2 - t3) / 4

    def _calculate_N(self, indices: numpy.ndarray) -> float:
        c123 = self._calculate_signed_parallelogram_area(indices=indices[[1, 2, 3]])
        c134 = self._calculate_signed_parallelogram_area(indices=indices[[1, 3, 4]])
        c023 = self._calculate_signed_parallelogram_area(indices=indices[[0, 2, 3]])
        c014 = self._calculate_signed_parallelogram_area(indices=indices[[0, 1, 4]])
        c1234 = self._calculate_signed_parallelogram_area(indices=indices[[1, 2, 3, 4]])
        c012 = self._calculate_signed_parallelogram_area(indices=indices[[0, 1, 2]])
        c034 = self._calculate_signed_parallelogram_area(indices=indices[[0, 3, 4]])
        c2314 = self._calculate_signed_parallelogram_area(indices=indices[[2, 3, 1, 4]])

        t1 = -c123 * c134
        t2 = numpy.square(c023) * numpy.square(c014) * c1234
        t3 = numpy.square(c012) * numpy.square(c034) * c2314
        t4 = c012 * c023 * c014 * c034 * (c134 - c123)
        return (t1 * (t2 + t3 + t4)) / 4

    def _calculate_elliptic_area(self, indices: numpy.ndarray) -> float:
        S = self._calculate_S(indices=indices)
        N = self._calculate_N(indices=indices)
        return N / (2 * S)

    def _calculate_equiaffine_curvature_at_point(self, indices: numpy.ndarray) -> float:
        T = self._calculate_T(indices=indices)
        S = self._calculate_S(indices=indices)
        return S / numpy.power(T, 2 / 3)

    def _calculate_equiaffine_arclength_for_quintuple(self, indices):
        k = self._calculate_equiaffine_curvature_at_point(indices=indices)
        area = self._calculate_elliptic_area(indices=indices)
        return 2 * area * numpy.abs(k)

    # def calculate_equiaffine_arclength(self):
    #     s = numpy.zeros(curve.shape[0])
    #     for i in numpy.arange(start=1, stop=curve.shape[0] - 1, step=3):
    #         indices = numpy.arange(start=i - 1, stop=i + 4)
    #         s[i] = s[i - 1] + calculate_equiaffine_arclength_for_quintuple(curve=curve, indices=indices)
    #     return s

    def calculate_equiaffine_k(self):
        k = numpy.zeros(self._points.shape[0])
        for i in range(self._points.shape[0]):
            indices = numpy.array(list(range((i - 2), (i + 3))))
            indices = numpy.mod(indices, self._points.shape[0])
            k[i] = self._calculate_equiaffine_curvature_at_point(indices=indices)

        if not self._closed:
            k[0] = numpy.nan
            k[1] = numpy.nan
            k[-1] = numpy.nan
            k[-2] = numpy.nan

        return k

    def calculate_equiaffine_s(self):
        s_euclidean = self.calculate_euclidean_s()
        k_euclidean = numpy.abs(self.calculate_euclidean_k())
        ds_euclidean = numpy.diff(a=s_euclidean, n=1, axis=0)
        ds_euclidean = numpy.concatenate((numpy.array([0]), ds_euclidean))
        k_euclidean_cbrt = numpy.cbrt(k_euclidean)
        ds_equiaffine = k_euclidean_cbrt * ds_euclidean
        s = numpy.cumsum(a=ds_equiaffine, axis=0)
        return s

    def calculate_equiaffine_dk_ds(self) -> numpy.ndarray:
        return self._calculate_dk_ds(calculate_k=self.calculate_equiaffine_k, calculate_s=self.calculate_equiaffine_s)

    # -------------------------------------------------
    # general curvature and arclength approximation
    # -------------------------------------------------

    def _calculate_dk_ds(self, calculate_k: Callable[[], numpy.ndarray], calculate_s: Callable[[], numpy.ndarray]) -> numpy.ndarray:
        k = calculate_k()
        s = calculate_s()
        dk = PlanarCurve._calculate_gradient(array=k, closed=self._closed, normalize=False)
        ds = PlanarCurve._calculate_gradient(array=s, closed=self._closed, normalize=False)
        ks = dk / ds
        return ks

    # -------------------------------------------------
    # curve smoothing
    # -------------------------------------------------

    def smooth_curve(self, iterations: int, window_length: int, poly_order: int):
        smoothed_points = numpy.copy(self._points)
        for _ in range(iterations):
            x = smoothed_points[:, 0]
            y = smoothed_points[:, 1]
            smoothed_points[:, 0] = scipy.signal.savgol_filter(x=x, window_length=window_length, polyorder=poly_order, mode='wrap')
            smoothed_points[:, 1] = scipy.signal.savgol_filter(x=y, window_length=window_length, polyorder=poly_order, mode='wrap')

    # -------------------------------------------------
    # curve evolution
    # -------------------------------------------------

    def evolve_curve(self, evolution_iterations: int, evolution_dt: float, smoothing_window_length: int, smoothing_poly_order: int, smoothing_iterations: int):
        for _ in range(evolution_iterations):
            k = self.calculate_euclidean_k()
            normal = self.calculate_normals()
            normal = sklearn.preprocessing.normalize(X=normal, axis=1, norm='l2')
            delta = normal * k[:, numpy.newaxis]
            self._points = self._points + evolution_dt * delta
            self.smooth_curve(iterations=smoothing_iterations, window_length=smoothing_window_length, poly_order=smoothing_poly_order)

    # -------------------------------------------------
    # curve sampling
    # -------------------------------------------------

    def sample_curve(self, sampling_ratio: float, discrete_distribution: DiscreteDistribution) -> PlanarCurve:
        points_count = self._points.shape[0]
        sampling_points_count = int(sampling_ratio * points_count)
        indices = discrete_distribution.sample_pdf(samples_count=sampling_points_count)
        sampled_planar_curve = PlanarCurve(points=self._points[indices], closed=self._closed)
        sampled_planar_curve._reference_curve = self
        sampled_planar_curve._reference_indices = indices
        sampled_planar_curve._discrete_distribution = discrete_distribution
        return sampled_planar_curve

    def extract_curve_neighborhood(self, center_point_index: int, supporting_points_count: int) -> PlanarCurve:
        reference_curve_indices = self._reference_indices.copy()
        reference_curve_indices = numpy.append(arr=reference_curve_indices, values=[center_point_index])
        reference_curve_indices = numpy.unique(ar=reference_curve_indices)
        center_point_meta_index = numpy.where(reference_curve_indices == center_point_index)[0]
        curve_neighborhood_meta_indices = numpy.arange(start=center_point_meta_index - supporting_points_count, stop=center_point_meta_index + supporting_points_count + 1)
        curve_neighborhood_meta_indices = numpy.mod(curve_neighborhood_meta_indices, reference_curve_indices.shape[0])
        curve_neighborhood_indices = reference_curve_indices[curve_neighborhood_meta_indices]
        curve_neighborhood_points = self._reference_curve.points[curve_neighborhood_indices]
        return PlanarCurve(points=curve_neighborhood_points, closed=False)

    # def extract_curve_section(self, start_point_index: int, end_point_index: int, supporting_points_count: int) -> PlanarCurve:
    #     if rng is None:
    #         rng = numpy.random.default_rng()
    #
    #     curve_points_count = curve.shape[0]
    #
    #     if start_point_index > end_point_index:
    #         start_point_index = -(curve_points_count - start_point_index)
    #         # end_point_index = -end_point_index
    #
    #     indices_pool = numpy.mod(numpy.array(list(range(start_point_index + 1, end_point_index))), curve.shape[0])
    #     bins = int(numpy.abs(end_point_index - start_point_index) + 1) - 2
    #     supporting_points_count = supporting_points_count - 2 if supporting_points_count is not None else bins
    #
    #     # uniform = False
    #     if uniform is False:
    #         dist = discrete_distribution.random_discrete_dist(bins=bins, multimodality=multimodality, max_density=1, count=1)[0]
    #         meta_indices = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=supporting_points_count)
    #     else:
    #         meta_indices = list(range(end_point_index - start_point_index - 1))
    #         meta_indices = numpy.sort(rng.choice(a=meta_indices, size=supporting_points_count, replace=False))
    #
    #     indices = indices_pool[meta_indices]
    #     indices = numpy.concatenate(([numpy.mod(start_point_index, curve.shape[0])], indices, [numpy.mod(end_point_index, curve.shape[0])]))
    #
    #     return indices

    def get_random_point_index(self) -> int:
        return numpy.random.randint(low=0, high=self._points.shape[0])

    # -------------------------------------------------
    # curve transformation
    # -------------------------------------------------
    
    def center_curve(self):
        center_of_mass = numpy.mean(self._points, axis=0)
        return self.translate_curve(offset=-center_of_mass)

    def translate_curve(self, offset: numpy.ndarray):
        self._points = self._points + offset

    def rotate_curve(self, radians: float):
        transform = transformations.generate_rotation_transform_2d(radians)
        self._points = self._points.dot(transform)

    def reflect_curve_horizontally(self):
        transform = transformations.generate_horizontal_reflection_transform_2d()
        self._points = self._points.dot(transform)

    def reflect_curve_vertically(self):
        transform = transformations.generate_vertical_reflection_transform_2d()
        self._points = self._points.dot(transform)

    def transform_curve(self, transform: numpy.ndarray):
        self._points = self._points.dot(transform)

    def orient_cw(self):
        if self.is_ccw():
            self._points = numpy.flip(m=self._points, axis=0)

    def orient_ccw(self):
        if self.is_cw():
            self._points = numpy.flip(m=self._points, axis=0)

    # -------------------------------------------------
    # curve queries
    # -------------------------------------------------
    def is_ccw(self) -> bool:
        index1 = self._get_first_index()
        index2 = self._get_middle_index()
        index3 = self._get_last_index()
        eps = 1e-9
        pointer01 = self._points[index1] - self._points[index2]
        pointer12 = self._points[index3] - self._points[index2]
        pointer12 = pointer12 / (numpy.linalg.norm(pointer12) + eps)
        normal01 = numpy.array([-pointer01[1], pointer01[0]])
        normal01 = normal01 / (numpy.linalg.norm(normal01) + eps)
        return numpy.dot(pointer12, normal01) < 0

    def is_cw(self) -> bool:
        return not self.is_ccw()

    def is_closed(self) -> bool:
        eps = 1e-2
        first_point = self._points[0]
        last_point = self._points[-1]
        distance = numpy.linalg.norm(x=first_point - last_point, ord=2)
        if distance < eps:
            return True

        return False

    # -------------------------------------------------
    # curve normalization
    # -------------------------------------------------

    def normalize_curve(self, force_ccw: bool, force_endpoint: bool):
        if force_ccw is True:
            self.orient_ccw()

        self.translate_curve(offset=-self._points[0])

        radians = self._calculate_secant_angle()
        self.rotate_curve(radians=-radians)

        if force_endpoint is True:
            end_point = self._points[-1]
            if end_point[0] < 0:
                transform = transformations.generate_vertical_reflection_transform_2d()
                self._points = numpy.matmul(self._points, transform)

            if end_point[1] < 0:
                transform = transformations.generate_horizontal_reflection_transform_2d()
                self._points = numpy.matmul(self._points, transform)

    def _calculate_secant_angle(self) -> float:
        index1 = self._get_first_index()
        index2 = self._get_middle_index()
        secant = self._points[index2] - self._points[index1]
        return numpy.arctan2(secant[1], secant[0])

    def _get_first_index(self) -> int:
        return 0

    def _get_middle_index(self) -> int:
        return int(numpy.floor(self._points.shape[0] / 2))

    def _get_last_index(self) -> int:
        return self._points.shape[0] - 1

    # -------------------------------------------------
    # curve visualization
    # -------------------------------------------------
    def plot_scattered_curve(self, ax: matplotlib.axes.Axes, point_size: float = 2, alpha: float = 1, cmap: str = 'red', zorder: int = 1):
        x = self._points[:, 0]
        y = self._points[:, 1]
        color_indices = numpy.linspace(0.0, 1.0, self.reference_curve.points_count)
        c = color_indices[self._reference_indices]
        deep_signature.visualization.plot_multicolor_scatter(x=x, y=y, c=c, ax=ax, point_size=point_size, alpha=alpha, cmap=cmap, zorder=zorder)

    def plot_lined_curve(self, ax: matplotlib.axes.Axes, line_width: float = 2, alpha: float = 1, cmap: str = 'red', zorder: int = 1):
        x = self._points[:, 0]
        y = self._points[:, 1]
        deep_signature.visualization.plot_multicolor_line(x=x, y=y, ax=ax, line_width=line_width, alpha=alpha, cmap=cmap, zorder=zorder)


# =================================================
# PlanarCurvesGenerator Class
# =================================================
class PlanarCurvesManager:
    def __init__(self, curves_count: int):
        super().__init__()
        self._curves_count = curves_count
        self._planar_curves = []

    @property
    def planar_curves(self):
        return self._planar_curves

    # def plot_curves(self, curves_count: int, figsize: Tuple[int, int]):
    #     for i in range(curves_count):
    #         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    #         self._planar_curves[i].plot_curve(ax=ax)

    def get_random_planar_curve(self) -> PlanarCurve:
        index = numpy.random.randint(low=0, high=len(self._planar_curves))
        return self._planar_curves[index]

    # def _post_load(self):
    #     self._planar_curves = [PlanarCurve(points=item, closed=True) for item in self._items]
    #     for planar_curve in self._planar_curves:
    #         planar_curve.center_curve()
    #
    #     random.shuffle(self._planar_curves)
        # self._planar_curves = self._planar_curves[:self._curves_count]

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
# LevelCurvesGenerator Class
# =================================================
class LevelCurvesGenerator(ParallelProcessor):
    def __init__(
            self,
            curves_count: int,
            images_base_dir_path: str,
            sigmas: List[int],
            min_contour_level: float,
            max_contour_level: float,
            min_points_count: int,
            max_points_count: int,
            flat_point_threshold: float,
            max_flat_points_ratio: float,
            min_equiaffine_std: float,
            smoothing_iterations: int,
            smoothing_window_length: int,
            smoothing_poly_order: int):
        self._curves_count = curves_count
        self._images_base_dir_path = os.path.normpath(images_base_dir_path)
        self._sigmas = sigmas
        self._min_contour_level = min_contour_level
        self._max_contour_level = max_contour_level
        self._min_points_count = min_points_count
        self._max_points_count = max_points_count
        self._flat_point_threshold = flat_point_threshold
        self._max_flat_points_ratio = max_flat_points_ratio
        self._min_equiaffine_std = min_equiaffine_std
        self._smoothing_iterations = smoothing_iterations
        self._smoothing_window_length = smoothing_window_length
        self._smoothing_poly_order = smoothing_poly_order
        self._image_file_paths = self._get_image_file_paths()
        super().__init__()

    def _generate_items(self):
        image_file_path = self._image_file_paths[numpy.random.randint(low=0, high=len(self._image_file_paths))]
        sigma = self._sigmas[numpy.random.randint(low=0, high=len(self._sigmas))]
        contour_level = numpy.random.uniform(low=self._min_contour_level, high=self._max_contour_level)
        contours = self._try_extract_contours(image_file_path=image_file_path, sigma=sigma, contour_level=contour_level)
        return self._try_get_valid_contours(contours=contours)

    def _try_extract_contours(self, image_file_path: str, sigma: int, contour_level: float) -> Union[None, List[numpy.ndarray]]:
        try:
            image = skimage.io.imread(image_file_path)
        except:
            return None

        if len(image.shape) == 3 and image.shape[2] == 4:
            image = skimage.color.rgba2rgb(image)
        elif len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)

        gray_image = skimage.color.rgb2gray(image)
        gray_image_filtered = skimage.filters.gaussian(gray_image, sigma=sigma)
        contours = skimage.measure.find_contours(gray_image_filtered, contour_level)
        contours = [contour for contour in contours if self._min_points_count <= contour.shape[0] <= self._max_points_count]
        # contours.sort(key=lambda contour: contour.shape[0], reverse=True)
        return contours

    def _try_get_valid_contours(self, contours: List[numpy.ndarray]) -> List[numpy.ndarray]:
        valid_contours = []
        if contours is not None:
            for contour in contours:
                contour = contour[0:-1]
                planar_curve = PlanarCurve(points=contour)
                planar_curve.smooth_curve(iterations=self._smoothing_iterations, window_length=self._smoothing_window_length, poly_order=self._smoothing_poly_order)
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

        return valid_contours

    def _get_image_file_paths(self) -> List[str]:
        image_file_paths = []
        for sub_dir_path, _, file_names in os.walk(self._images_base_dir_path):
            if sub_dir_path == self._images_base_dir_path:
                continue

            for file_name in file_names:
                image_file_path = os.path.normpath(os.path.join(sub_dir_path, file_name))
                image_file_paths.append(image_file_path)

        return image_file_paths

    # def _zip_iterable(self) -> List:
    #     image_file_paths = []
    #     for sub_dir_path, _, file_names in os.walk(self._images_base_dir_path):
    #         if sub_dir_path == self._images_base_dir_path:
    #             continue
    #
    #         for file_name in file_names:
    #             image_file_path = os.path.normpath(os.path.join(sub_dir_path, file_name))
    #             image_file_paths.append(image_file_path)
    #
    #     image_files_count = len(image_file_paths)
    #     sigmas_count = len(self._sigmas)
    #     contour_levels_count = len(self._contour_levels)
    #
    #     max_curves_count = image_files_count * sigmas_count * contour_levels_count
    #     # if curves_count > max_curves_count:
    #     #     raise ValueError('curves_count exceeds maximum')
    #
    #     argument_packs = []
    #     argument_packs.append([*image_file_paths] * (sigmas_count * contour_levels_count))
    #     argument_packs.append([*self._sigmas] * (image_files_count * contour_levels_count))
    #     argument_packs.append([*self._contour_levels] * (image_files_count * sigmas_count))
    #
    #     entry_names = inspect.getfullargspec(LevelCurveManager._zip_iterable).args[4:]
    #     entry_names.insert(0, 'image_file_path')
    #     entry_names.insert(1, 'sigma')
    #     entry_names.insert(2, 'contour_level')
    #     iterable = [dict(zip(entry_names, values)) for values in zip(*argument_packs)]
    #
    #     return iterable


# =================================================
# PlanarCurvesManager Class
# =================================================
class PlanarCurvesManager:
    def __init__(self):
        super().__init__()
        self._planar_curves = []

    @property
    def planar_curves(self) -> List[PlanarCurve]:
        return self._planar_curves

    @property
    def planar_curves_count(self) -> int:
        return len(self._planar_curves)

    # def plot_curves(self, curves_count: int, figsize: Tuple[int, int]):
    #     for i in range(curves_count):
    #         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    #         self._planar_curves[i].plot_curve(ax=ax)

    def get_random_planar_curve(self) -> PlanarCurve:
        index = numpy.random.randint(low=0, high=len(self._planar_curves))
        return self._planar_curves[index]

    def load(self, curves_file_path: str):
        curves_points = numpy.load(file=os.path.normpath(path=curves_file_path), allow_pickle=True)
        self._planar_curves = [PlanarCurve(points=points, closed=True) for points in curves_points]
        for planar_curve in self._planar_curves:
            planar_curve.center_curve()
        random.shuffle(self._planar_curves)


