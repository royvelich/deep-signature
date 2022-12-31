# python peripherals
from __future__ import annotations
from typing import Union, List, Callable
import os
import itertools
from pathlib import Path

# numpy
import numpy

# scipy
import scipy.signal

# sklearn
import sklearn.preprocessing

# pytorch
import torch
import torch.nn

# deep_signature
from deep_signature.core.discrete_distributions import DiscreteDistribution

# shapely
from shapely.geometry import Polygon

# matplotlib
import matplotlib
import matplotlib.axes
import matplotlib.figure

# deep_signature
import deep_signature.manifolds.planar_curves.visualization
from deep_signature.core.base import SeedableObject
from deep_signature.core import transformations


# =================================================
# PlanarCurve Class
# =================================================
class PlanarCurve(SeedableObject):
    def __init__(self, points: numpy.ndarray, closed: Union[None, bool] = None, reference_curve: Union[None, PlanarCurve] = None, reference_indices: Union[None, numpy.ndarray] = None):
        super().__init__()
        self._points = points
        self._indices = numpy.array(list(range(points.shape[0])))

        if closed is None:
            self._closed = self.is_closed()
        else:
            self._closed = closed

        if reference_curve is None:
            self._reference_curve = self
            self._reference_indices = self._indices
        else:
            self._reference_curve = reference_curve
            self._reference_indices = reference_indices

        self._discrete_distribution = None

    # -------------------------------------------------
    # properties
    # -------------------------------------------------

    @property
    def points(self) -> numpy.ndarray:
        return self._points.astype(numpy.float32)

    @property
    def indices(self) -> numpy.ndarray:
        return self._indices

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
        for item in itertools.combinations(indices, 3):
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
    # approximation
    # -------------------------------------------------
    def approximate_curve_signature(self, model: torch.nn.Module, supporting_points_count: int, device: torch.device) -> numpy.ndarray:
        local_signature = numpy.zeros([self.points_count, 2])
        curve_neighborhoods = self.extract_curve_neighborhoods(supporting_points_count=supporting_points_count)
        model = model.to(device=device)
        for i, curve_neighborhood in enumerate(curve_neighborhoods):
            curve_neighborhood.normalize_curve(force_ccw=False, force_endpoint=False)
            batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(curve_neighborhood.points), dim=0), dim=0).to(device)
            with torch.no_grad():
                x = model(batch_data)
                local_signature[i, :] = torch.squeeze(torch.squeeze(x, dim=0), dim=0).cpu().detach().numpy()

        return local_signature

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
        sampled_planar_curve = PlanarCurve(points=self._points[indices], closed=self._closed, reference_curve=self, reference_indices=indices)
        sampled_planar_curve._discrete_distribution = discrete_distribution
        return sampled_planar_curve

    def extract_curve_neighborhood_wrt_reference(self, center_point_index: int, supporting_points_count: int) -> PlanarCurve:
        reference_curve_indices = self._reference_indices.copy()
        reference_curve_indices = numpy.append(arr=reference_curve_indices, values=[center_point_index])
        reference_curve_indices = numpy.unique(ar=reference_curve_indices)
        return self._extract_curve_neighborhood(center_point_index=center_point_index, supporting_points_count=supporting_points_count, curve_indices=reference_curve_indices, add_reference_curve=False)
        # center_point_meta_index = numpy.where(reference_curve_indices == center_point_index)[0]
        # curve_neighborhood_meta_indices = numpy.arange(start=center_point_meta_index - supporting_points_count, stop=center_point_meta_index + supporting_points_count + 1)
        # curve_neighborhood_meta_indices = numpy.mod(curve_neighborhood_meta_indices, reference_curve_indices.shape[0])
        # curve_neighborhood_indices = reference_curve_indices[curve_neighborhood_meta_indices]
        # curve_neighborhood_points = self._reference_curve.points[curve_neighborhood_indices]
        # return PlanarCurve(points=curve_neighborhood_points, closed=False)

    def extract_curve_neighborhood(self, center_point_index: int, supporting_points_count: int) -> PlanarCurve:
        return self._extract_curve_neighborhood(center_point_index=center_point_index, supporting_points_count=supporting_points_count, curve_indices=self.indices, add_reference_curve=True)

    def extract_curve_neighborhoods(self, supporting_points_count: int) -> List[PlanarCurve]:
        curve_neighborhoods = []
        for index in self.indices:
            curve_neighborhoods.append(self.extract_curve_neighborhood(center_point_index=index, supporting_points_count=supporting_points_count))

        return curve_neighborhoods

    def _extract_curve_neighborhood(self, center_point_index: int, supporting_points_count: int, curve_indices: numpy.ndarray, add_reference_curve: bool) -> PlanarCurve:
        center_point_meta_index = numpy.where(curve_indices == center_point_index)[0]
        curve_neighborhood_meta_indices = numpy.arange(start=center_point_meta_index - supporting_points_count, stop=center_point_meta_index + supporting_points_count + 1)
        curve_neighborhood_meta_indices = numpy.mod(curve_neighborhood_meta_indices, curve_indices.shape[0])
        curve_neighborhood_indices = curve_indices[curve_neighborhood_meta_indices]
        curve_neighborhood_points = self._reference_curve.points[curve_neighborhood_indices]

        if add_reference_curve is True:
            reference_curve = self
            reference_indices = curve_neighborhood_indices
        else:
            reference_curve = None
            reference_indices = None

        return PlanarCurve(points=curve_neighborhood_points, closed=False, reference_curve=reference_curve, reference_indices=reference_indices)

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
        return self._rng.integers(low=0, high=self._points.shape[0])

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

    def is_within(self, planar_curve: PlanarCurve) -> bool:
        reference_polygon = Polygon(shell=self._points)
        query_polygon = Polygon(shell=planar_curve.points)
        return reference_polygon.within(query_polygon)

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
        deep_signature.manifolds.planar_curves.visualization.plot_multicolor_scatter(x=x, y=y, c=c, ax=ax, point_size=point_size, alpha=alpha, cmap=cmap, zorder=zorder)

    def plot_lined_curve(self, ax: matplotlib.axes.Axes, line_width: float = 2, alpha: float = 1, cmap: str = 'red', zorder: int = 1):
        x = self._points[:, 0]
        y = self._points[:, 1]
        deep_signature.manifolds.planar_curves.visualization.plot_multicolor_line(x=x, y=y, ax=ax, line_width=line_width, alpha=alpha, cmap=cmap, zorder=zorder)

    def plot_scattered_signature(self, model: torch.nn.Module, supporting_points_count: int, device: torch.device, ax: List[matplotlib.axes.Axes], point_size: float = 2, alpha: float = 1, cmap: str = 'red', zorder: int = 1):
        signature = self.approximate_curve_signature(model=model, supporting_points_count=supporting_points_count, device=device)
        x = numpy.array(list(range(signature.shape[0])))
        kappa = signature[:, 0]
        kappa_s = signature[:, 1]
        c = numpy.linspace(0.0, 1.0, signature.shape[0])
        line_style = ''
        deep_signature.manifolds.planar_curves.visualization.plot_line(x=x, y=kappa, ax=ax[0], line_style=line_style, alpha=alpha, zorder=zorder)
        deep_signature.manifolds.planar_curves.visualization.plot_line(x=x, y=kappa_s, ax=ax[1], line_style=line_style, alpha=alpha, zorder=zorder)
        deep_signature.manifolds.planar_curves.visualization.plot_multicolor_scatter(x=kappa, y=kappa_s, c=c, ax=ax[2], point_size=point_size, alpha=alpha, cmap=cmap, zorder=zorder)

    # def plot_curve_signature(self, model: torch.nn.Module, supporting_points_count: int, device: torch.device, point_size: float = 2, alpha: float = 1):
    #     fig, axes = matplotlib.pyplot.subplots(nrows=4, ncols=1, figsize=(40, 10))
    #     invariant = self.approximate_curve_signature(model=model, supporting_points_count=supporting_points_count, device=device)
    #     self.plot_scattered_curve(ax=axes[0], point_size=point_size, cmap='red')
    #     self.plot_scattered_curve(ax=axes[0], point_size=point_size, cmap='red')
    #     self.plot_scattered_curve(ax=axes[0], point_size=point_size, cmap='red')

        # discrete_distribution = discrete_distributions.MultimodalGaussianDiscreteDistribution(bins_count=planar_curve.points_count, multimodality=10)
        # # discrete_distribution = discrete_distributions.UniformDiscreteDistribution(bins_count=planar_curve.points_count)
        # discrete_distribution.plot_dist(ax=axes[0])
        # sampled_planar_curve = planar_curve.sample_curve(sampling_ratio=sampling_ratio, discrete_distribution=discrete_distribution)
        # sampled_planar_curve.plot_scattered_curve(ax=axes[1], cmap='hsv')
        # matplotlib.pyplot.show()


# =================================================
# PlanarCurvesManager Class
# =================================================
class PlanarCurvesManager(SeedableObject):
    def __init__(self, curves_file_path: Path):
        super().__init__()
        self._curves_file_path = curves_file_path
        self._planar_curves = self._load_curves()

    @property
    def planar_curves(self) -> List[PlanarCurve]:
        return self._planar_curves

    @property
    def planar_curves_count(self) -> int:
        return len(self._planar_curves)

    def get_random_planar_curve(self) -> PlanarCurve:
        index = self._rng.integers(low=0, high=len(self._planar_curves))
        return self._planar_curves[index]

    def _load_curves(self) -> List[PlanarCurve]:
        curves_points = numpy.load(file=os.path.normpath(path=self._curves_file_path), allow_pickle=True)
        planar_curves = [PlanarCurve(points=points, closed=True) for points in curves_points]
        for planar_curve in planar_curves:
            planar_curve.center_curve()
        return planar_curves

