# python peripherals
import os
import multiprocessing
import pathlib
import inspect

# numpy
import numpy

# skimage
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure

# deep_signature
from deep_signature.data_manipulation import curve_processing
from deep_signature.utils import utils


class CurvesGenerator:
    _label = "curves"

    @classmethod
    def generate_curves(cls, dir_path, curves_count, chunksize, **kwargs):
        curves = []
        iterable = cls._zip_iterable(curves_count=curves_count, **kwargs)

        def reduce_func(curve):
            if curve is not None:
                curves.append(curve)
                print(f'Curves Count: {len(curves)}')

        utils.par_proc(
            map_func=cls._map_func,
            reduce_func=reduce_func,
            iterable=iterable,
            chunksize=chunksize,
            label=cls._label
        )

        # curves = curves[:curves_count]
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        numpy.save(file=os.path.normpath(os.path.join(dir_path, f'curves.npy')), arr=numpy.array(curves, dtype=object))

    @staticmethod
    def load_curves(dir_path):
        return numpy.load(file=os.path.normpath(os.path.join(dir_path, f'curves.npy')), allow_pickle=True)

    @classmethod
    def _map_func(cls, kwargs):
        return cls._generate_curve(**kwargs)

    @staticmethod
    def _generate_curve():
        raise NotImplemented

    @staticmethod
    def _zip_iterable(curves_count):
        raise NotImplemented


class CirclesGenerator(CurvesGenerator):
    _label = "circles"

    @staticmethod
    def _generate_curve(min_radius, max_radius, sampling_density):
        radius = float(numpy.random.uniform(low=min_radius, high=max_radius, size=1))
        circumference = 2 * radius * numpy.pi
        points_count = int(numpy.round(sampling_density * circumference))
        radians_delta = 2 * numpy.pi / points_count
        pointer = numpy.array([radius, 0])
        circle = numpy.empty((points_count, 2))
        for i in range(points_count):
            circle[i] = curve_processing.rotate_curve(curve=pointer, radians=i * radians_delta)

        return circle

    @staticmethod
    def _zip_iterable(curves_count, **kwargs):
        return [kwargs] * curves_count


class LevelCurvesGenerator(CurvesGenerator):
    _label = "level curves"

    @staticmethod
    def _generate_curve(image_file_path, sigma, contour_level, min_points, max_points, flat_point_threshold, max_flat_points_ratio, max_abs_kappa):
        try:
            image = skimage.io.imread(image_file_path)
        except:
            return None

        gray_image = skimage.color.rgb2gray(image)
        gray_image_filtered = skimage.filters.gaussian(gray_image, sigma=sigma)
        contours = skimage.measure.find_contours(gray_image_filtered, contour_level)
        contours = [contour for contour in contours if min_points <= contour.shape[0] <= max_points]
        contours.sort(key=lambda contour: contour.shape[0], reverse=True)

        eps = 1e-12
        for contour in contours:
            first_point = contour[0]
            last_point = contour[-1]
            distance = numpy.linalg.norm(x=first_point - last_point, ord=2)
            if distance > eps:
                continue

            curve = curve_processing.smooth_curve(
                curve=contour,
                iterations=6,
                window_length=99,
                poly_order=2)

            kappa = curve_processing.calculate_euclidean_curvature(curve)
            min_kappa = numpy.abs(numpy.min(kappa))
            max_kappa = numpy.abs(numpy.max(kappa))
            flat_points = numpy.sum(numpy.array([1 if x < flat_point_threshold else 0 for x in numpy.abs(kappa)]))
            flat_points_ratio = flat_points / len(kappa)

            if min_kappa > max_abs_kappa:
                continue

            if max_kappa > max_abs_kappa:
                continue

            if flat_points_ratio > max_flat_points_ratio:
                continue

            kappa_equiaffine = curve_processing.calculate_equiaffine_curvature(curve)
            kappa_equiaffine_cleaned = kappa_equiaffine[~numpy.isnan(kappa_equiaffine)]
            equiaffine_std = numpy.std(kappa_equiaffine_cleaned)
            if equiaffine_std < 0.05:
                continue

            return curve
        return None

    @staticmethod
    def _zip_iterable(curves_count, images_base_dir_path, sigmas, contour_levels, min_points, max_points, flat_point_threshold, max_flat_points_ratio, max_abs_kappa):
        image_file_paths = []
        images_base_dir_path = os.path.normpath(images_base_dir_path)
        for sub_dir_path, _, file_names in os.walk(images_base_dir_path):
            if sub_dir_path == images_base_dir_path:
                continue

            for file_name in file_names:
                image_file_path = os.path.normpath(os.path.join(sub_dir_path, file_name))
                image_file_paths.append(image_file_path)

        image_files_count = len(image_file_paths)
        sigmas_count = len(sigmas)
        contour_levels_count = len(contour_levels)

        max_curves_count = image_files_count * sigmas_count * contour_levels_count
        if curves_count > max_curves_count:
            raise ValueError('curves_count exceeds maximum')

        argument_packs = []
        argument_packs.append([*image_file_paths] * (sigmas_count * contour_levels_count))
        argument_packs.append([*sigmas] * (image_files_count * contour_levels_count))
        argument_packs.append([*contour_levels] * (image_files_count * sigmas_count))
        argument_packs.append([min_points] * max_curves_count)
        argument_packs.append([max_points] * max_curves_count)
        argument_packs.append([flat_point_threshold] * max_curves_count)
        argument_packs.append([max_flat_points_ratio] * max_curves_count)
        argument_packs.append([max_abs_kappa] * max_curves_count)

        entry_names = inspect.getfullargspec(LevelCurvesGenerator._zip_iterable).args[4:]
        entry_names.insert(0, 'image_file_path')
        entry_names.insert(1, 'sigma')
        entry_names.insert(2, 'contour_level')
        iterable = [dict(zip(entry_names, values)) for values in zip(*argument_packs)]

        return iterable
