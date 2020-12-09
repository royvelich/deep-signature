# python peripherals
import random
import os
import math
import multiprocessing
import itertools
import pathlib
import inspect

# numpy
import numpy

# skimage
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure

# matplotlib
import matplotlib.pyplot as plt

# deep_signature
from deep_signature import curve_processing
from deep_signature import curve_sampling
from deep_signature import discrete_distribution
from deep_signature import utils


class CurveSectionSampleManager:
    def __init__(self, curve_section, dist, sampling_points_count, supporting_points_count, center_point_index):
        self._curve_section = curve_section
        self._dist = dist
        self._sampling_points_count = sampling_points_count
        self._center_point_index = center_point_index
        self._supporting_points_count = supporting_points_count
        self._curve_section_sample = None
        self._supporting_curve_section_sample = None
        self._indices_pool = None
        self._supporting_point_indices = None

    @property
    def curve_section(self):
        return self._curve_section

    @property
    def dist(self):
        return self._dist

    @property
    def supporting_points_count(self):
        return self._supporting_points_count

    @property
    def center_point_index(self):
        return self._center_point_index

    @property
    def curve_section_sample(self):
        return self._curve_section[self._indices_pool]

    @property
    def curve_section_sample_normalized(self):
        return curve_processing.translate_curve(
            curve=self.curve_section_sample,
            offset=-self._curve_section[self._center_point_index])

    @property
    def supporting_curve_section_sample(self):
        return self._curve_section[self._supporting_point_indices]

    @property
    def supporting_curve_section_sample_normalized(self):
        return curve_processing.translate_curve(
            curve=self.supporting_curve_section_sample,
            offset=-self._curve_section[self._center_point_index])

    @property
    def indices_pool(self):
        return self._indices_pool

    @property
    def supporting_point_indices(self):
        return self._supporting_point_indices

    def sample_curve_section(self):
        self._indices_pool = discrete_distribution.sample_discrete_dist(
            dist=self._dist,
            sampling_points_count=self._sampling_points_count)

        self._supporting_point_indices = curve_sampling.sample_supporting_point_indices(
            curve=self._curve_section,
            center_point_index=self._center_point_index,
            indices_pool=self._indices_pool,
            supporting_point_count=self._supporting_points_count)


class CurveSectionManager:
    def __init__(self, curve_section, sampling_points_count, supporting_points_count, center_point_index, center_point_curvature):
        self._curve_section = curve_section
        self._center_point_index = center_point_index
        self._center_point_curvature = center_point_curvature
        self._supporting_points_count = supporting_points_count
        self._sampling_points_count = sampling_points_count
        self._curve_section_sample_managers = []

    @property
    def curve_section(self):
        return self._curve_section

    @property
    def supporting_points_count(self):
        return self._supporting_points_count

    @property
    def center_point_index(self):
        return self._center_point_index

    @property
    def center_point_curvature(self):
        return self._center_point_curvature

    @property
    def curve_section_sample_managers(self):
        return self._curve_section_sample_managers

    def generate_curve_section_sample_managers(self, dists):
        for dist in dists:
            curve_section_sample_manager = CurveSectionSampleManager(
                curve_section=self._curve_section,
                dist=dist,
                sampling_points_count=self._sampling_points_count,
                supporting_points_count=self._supporting_points_count,
                center_point_index=self._center_point_index
            )

            curve_section_sample_manager.sample_curve_section()

            self._curve_section_sample_managers.append(curve_section_sample_manager)


class CurveManager:
    def __init__(
            self,
            curve,
            rotation_factor,
            sampling_factor,
            multimodality_factor,
            supporting_points_count,
            sampling_points_count,
            sectioning_points_count,
            section_points_count,
            evolution_iterations,
            evolution_dt,
            sampling_points_ratio=None,
            sectioning_points_ratio=None):

        self._rotation_factor = rotation_factor
        self._sampling_factor = sampling_factor
        self._multimodality_factor = multimodality_factor
        self._supporting_points_count = supporting_points_count
        self._sampling_points_ratio = sampling_points_ratio
        self._sampling_points_count = sampling_points_count
        self._sectioning_points_ratio = sectioning_points_ratio
        self._sectioning_points_count = sectioning_points_count
        self._section_points_count = section_points_count
        self._evolution_iterations = evolution_iterations
        self._evolution_dt = evolution_dt
        self._max_density = 1 / self._sampling_points_count

        self._curve = curve_processing.normalize_curve(curve)
        self._curve_points_count = self._curve.shape[0]

        # if sampling_points_ratio is not None:
        #     self._sampling_points_count = int(self._curve.shape[0] * sampling_points_ratio)

        if sectioning_points_ratio is not None:
            self._sectioning_points_count = int(self._curve.shape[0] * sectioning_points_ratio)

        self._curvature = curve_processing.calculate_curvature(self._curve)
        self._curve_section_managers = []
        self._negative_pairs = []
        self._positive_pairs = []

    @property
    def curve(self):
        return self._curve

    @property
    def curvature(self):
        return self._curvature

    @property
    def evolved_curve(self):
        return self._evolved_curve

    @property
    def curve_section_managers(self):
        return self._curve_section_managers

    @property
    def negative_pairs(self):
        return self._negative_pairs

    @property
    def positive_pairs(self):
        return self._positive_pairs

    def generate_negative_pairs(self, min_curvature_diff, max_curvature_diff, min_norm_diff, min_curvature):
        dists_count = self._sectioning_points_count * self._sampling_factor
        bins = 2*self._section_points_count + 1

        dists = discrete_distribution.random_discrete_dist(
            bins=bins,
            multimodality=self._multimodality_factor,
            max_density=self._max_density,
            count=dists_count)

        center_point_indices = numpy.linspace(
            start=0,
            stop=self._curve_points_count,
            num=self._sectioning_points_count,
            endpoint=False,
            dtype=int)

        self._negative_pairs = []
        for i, center_point_index in enumerate(center_point_indices):
            start_dist_index = i * self._sampling_factor
            end_dist_index = (i + 1) * self._sampling_factor
            current_dists = dists[start_dist_index:end_dist_index]
            center_point_curvature = self._curvature[center_point_index]
            if center_point_curvature > min_curvature:
                for dist in current_dists:
                    rolling_indices = numpy.arange(self._curve_points_count)
                    rolling_indices = numpy.roll(rolling_indices, -(center_point_index + 1))
                    rolling_indices = numpy.delete(rolling_indices, self._curve_points_count - 1)
                    for rolling_point_index in rolling_indices:
                        rolling_point_curvature = self._curvature[rolling_point_index]
                        curvature_diff = numpy.abs(center_point_curvature - rolling_point_curvature)
                        if (curvature_diff > min_curvature_diff) and (curvature_diff < max_curvature_diff) and (numpy.sign(center_point_curvature) == numpy.sign(rolling_point_curvature)):
                            curve1_section_sample = self._generate_curve_section_sample(
                                center_point_index=center_point_index,
                                dist=dist
                            )

                            curve2_section_sample = self._generate_curve_section_sample(
                                center_point_index=rolling_point_index,
                                dist=dist
                            )

                            curve1_section_sample_matched, curve2_section_sample_matched = curve_processing.match_curve_sample_tangents(
                                curve_sample1=curve1_section_sample,
                                curve_sample2=curve2_section_sample,
                                index1=2,
                                index2=3
                            )

                            # self._negative_pairs.append([curve1_section_sample_matched, curve2_section_sample_matched])
                            # break

                            section_sample_diff = numpy.linalg.norm(curve1_section_sample_matched - curve2_section_sample_matched)
                            if section_sample_diff < min_norm_diff:
                                self._negative_pairs.append([curve1_section_sample_matched, curve2_section_sample_matched])
                                break

    def generate_positive_pairs(self):
        dists_count = self._sectioning_points_count * self._sampling_factor
        bins = 2*self._section_points_count + 1

        dists = discrete_distribution.random_discrete_dist(
            bins=bins,
            multimodality=self._multimodality_factor,
            max_density=self._max_density,
            count=dists_count)

        center_point_indices = numpy.linspace(
            start=0,
            stop=self._curve_points_count,
            num=self._sectioning_points_count,
            endpoint=False,
            dtype=int)

        self._positive_pairs = []
        for i, center_point_index in enumerate(center_point_indices):
            start_dist_index = i * self._sampling_factor
            end_dist_index = (i + 1) * self._sampling_factor
            current_dists = dists[start_dist_index:end_dist_index]
            for _ in range(self._rotation_factor):
                for dist1, dist2 in zip(current_dists[::2], current_dists[1::2]):
                    curve1_section_sample = self._generate_curve_section_sample(
                        center_point_index=center_point_index,
                        dist=dist1
                    )

                    curve2_section_sample = self._generate_curve_section_sample(
                        center_point_index=center_point_index,
                        dist=dist2
                    )

                    radians = numpy.random.uniform(0, 2*numpy.pi, 2)
                    curve1_section_sample = curve_processing.rotate_curve(curve=curve1_section_sample, radians=radians[0])
                    curve2_section_sample = curve_processing.rotate_curve(curve=curve2_section_sample, radians=radians[1])

                    self._positive_pairs.append([curve1_section_sample, curve2_section_sample])

    def _generate_curve_section_sample(self, center_point_index, dist):
        curve_start_index = center_point_index - self._section_points_count
        curve_end_index = center_point_index + self._section_points_count + 1
        curve_section_indices = numpy.mod(numpy.arange(curve_start_index, curve_end_index), self._curve_points_count)
        curve_section = self._curve[curve_section_indices]
        curve_section_sample_manager = CurveSectionSampleManager(
            curve_section=curve_section,
            dist=dist,
            sampling_points_count=self._sampling_points_count,
            supporting_points_count=self._supporting_points_count,
            center_point_index=self._section_points_count,
        )
        curve_section_sample_manager.sample_curve_section()
        return curve_section_sample_manager.supporting_curve_section_sample_normalized

    def generate_curve_sections(self):
        max_density = 1 / self._sampling_points_count
        dists_count = self._sectioning_points_count * self._sampling_factor
        bins = 2*self._section_points_count + 1

        dists = discrete_distribution.random_discrete_dist(
            bins=bins,
            multimodality=self._multimodality_factor,
            max_density=max_density,
            count=dists_count)

        center_point_indices = numpy.linspace(
            start=0,
            stop=self._curve_points_count,
            num=self._sectioning_points_count,
            endpoint=False,
            dtype=int)

        for i, center_point_index in enumerate(center_point_indices):
            start_dist_index = i * self._sampling_factor
            end_dist_index = (i + 1) * self._sampling_factor

            start_curve_index = center_point_index - self._section_points_count
            end_curve_index = center_point_index + self._section_points_count + 1

            curve_indices = numpy.mod(numpy.arange(start_curve_index, end_curve_index), self._curve_points_count)
            curve_section = self._curve[curve_indices]
            curve_section_dists = dists[start_dist_index:end_dist_index]

            curve_section_manager = CurveSectionManager(
                curve_section=curve_section,
                supporting_points_count=self._supporting_points_count,
                sampling_points_count=self._sampling_points_count,
                center_point_index=self._section_points_count,
                center_point_curvature=self._curvature[center_point_index]
            )

            curve_section_manager.generate_curve_section_sample_managers(curve_section_dists)

            self._curve_section_managers.append(curve_section_manager)


class CurveDatasetGenerator:
    def __init__(self):
        self._curves = []
        self._curves_count = 0
        self._negative_pairs = []
        self._positive_pairs = []
        self._curve_managers = []

    @property
    def negative_pairs(self):
        return self._negative_pairs

    @property
    def positive_pairs(self):
        return self._positive_pairs

    @property
    def curve_managers(self):
        return self._curve_managers

    def generate_curves(self, dir_path, chunk_size, plot_curves=False):
        curves = []

        def extend_curves(curve_chunk):
            curves.extend(curve_chunk)

        print('Generating curves:')
        curves = []
        image_file_paths = []
        base_dir_path = os.path.normpath(dir_path)
        for sub_dir_path, _, file_names in os.walk(base_dir_path):

            if sub_dir_path == base_dir_path:
                continue

            for file_name in file_names:
                image_file_path = os.path.normpath(os.path.join(sub_dir_path, file_name))
                image_file_paths.append(image_file_path)

        image_files_count = len(image_file_paths)
        print(f'    - {image_files_count} images detected.')

        # image_file_path_chunks = utils.chunks(image_file_paths, chunk_size)

        print('    - Creating pool...', end="")
        pool = multiprocessing.Pool()
        print('\r    - Creating pool... Done.')

        print('    - Processing images...', end="")
        for i, curve_chunk in enumerate(pool.imap_unordered(
                func=CurveDatasetGenerator._process_image_file_path,
                iterable=image_file_paths,
                chunksize=chunk_size)):

            extend_curves(curve_chunk)
            print('\r    - Processing images... {0:.1%} Done.'.format((i+1) / image_files_count), end="")

        print('\r    - Processing images... {0:.1%} Done.'.format((i+1) / image_files_count), end="")
        self._curves = curves
        return curves

    def save_curves(self, dir_path, curve_per_file):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        curve_chunks = utils.chunks(input_list=self._curves, chunks_count=curve_per_file)
        for i, curve_chunk in enumerate(curve_chunks):
            numpy.save(file=os.path.normpath(os.path.join(dir_path, f'curves{i}.npy')), arr=numpy.array(curve_chunk, dtype=object))

    def generate_curve_managers(
            self,
            rotation_factor,
            sampling_factor,
            multimodality_factor,
            supporting_points_count,
            sampling_points_count,
            sectioning_points_count,
            section_points_count,
            evolution_iterations,
            evolution_dt,
            curves_dir_path,
            curve_managers_dir_path,
            curve_managers_per_file,
            sampling_points_ratio=None,
            sectioning_points_ratio=None,
            limit=None,
            chunk_size=5):
        print('Generating dataset curves:')

        args = inspect.getfullargspec(CurveDatasetGenerator.generate_curve_managers).args
        args.remove('self')
        args.remove('chunk_size')
        args.remove('limit')
        args.remove('curves_dir_path')
        args.remove('curve_managers_dir_path')
        args.remove('curve_managers_per_file')
        args_dict = {k: v for k, v in locals().items() if k in args}
        processed_curves_count = 0
        curve_managers = []

        curves_count = CurveDatasetGenerator._count_curves(dir_path=curves_dir_path, limit=limit)
        curves = utils.create_data_generator(dir_path=curves_dir_path, limit=limit)

        print('    - Creating pool...', end="")
        pool = multiprocessing.Pool()
        print('\r    - Creating pool... Done.')

        print('    - Processing curve managers...', end="")
        for curve_chunk in curves:
            args_dicts = [args_dict] * len(curve_chunk)
            zipped_curves = list(zip(curve_chunk, args_dicts))
            for i, curve_manager in enumerate(pool.imap_unordered(CurveDatasetGenerator._process_curve_sections, zipped_curves, chunk_size)):
                processed_curves_count += 1
                curve_managers.append(curve_manager)
                if len(curve_managers) == curve_managers_per_file:
                    curve_managers_index = int(processed_curves_count / curve_managers_per_file) - 1
                    numpy.save(file=os.path.normpath(os.path.join(curve_managers_dir_path, f'curve_managers{curve_managers_index}.npy')), arr=numpy.array(curve_managers, dtype=object))
                    curve_managers = []
                print('\r    - Processing curve managers... {0:.1%} Done.'.format(processed_curves_count / curves_count), end="")

        print('\r    - Processing curve managers... {0:.1%} Done.'.format(processed_curves_count / curves_count))

    def generate_negative_pairs(
            self,
            rotation_factor,
            sampling_factor,
            multimodality_factor,
            supporting_points_count,
            sampling_points_count,
            sectioning_points_count,
            section_points_count,
            evolution_iterations,
            evolution_dt,
            curves_dir_path,
            curve_managers_dir_path,
            curve_managers_per_file,
            min_curvature,
            min_curvature_diff,
            max_curvature_diff,
            min_norm_diff,
            sampling_points_ratio=None,
            sectioning_points_ratio=None,
            limit=None,
            chunk_size=5):
        args = inspect.getfullargspec(CurveDatasetGenerator.generate_negative_pairs).args
        args_dict = {k: v for k, v in locals().items() if k in args}
        processed_curves_count = 0
        self._negative_pairs = []

        curves_count = CurveDatasetGenerator._count_curves(dir_path=curves_dir_path, limit=limit)
        curves = utils.create_data_generator(dir_path=curves_dir_path, limit=limit)

        print('    - Creating pool...', end="")
        pool = multiprocessing.Pool()
        print('\r    - Creating pool... Done.')

        print('    - Generating negative pairs...', end="")
        for curve_chunk in curves:
            args_dicts = [args_dict] * len(curve_chunk)
            zipped_curves = list(zip(curve_chunk, args_dicts))
            for i, negative_pairs in enumerate(pool.imap_unordered(CurveDatasetGenerator._process_curve_negative_pairs, zipped_curves, chunk_size)):
                processed_curves_count += 1
                self._negative_pairs.extend(negative_pairs)
                # if len(curve_managers) == curve_managers_per_file:
                #     curve_managers_index = int(processed_curves_count / curve_managers_per_file) - 1
                #     numpy.save(file=os.path.normpath(os.path.join(curve_managers_dir_path, f'curve_managers{curve_managers_index}.npy')), arr=numpy.array(curve_managers, dtype=object))
                #     curve_managers = []
                print('\r    - Generating negative pairs... {0:.1%} Done.'.format(processed_curves_count / curves_count), end="")

        print('\r    - Generating negative pairs... {0:.1%} Done.'.format(processed_curves_count / curves_count))

    def generate_positive_pairs(
            self,
            rotation_factor,
            sampling_factor,
            multimodality_factor,
            supporting_points_count,
            sampling_points_count,
            sectioning_points_count,
            section_points_count,
            evolution_iterations,
            evolution_dt,
            curves_dir_path,
            curve_managers_dir_path,
            curve_managers_per_file,
            min_curvature,
            min_curvature_diff,
            max_curvature_diff,
            min_norm_diff,
            sampling_points_ratio=None,
            sectioning_points_ratio=None,
            limit=None,
            chunk_size=5):
        args = inspect.getfullargspec(CurveDatasetGenerator.generate_positive_pairs).args
        args_dict = {k: v for k, v in locals().items() if k in args}
        processed_curves_count = 0
        self._positive_pairs = []

        curves_count = CurveDatasetGenerator._count_curves(dir_path=curves_dir_path, limit=limit)
        curves = utils.create_data_generator(dir_path=curves_dir_path, limit=limit)

        print('    - Creating pool...', end="")
        pool = multiprocessing.Pool()
        print('\r    - Creating pool... Done.')

        print('    - Generating positive pairs...', end="")
        for curve_chunk in curves:
            args_dicts = [args_dict] * len(curve_chunk)
            zipped_curves = list(zip(curve_chunk, args_dicts))
            for i, positive_pairs in enumerate(pool.imap_unordered(CurveDatasetGenerator._process_curve_positive_pairs, zipped_curves, chunk_size)):
                processed_curves_count += 1
                self._positive_pairs.extend(positive_pairs)
                print('\r    - Generating positive pairs... {0:.1%} Done.'.format(processed_curves_count / curves_count), end="")

        print('\r    - Generating positive pairs... {0:.1%} Done.'.format(processed_curves_count / curves_count))

    def save_negative_pairs(self, dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        numpy.save(file=os.path.normpath(os.path.join(dir_path, f'negative_pairs.npy')), arr=numpy.array(self._negative_pairs, dtype=object))

    def load_negative_pairs(self, dir_path):
        self._negative_pairs = numpy.load(file=os.path.normpath(os.path.join(dir_path, f'negative_pairs.npy')), allow_pickle=True)

    def save_positive_pairs(self, dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        numpy.save(file=os.path.normpath(os.path.join(dir_path, f'positive_pairs.npy')), arr=numpy.array(self._positive_pairs, dtype=object))

    def load_positive_pairs(self, dir_path):
        self._positive_pairs = numpy.load(file=os.path.normpath(os.path.join(dir_path, f'positive_pairs.npy')), allow_pickle=True)

    # @staticmethod
    # def _save_curve_managers(self, dir_path, curve_managers_per_file):
    #     pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    #     curve_manager_chunks = utils.chunks(input_list=self._curve_managers, chunks_count=curve_managers_per_file)
    #     for i, curve_manager_chunk in enumerate(curve_manager_chunks):
    #         numpy.save(file=os.path.normpath(os.path.join(dir_path, f'curve_managers{i}.npy')), arr=numpy.array(curve_manager_chunk, dtype=object))

    @staticmethod
    def _process_curve_sections(zipped_curve):
        curve = zipped_curve[0]
        args = zipped_curve[1]
        args['curve'] = curve
        curve_manager = CurveManager(**args)
        curve_manager.generate_curve_sections()
        return curve_manager

    @staticmethod
    def _process_curve_negative_pairs(zipped_curve):
        curve = zipped_curve[0]
        args = zipped_curve[1].copy()
        min_curvature_diff = args['min_curvature_diff']
        max_curvature_diff = args['max_curvature_diff']
        min_curvature = args['min_curvature']
        min_norm_diff = args['min_norm_diff']
        del args['self']
        del args['chunk_size']
        del args['limit']
        del args['curves_dir_path']
        del args['curve_managers_dir_path']
        del args['curve_managers_per_file']
        del args['min_curvature_diff']
        del args['max_curvature_diff']
        del args['min_curvature']
        del args['min_norm_diff']
        args['curve'] = curve
        curve_manager = CurveManager(**args)
        curve_manager.generate_negative_pairs(
            min_curvature_diff=min_curvature_diff,
            max_curvature_diff=max_curvature_diff,
            min_norm_diff=min_norm_diff,
            min_curvature=min_curvature
        )
        return curve_manager.negative_pairs

    @staticmethod
    def _process_curve_positive_pairs(zipped_curve):
        curve = zipped_curve[0]
        args = zipped_curve[1].copy()
        del args['self']
        del args['chunk_size']
        del args['limit']
        del args['curves_dir_path']
        del args['curve_managers_dir_path']
        del args['curve_managers_per_file']
        del args['min_curvature_diff']
        del args['max_curvature_diff']
        del args['min_curvature']
        del args['min_norm_diff']
        args['curve'] = curve
        curve_manager = CurveManager(**args)
        curve_manager.generate_positive_pairs()
        return curve_manager.positive_pairs

    @staticmethod
    def _process_image_file_path(image_file_path, plot_curves=False):
        curves = []

        try:
            image = skimage.io.imread(image_file_path)
        except:
            return curves

        sigmas = [2, 4, 8, 16, 32]
        contour_levels = [0.2, 0.5, 0.8]
        for sigma in sigmas:
            for contour_level in contour_levels:
                curve, kappa = CurveDatasetGenerator._extract_curve_from_image(
                    image=image,
                    sigma=sigma,
                    contour_level=contour_level,
                    min_points=1000,
                    max_points=6000,
                    flat_point_threshold=1e-3,
                    max_flat_points_ratio=0.04,
                    max_abs_kappa=8)

                if curve is not None:
                    curves.append(curve)
                    if plot_curves is True:
                        fig, ax = plt.subplots(2, 1)
                        ax[0].plot(curve[:, 0], curve[:, 1])
                        ax[1].plot(range(len(kappa)), kappa)
                        plt.show()

        return curves

    @staticmethod
    def _extract_curve_from_image(image, sigma, contour_level, min_points, max_points, flat_point_threshold, max_flat_points_ratio, max_abs_kappa):
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

            kappa = curve_processing.calculate_curvature(curve)
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

            # kappa_std = numpy.sqrt(numpy.var(kappa))
            # kappa_mean = numpy.abs(numpy.mean(kappa))

            # if kappa_mean < min_kappa_mean:
            #     continue
            #
            # if kappa_std < min_kappa_std:
            #     continue

            return curve, kappa
        return None, None

    @staticmethod
    def _count_curves(dir_path, limit):
        curves_count = 0
        curves = utils.create_data_generator(dir_path=dir_path, limit=limit)
        for curve_chunk in curves:
            for _ in curve_chunk:
                curves_count += 1

        return curves_count
