# python peripherals
import random
import os
import math
import multiprocessing
import itertools

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

class CurveSectionConfiguration:
    def __init__(self, center_point_index, radians, reflection):
        self._center_point_index = center_point_index
        self._radians = radians
        self._reflection = reflection

    @property
    def center_point_index(self):
        return self._center_point_index

    @property
    def radians(self):
        return self._radians

    @property
    def reflection(self):
        return self._reflection

    # def sample(self, curve, indices_pool, supporting_points_count):
    #     # start = self._center_point_index - supporting_points_count
    #     # stop = self._center_point_index + supporting_points_count + 1
    #
    #     right_supporting_points_indices = []
    #     i = self._center_point_index
    #     while True:
    #         if len(right_supporting_points_indices) == supporting_points_count + 1:
    #             break
    #         i_mod = numpy.mod(i, curve.shape[0])
    #         if i_mod in sampling_indices:
    #             right_supporting_points_indices.append(i_mod)
    #         i += 1
    #
    #     left_supporting_points_indices = []
    #     i = self._center_point_index - 1
    #     while True:
    #         if len(left_supporting_points_indices) == supporting_points_count:
    #             break
    #         i_mod = numpy.mod(i, curve.shape[0])
    #         if i_mod in sampling_indices:
    #             left_supporting_points_indices.append(i_mod)
    #         i -= 1
    #
    #     supporting_points_indices = numpy.sort(numpy.concatenate((left_supporting_points_indices, right_supporting_points_indices)))
    #
    #     # point_indices = numpy.arange(start, stop)
    #     # point_indices = numpy.mod(point_indices, curve.shape[0])
    #     # sampled_point_indices = point_indices[sampling_indices]
    #
    #     transformed_curve = curve_processing.translate_curve(curve=curve, offset=-curve[self._center_point_index])
    #     transformed_curve = curve_processing.transform_curve(curve=transformed_curve, radians=self._radians, reflection=self._reflection)
    #     return transformed_curve[supporting_points_indices], transformed_curve, supporting_points_indices


class CurveSection:
    def __init__(self, center_point_index, rotation_factor, apply_reflections=False):
        self._center_point_index = center_point_index
        self._rotation_factor = rotation_factor
        self._apply_reflections = apply_reflections
        self._curve_section_configurations = CurveSection._generate_curve_section_configurations(
            center_point_index=center_point_index,
            rotation_factor=rotation_factor,
            apply_reflections=apply_reflections)

    @property
    def curve_section_configurations(self):
        return self._curve_section_configurations

    def generate_pairs(self, pairs_count, shuffle=True):
        pairs = []

        configurations = self._curve_section_configurations
        if shuffle:
            random.shuffle(configurations)

        for i, pair in enumerate(itertools.combinations(configurations, 2)):
            if i == pairs_count:
                break
            pairs.append(pair)

        return pairs

    @staticmethod
    def _generate_curve_section_configurations(center_point_index, rotation_factor, apply_reflections):
        configurations = []

        reflection_types = ['none']
        if apply_reflections is True:
            reflection_types.extend(['horizontal', 'vertical'])

        for reflection in reflection_types:
            for rotation_index in range(rotation_factor):
                radians = rotation_index * ((2 * math.pi) / rotation_factor)
                configuration = CurveSectionConfiguration(
                    center_point_index=center_point_index,
                    radians=radians,
                    reflection=reflection)
                configurations.append(configuration)

        return configurations


class CurveDataGenerator:
    def __init__(self, curve, rotation_factor, sectioning_factor, sampling_factor, multimodality_factor, supporting_points_count, sampling_points_count, sampling_points_ratio=None):
        self._rotation_factor = rotation_factor
        self._sectioning_factor = sectioning_factor
        self._sampling_factor = sampling_factor
        self._multimodality_factor = multimodality_factor
        self._supporting_points_count = supporting_points_count
        self._sampling_points_ratio = sampling_points_ratio
        self._sampling_points_count = sampling_points_count
        if sampling_points_ratio is not None:
            self._sampling_points_count = int(curve.shape[0] * sampling_points_ratio)

        self._curve = curve
        self._evolved_curve = curve_processing.evolve_curve(
            curve=curve,
            evolution_iterations=2,
            evolution_dt=1e-12,
            smoothing_window_length=99,
            smoothing_poly_order=2,
            smoothing_iterations=6)

        self._curvature = curve_processing.calculate_curvature(curve)
        # self._curve_sections = CurveDataGenerator._generate_curve_sections(
        #     curve_points_count=self._sampling_points_count,
        #     rotation_factor=rotation_factor,
        #     sectioning_factor=sectioning_factor)

        self._curve_sections = CurveDataGenerator._generate_curve_sections(
            curve_points_count=self._curve.shape[0],
            rotation_factor=rotation_factor,
            sectioning_factor=sectioning_factor)

    # def save(self, dir_path):
    #     curve_dir_path = os.path.normpath(os.path.join(dir_path, str(self._curve_id)))
    #     os.mkdir(curve_dir_path)
    #     for curve_configuration in self._curve_configurations:
    #         curve_configuration.save(curve_dir_path)

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
    def curve_sections(self):
        return self._curve_sections

    def generate_negative_pairs(self):
        negative_pairs = []
        # bins = 2*self._supporting_points_count + 1
        bins = self._curve.shape[0]
        # count = self._rotation_factor * self._sectioning_factor * self._sampling_factor
        count = self._sectioning_factor * self._sampling_factor
        max_density = 1 / self._sampling_points_count
        dists = discrete_distribution.random_discrete_dist(bins=bins, multimodality=self._multimodality_factor, max_density=max_density, count=count)
        dist_index = 0
        for curve_section in self._curve_sections:
            for _ in range(self._sampling_factor):
                dist = dists[dist_index, :]
                for curve_section_configuration in curve_section.curve_section_configurations:
                    indices_pool = discrete_distribution.sample_discrete_dist(
                        dist=dist,
                        sampling_points_count=self._sampling_points_count)

                    # center_point_index = indices_pool[curve_section_configuration.center_point_index]
                    center_point_index = curve_section_configuration.center_point_index

                    supporting_points_indices = curve_sampling.sample_supporting_points_indices(
                        curve=self._curve,
                        center_point_index=center_point_index,
                        indices_pool=indices_pool,
                        supporting_points_count=self._supporting_points_count)

                    transformed_curve = curve_processing.translate_curve(
                        curve=self._curve,
                        offset=-self._curve[center_point_index])

                    transformed_curve = curve_processing.transform_curve(
                        curve=transformed_curve,
                        radians=curve_section_configuration.radians,
                        reflection=curve_section_configuration.reflection)

                    transformed_evolved_curve = curve_processing.translate_curve(
                        curve=self._evolved_curve,
                        offset=-self._evolved_curve[center_point_index])

                    transformed_evolved_curve = curve_processing.transform_curve(
                        curve=transformed_evolved_curve,
                        radians=curve_section_configuration.radians,
                        reflection=curve_section_configuration.reflection)

                    negative_pair = {
                        "curve": transformed_curve,
                        "evolved_curve": transformed_evolved_curve,
                        "supporting_points_indices": supporting_points_indices,
                        "indices_pool": indices_pool,
                        "dist": dist,
                        "center_point_index": center_point_index
                    }

                    negative_pairs.append(negative_pair)
                dist_index += 1
        return negative_pairs

    def generate_positive_pairs(self):
        positive_pairs = []
        bins = self._curve.shape[0]
        count = 2 * self._sectioning_factor * self._sampling_factor
        max_density = 1 / self._sampling_points_count
        dists = discrete_distribution.random_discrete_dist(bins=bins, multimodality=self._multimodality_factor, max_density=max_density, count=count)
        dist_index = 0
        for curve_section in self._curve_sections:
            for _ in range(self._sampling_factor):
                dist1 = dists[dist_index, :]
                dist2 = dists[dist_index+1, :]
                for curve_section_configuration in curve_section.curve_section_configurations:
                    indices_pool1 = discrete_distribution.sample_discrete_dist(
                        dist=dist1,
                        sampling_points_count=self._sampling_points_count)

                    indices_pool2 = discrete_distribution.sample_discrete_dist(
                        dist=dist2,
                        sampling_points_count=self._sampling_points_count)

                    # center_point_index = indices_pool[curve_section_configuration.center_point_index]
                    center_point_index = curve_section_configuration.center_point_index

                    supporting_points_indices1 = curve_sampling.sample_supporting_points_indices(
                        curve=self._curve,
                        center_point_index=center_point_index,
                        indices_pool=indices_pool1,
                        supporting_points_count=self._supporting_points_count)

                    supporting_points_indices2 = curve_sampling.sample_supporting_points_indices(
                        curve=self._curve,
                        center_point_index=center_point_index,
                        indices_pool=indices_pool2,
                        supporting_points_count=self._supporting_points_count)

                    transformed_curve = curve_processing.translate_curve(
                        curve=self._curve,
                        offset=-self._curve[center_point_index])

                    transformed_curve = curve_processing.transform_curve(
                        curve=transformed_curve,
                        radians=curve_section_configuration.radians,
                        reflection=curve_section_configuration.reflection)

                    positive_pair = {
                        "curve": transformed_curve,
                        "supporting_points_indices1": supporting_points_indices1,
                        "supporting_points_indices2": supporting_points_indices2,
                        "indices_pool1": indices_pool1,
                        "indices_pool2": indices_pool2,
                        "dist1": dist1,
                        "dist2": dist2,
                        "center_point_index": center_point_index
                    }

                    positive_pairs.append(positive_pair)
                dist_index += 2
        return positive_pairs


    @staticmethod
    def _generate_curve_sections(curve_points_count, rotation_factor, sectioning_factor):
        curve_sections = []
        indices = numpy.linspace(start=0, stop=curve_points_count, num=sectioning_factor, endpoint=False, dtype=int)
        # indices = numpy.random.randint(low=curve_points_count, size=sectioning_factor)
        for center_point_index in indices:
            curve_section = CurveSection(
                center_point_index=center_point_index,
                rotation_factor=rotation_factor,
                apply_reflections=False)

            curve_sections.append(curve_section)

        return curve_sections


class CurveDatasetGenerator:
    def __init__(self):
        self._curves = []
        self._negative_pairs = []
        self._positive_pairs = []

    @property
    def negative_pairs(self):
        return self._negative_pairs

    @property
    def positive_pairs(self):
        return self._positive_pairs

    # def save(self, dir_path, pairs_per_curve, rotation_factor, sampling_factor, sample_points, metadata_only=False, chunk_size=5):
    #
    #     if metadata_only is False:
    #         print('Saving dataset curves:')
    #
    #         def save_curve(curve):
    #             curve.save(dir_path=dir_path)
    #
    #         DatasetGenerator._process_curves(
    #             raw_curves=self._raw_curves,
    #             predicate=save_curve,
    #             rotation_factor=rotation_factor,
    #             sampling_factor=sampling_factor,
    #             sample_points=sample_points,
    #             limit=None,
    #             chunk_size=chunk_size)
    #
    #     print('Saving dataset metadata:')
    #
    #     DatasetGenerator._save_dataset_metadata(
    #         dir_path=dir_path,
    #         curves_count=len(self._raw_curves),
    #         pairs_per_curve=pairs_per_curve,
    #         rotation_factor=rotation_factor,
    #         sampling_factor=sampling_factor,
    #         sample_points=sample_points)

    def generate_curves(self, dir_path, plot_curves=False):
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
        for i, image_file_path in enumerate(image_file_paths):
            print('\r    - Processing images... {0:.1%} Done.'.format(i / image_files_count), end="")
            try:
                image = skimage.io.imread(image_file_path)
            except:
                continue

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

        print('    - Processing images... {0:.1%} Done.'.format(i / image_files_count), end="")
        self._curves = curves
        return curves

    def save_curves(self, dir_path):
        numpy.save(file=os.path.join(dir_path, 'curves.npy'), arr=self._curves)

    def load_curves(self, file_path, shuffle=True):
        self._curves = numpy.load(file=file_path, allow_pickle=True)
        if shuffle is True:
            random.shuffle(self._curves)
        return self._curves

    def generate_dataset(self, rotation_factor, sectioning_factor, sampling_factor, multimodality_factor, supporting_points_count, sampling_points_count, sampling_points_ratio=None, limit=5, chunk_size=5):
        print('Generating dataset curves:')

        positive_pairs = []
        negative_pairs = []

        def add_curve_pairs(negative_sample_pairs, positive_sample_pairs):
            negative_pairs.extend(negative_sample_pairs)
            positive_pairs.extend(positive_sample_pairs)

        curve_data_generators = []
        for curve in self._curves:
            curve_data_generator = CurveDataGenerator(
                curve=curve,
                rotation_factor=rotation_factor,
                sectioning_factor=sectioning_factor,
                sampling_factor=sampling_factor,
                multimodality_factor=multimodality_factor,
                supporting_points_count=supporting_points_count,
                sampling_points_count=sampling_points_count,
                sampling_points_ratio=sampling_points_ratio)
            curve_data_generators.append(curve_data_generator)

        CurveDatasetGenerator._process_curve_data_generator(
            curve_data_generators=curve_data_generators,
            predicate=add_curve_pairs,
            limit=limit,
            chunk_size=chunk_size)

        self._negative_pairs = negative_pairs
        self._positive_pairs = positive_pairs
        return negative_pairs, positive_pairs

    def save_dataset(self, dir_path):
        numpy.save(file=os.path.join(dir_path, 'negative_pairs.npy'), arr=self._negative_pairs)
        numpy.save(file=os.path.join(dir_path, 'positive_pairs.npy'), arr=self._positive_pairs)

    @staticmethod
    def _process_curve_data_generator(curve_data_generators, predicate, limit=None, chunk_size=5):
        if limit is not None:
            curve_data_generators = curve_data_generators[:limit]

        curve_data_generator_chunks = CurveDatasetGenerator._chunks(curve_data_generators, chunk_size)

        print('    - Creating pool...', end="")
        pool = multiprocessing.Pool()
        print('\r    - Creating pool... Done.')

        print('    - Processing curve data generators...', end="")
        for i, processed_curve_data_generator_chunk in enumerate(pool.imap_unordered(CurveDatasetGenerator._process_curve_data_generator_chunk, curve_data_generator_chunks, 1)):
            negative_sample_pairs = processed_curve_data_generator_chunk["negative_sample_pairs"]
            positive_sample_pairs = processed_curve_data_generator_chunk["positive_sample_pairs"]
            predicate(negative_sample_pairs, positive_sample_pairs)
            print('\r    - Processing curve data generators... {0:.1%} Done.'.format((i+1) / len(curve_data_generator_chunks)), end="")

        print('\r    - Processing curve data generators... {0:.1%} Done.'.format((i+1) / len(curve_data_generator_chunks)))

    @staticmethod
    def _process_curve_data_generator_chunk(curve_data_generator_chunk):
        negative_sample_pairs = []
        positive_sample_pairs = []
        for curve_data_generator in curve_data_generator_chunk:
            negative_pairs = curve_data_generator.generate_negative_pairs()
            positive_pairs = curve_data_generator.generate_positive_pairs()

            for negative_pair in negative_pairs:
                curve = negative_pair['curve']
                evolved_curve = negative_pair['evolved_curve']
                supporting_points_indices = negative_pair['supporting_points_indices']
                curve_section_sample = curve[supporting_points_indices]
                evolved_curve_section_sample = evolved_curve[supporting_points_indices]
                negative_sample_pairs.append([curve_section_sample, evolved_curve_section_sample])

            for positive_pair in positive_pairs:
                curve = positive_pair['curve']
                supporting_points_indices1 = positive_pair['supporting_points_indices1']
                supporting_points_indices2 = positive_pair['supporting_points_indices2']
                curve_section_sample1 = curve[supporting_points_indices1]
                curve_section_sample2 = curve[supporting_points_indices2]
                positive_sample_pairs.append([curve_section_sample1, curve_section_sample2])

        return {
            "negative_sample_pairs": negative_sample_pairs,
            "positive_sample_pairs": positive_sample_pairs
        }

    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    @staticmethod
    def _chunks(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

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
