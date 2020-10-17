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

    def sample(self, curve, sampling_indices, supporting_points_count):
        start = self._center_point_index - supporting_points_count
        stop = self._center_point_index + supporting_points_count + 1

        point_indices = numpy.arange(start, stop)
        point_indices = numpy.mod(point_indices, curve.shape[0])
        sampled_point_indices = point_indices[sampling_indices]

        transformed_curve = curve_processing.translate_curve(curve=curve, offset=-curve[self._center_point_index])
        transformed_curve = curve_processing.transform_curve(curve=transformed_curve, radians=self._radians, reflection=self._reflection)
        return transformed_curve[sampled_point_indices], transformed_curve


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
    def __init__(self, curve, rotation_factor, sectioning_factor, sampling_factor, multimodality_factor, sampling_points_count, supporting_points_count):
        self._rotation_factor = rotation_factor
        self._sectioning_factor = sectioning_factor
        self._sampling_factor = sampling_factor
        self._multimodality_factor = multimodality_factor
        self._sampling_points_count = sampling_points_count
        self._supporting_points_count = supporting_points_count
        self._curve = curve
        self._evolved_curve = curve_processing.evolve_curve(
            curve=curve,
            evolution_iterations=3,
            evolution_dt=1e-4,
            smoothing_window_length=99,
            smoothing_poly_order=2,
            smoothing_iterations=6)

        self._curvature = curve_processing.calculate_curvature(curve)
        self._curve_sections = CurveDataGenerator._generate_curve_sections(
            curve_points_count=curve.shape[0],
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
        bins = 2*self._supporting_points_count + 1
        count = self._rotation_factor * self._sectioning_factor * self._sampling_factor
        dists = discrete_distribution.random_discrete_dist(bins=bins, multimodality=self._multimodality_factor, count=count)
        dist_index = 0
        for curve_section in self._curve_sections:
            for curve_section_configuration in curve_section.curve_section_configurations:
                for _ in range(self._sampling_factor):
                    dist = dists[dist_index, :]
                    sampling_indices = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=self._sampling_points_count)

                    curve_section_sample, transformed_curve = \
                        curve_section_configuration.sample(curve=self._curve, sampling_indices=sampling_indices, supporting_points_count=self._supporting_points_count)

                    evolved_curve_section_sample, transformed_evolved_curve = \
                        curve_section_configuration.sample(curve=self._evolved_curve, sampling_indices=sampling_indices, supporting_points_count=self._supporting_points_count)

                    negative_pair = {
                        "transformed_curve": transformed_curve,
                        "transformed_evolved_curve": transformed_evolved_curve,
                        "curve_section_sample": curve_section_sample,
                        "evolved_curve_section_sample": evolved_curve_section_sample,
                        "dist": dist,
                        "sampling_indices": sampling_indices
                    }

                    negative_pairs.append(negative_pair)
                    dist_index = dist_index + 1

        return negative_pairs


    @staticmethod
    def _generate_curve_sections(curve_points_count, rotation_factor, sectioning_factor):
        curve_sections = []
        indices = numpy.random.randint(low=curve_points_count, size=sectioning_factor)
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

    def load_curves(self, file_path):
        self._curves = numpy.load(file=file_path, allow_pickle=True)
        return self._curves

    # def generate_curve_dataset(self, rotation_factor, sectioning_factor, sampling_factor, multimodal_factor, sampling_points_count, supporting_points_count, limit=5, chunk_size=5):
    #     print('Generating dataset curves:')
    #
    #     curves = []
    #
    #     def add_curve(curve):
    #         curves.append(curve)
    #
    #     DatasetGenerator._process_curves(
    #         curves=self._curves,
    #         predicate=add_curve,
    #         rotation_factor=rotation_factor,
    #         sectioning_factor=sectioning_factor,
    #         sampling_factor=sampling_factor,
    #         multimodal_factor=multimodal_factor,
    #         sampling_points_count=sampling_points_count,
    #         supporting_points_count=supporting_points_count,
    #         limit=limit,
    #         chunk_size=chunk_size)
    #
    #     return curves

    # @staticmethod
    # def _generate_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor, positive):
    #     pairs = numpy.empty(shape=[curves_count * pairs_per_curve, 7]).astype(int)
    #     pairs_keys = {}
    #     for curve_index in range(curves_count):
    #         for curve_pair_index in range(pairs_per_curve):
    #             pair_index = pairs_per_curve * curve_index + curve_pair_index
    #             print(f'\r        - Creating pair #{pair_index}', end="")
    #             while True:
    #                 if positive is True:
    #                     factors = numpy.array([rotation_factor - 1, sampling_factor - 1])
    #                     curve1_indices = numpy.concatenate((numpy.array([curve_index]), numpy.round(numpy.random.rand(2) * factors))).astype(int)
    #                     curve2_indices = numpy.concatenate((numpy.array([curve_index]), numpy.round(numpy.random.rand(2) * factors))).astype(int)
    #                 else:
    #                     factors = numpy.array([curves_count - 1, rotation_factor - 1, sampling_factor - 1])
    #                     curve1_indices = numpy.round(numpy.random.rand(3) * factors).astype(int)
    #                     curve2_indices = numpy.round(numpy.random.rand(3) * factors).astype(int)
    #
    #                 if numpy.all(curve1_indices == curve2_indices):
    #                     continue
    #
    #                 curve1_indices_str = numpy.array2string(curve1_indices, precision=0, separator=',')
    #                 curve2_indices_str = numpy.array2string(curve2_indices, precision=0, separator=',')
    #
    #                 if curve1_indices_str > curve2_indices_str:
    #                     pair_key = f'{curve1_indices_str}_{curve2_indices_str}'
    #                 else:
    #                     pair_key = f'{curve2_indices_str}_{curve1_indices_str}'
    #
    #                 if pair_key in pairs_keys:
    #                     continue
    #
    #                 pairs_keys[pair_key] = {}
    #                 pairs[pair_index] = numpy.concatenate((numpy.array([int(positive)]), curve1_indices, curve2_indices)).astype(int)
    #                 break
    #
    #     print(f'\r        - Creating pair #{pair_index}')
    #     return pairs

    # @staticmethod
    # def _generate_positive_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor):
    #     return DatasetGenerator._generate_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor, True)
    #
    # @staticmethod
    # def _generate_negative_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor):
    #     return DatasetGenerator._generate_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor, False)

    # @staticmethod
    # def _save_dataset_metadata(dir_path, curves_count, pairs_per_curve, rotation_factor, sampling_factor, sample_points):
    #
    #     metadata = {
    #         'curves_count': curves_count,
    #         'rotation_factor': rotation_factor,
    #         'sampling_factor': sampling_factor,
    #         'sample_points': sample_points,
    #         'pairs': None
    #     }
    #
    #     print('    - Generating positive pairs:')
    #     positive_pairs = DatasetGenerator._generate_positive_pairs(
    #         curves_count=curves_count,
    #         pairs_per_curve=pairs_per_curve,
    #         rotation_factor=rotation_factor,
    #         sampling_factor=sampling_factor)
    #
    #     print('    - Generating negative pairs:')
    #     negative_pairs = DatasetGenerator._generate_negative_pairs(
    #         curves_count=curves_count,
    #         pairs_per_curve=pairs_per_curve,
    #         rotation_factor=rotation_factor,
    #         sampling_factor=sampling_factor)
    #
    #     print('    - Interweaving positive and negative pairs...', end="")
    #     metadata['pairs'] = numpy.empty(shape=[positive_pairs.shape[0] + negative_pairs.shape[0], 7]).astype(int)
    #     metadata['pairs'][0::2] = positive_pairs
    #     metadata['pairs'][1::2] = negative_pairs
    #     print('\r    - Interweaving positive and negative pairs... Done.')
    #
    #     print('    - Saving metadata...', end="")
    #     numpy.save(os.path.normpath(os.path.join(dir_path, 'metadata.npy')), metadata)
    #     print('\r    - Saving metadata... Done.')

    # @staticmethod
    # def _process_curves(raw_curves, predicate, rotation_factor, sampling_factor, sample_points, limit=None, chunk_size=5):
    #     if limit is not None:
    #         raw_curves = raw_curves[:limit]
    #
    #     extended_raw_curves = []
    #     for i, curve in enumerate(raw_curves):
    #         extended_raw_curves.append({
    #             'curve': curve,
    #             'curve_id': i,
    #             'rotation_factor': rotation_factor,
    #             'sampling_factor': sampling_factor,
    #             'sample_points': sample_points
    #         })
    #
    #     extended_raw_curves_chunks = DatasetGenerator._chunks(extended_raw_curves, chunk_size)
    #
    #     print('    - Creating pool...', end="")
    #     pool = multiprocessing.Pool()
    #     print('\r    - Creating pool... Done.')
    #
    #     print('    - Processing curves...', end="")
    #     for i, processed_curves_chunk in enumerate(pool.imap_unordered(DatasetGenerator._process_curves_chunk, extended_raw_curves_chunks, 1)):
    #         print('\r    - Processing curves... {0:.1%} Done.'.format((i+1) / len(extended_raw_curves_chunks)), end="")
    #         for curve in processed_curves_chunk:
    #             predicate(curve)
    #
    #     print('\r    - Processing curves... {0:.1%} Done.'.format((i + 1) / len(extended_raw_curves_chunks)))
    #
    # @staticmethod
    # def _process_curves_chunk(extended_raw_curves_chunk):
    #     generated_curves = []
    #     for curve in extended_raw_curves_chunk:
    #         generated_curves.append(CurveDataGenerator(
    #             curve=curve['curve'],
    #             curve_id=curve['curve_id'],
    #             rotation_factor=curve['rotation_factor'],
    #             sampling_factor=curve['sampling_factor'],
    #             sample_points=curve['sample_points']))
    #
    #     return generated_curves

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
