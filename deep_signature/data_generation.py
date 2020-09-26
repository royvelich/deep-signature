from multiprocessing import Pool
import scipy.io
import random
import os
import scipy.io
import numpy
import re
import scipy.stats as ss
from scipy.stats import truncnorm
import math


class CurveSample:
    def __init__(self, curve_sample_id, curve, sample_points):
        self._curve_sample_id = curve_sample_id
        self._bins = curve.shape[0]

        density_threshold = 1.0 / sample_points

        while True:
            self._dist = DistGenerator.generate_random_dist(self._bins)

            valid = True
            for i in range(self._bins):
                if self._dist[i] > density_threshold:
                    valid = False
                    break

            if valid is True:
                break

        accumulated_density = 0
        self._indices = []
        for i in range(self._bins):
            accumulated_density = accumulated_density + self._dist[i]
            if accumulated_density >= (density_threshold * len(self._indices)):
                self._indices.append(i)

        self._sorted_indices = numpy.sort(self._indices)[:sample_points]
        self._sampled_curve = curve[self._sorted_indices]
        assert (self._sampled_curve.shape[0] == sample_points)

    @property
    def indices(self):
        return self._indices

    @property
    def sorted_indices(self):
        return self._sorted_indices

    @property
    def sampled_curve(self):
        return self._sampled_curve

    @property
    def dist(self):
        return self._dist

    @property
    def bins(self):
        return self._bins

    def save(self, dir_path):
        curve_sample_dir_path = os.path.normpath(os.path.join(dir_path, str(self._curve_sample_id)))
        os.mkdir(curve_sample_dir_path)
        numpy.save(os.path.normpath(os.path.join(curve_sample_dir_path, 'sample.npy')), self._sampled_curve)


class CurveConfiguration:
    def __init__(self, curve_configuration_id, curve, radians, reflection, sampling_factor, sample_points):
        self._curve_configuration_id = curve_configuration_id
        self._radians = radians
        self._reflection = reflection
        self._sampling_factor = sampling_factor
        self._sample_points = sample_points
        self._curve = CurveConfiguration._apply_transform(curve, radians, reflection)
        self._curve_samples = CurveConfiguration._generate_curve_samples(curve=self._curve, sampling_factor=sampling_factor, sample_points=sample_points)

    @property
    def curve(self):
        return self._curve

    @property
    def curve_samples(self):
        return self._curve_samples

    def save(self, dir_path):
        curve_configuration_dir_path = os.path.normpath(os.path.join(dir_path, str(self._curve_configuration_id)))
        os.mkdir(curve_configuration_dir_path)
        for curve_sample in self._curve_samples:
            curve_sample.save(curve_configuration_dir_path)

    @staticmethod
    def _generate_curve_samples(curve, sampling_factor, sample_points):
        curve_samples = []
        for i in range(sampling_factor):
            curve_samples.append(CurveSample(curve_sample_id=i, curve=curve, sample_points=sample_points))

        return curve_samples

    @staticmethod
    def _apply_transform(curve, radians, reflection):
        reflection_transform = CurveConfiguration._identity_2d()
        if reflection == 'horizontal':
            reflection_transform = CurveConfiguration._horizontal_reflection_2d()
        elif reflection == 'vertical':
            reflection_transform = CurveConfiguration._vertical_reflection_2d()

        rotation_transform = CurveConfiguration._rotation_2d(radians)
        transform = rotation_transform.dot(reflection_transform)
        transformed_curve = curve.dot(transform)

        return transformed_curve

    @staticmethod
    def _rotation_2d(radians):
        c, s = numpy.cos(radians), numpy.sin(radians)
        return numpy.array([[c, s], [-s, c]])

    @staticmethod
    def _horizontal_reflection_2d():
        return numpy.array([[1, 0], [0, -1]])

    @staticmethod
    def _vertical_reflection_2d():
        return numpy.array([[-1, 0], [0, 1]])

    @staticmethod
    def _identity_2d():
        return numpy.array([[1, 0], [0, 1]])


class Curve:
    _reflection_types = ['none', 'horizontal', 'vertical']

    def __init__(self, curve_id, curve, rotation_factor, sampling_factor, sample_points):
        self._curve_id = curve_id
        self._rotation_factor = rotation_factor
        self._sampling_factor = sampling_factor
        self._sample_points = sample_points
        self._curve = Curve._normalize_curve(curve)
        self._curve_configurations = Curve._generate_curve_configurations(
            curve=self._curve,
            rotation_factor=rotation_factor,
            sampling_factor=sampling_factor,
            sample_points=sample_points)

    def save(self, dir_path):
        curve_dir_path = os.path.normpath(os.path.join(dir_path, str(self._curve_id)))
        os.mkdir(curve_dir_path)
        for curve_configuration in self._curve_configurations:
            curve_configuration.save(curve_dir_path)

    @property
    def curve(self):
        return self._curve

    @property
    def curve_configurations(self):
        return self._curve_configurations

    @staticmethod
    def _generate_curve_configurations(curve, rotation_factor, sampling_factor, sample_points):
        curve_configurations = []
        for i, reflection in enumerate(Curve._reflection_types):
            for rotation_index in range(rotation_factor):
                curve_configuration_id = rotation_factor * i + rotation_index
                radians = rotation_index * ((2 * math.pi) / rotation_factor)
                curve_configurations.append(
                    CurveConfiguration(
                        curve_configuration_id=curve_configuration_id,
                        curve=curve,
                        radians=radians,
                        reflection=reflection,
                        sampling_factor=sampling_factor,
                        sample_points=sample_points))

        return curve_configurations

    @staticmethod
    def _normalize_curve(curve):
        curve_center = numpy.mean(curve, axis=0)
        normalized_curve = curve - curve_center
        return normalized_curve


class DatasetGenerator:
    def __init__(self):
        self._raw_curves = []

    def save(self, dir_path, pairs_per_curve, rotation_factor, sampling_factor, sample_points, metadata_only=False, chunk_size=5):

        if metadata_only is False:
            print('Saving dataset curves:')

            def save_curve(curve):
                curve.save(dir_path=dir_path)

            DatasetGenerator._process_curves(
                raw_curves=self._raw_curves,
                predicate=save_curve,
                rotation_factor=rotation_factor,
                sampling_factor=sampling_factor,
                sample_points=sample_points,
                limit=None,
                chunk_size=chunk_size)

        print('Saving dataset metadata:')

        DatasetGenerator._save_dataset_metadata(
            dir_path=dir_path,
            curves_count=len(self._raw_curves),
            pairs_per_curve=pairs_per_curve,
            rotation_factor=rotation_factor,
            sampling_factor=sampling_factor,
            sample_points=sample_points)

    def generate_curves(self, rotation_factor, sampling_factor, sample_points, limit=5, chunk_size=5):

        print('Generating dataset curves:')

        curves = []

        def add_curve(curve):
            curves.append(curve)

        DatasetGenerator._process_curves(
            raw_curves=self._raw_curves,
            predicate=add_curve,
            rotation_factor=rotation_factor,
            sampling_factor=sampling_factor,
            sample_points=sample_points,
            limit=limit,
            chunk_size=chunk_size)

        return curves

    def load_raw_curves(self, dir_path):

        print('Loading raw curves:')

        base_dir = os.path.normpath(dir_path)
        datasets = []
        print(f'    - Reading curve files...\r', end="")
        for sub_dir, dirs, files in os.walk(base_dir):

            if sub_dir == base_dir:
                continue

            current_dataset = {}
            indices = []
            current_dataset['indices'] = indices
            for file in files:
                file_name, file_ext = os.path.splitext(file)
                if file_ext == '.mat':
                    mat_file_path = os.path.join(sub_dir, file)
                    mat_obj = scipy.io.loadmat(mat_file_path)
                    curves_obj = mat_obj['curves']
                    current_curves = []
                    current_dataset['mat_file_path'] = mat_file_path
                    current_dataset['curves'] = current_curves
                    for i in range(curves_obj.size):
                        current_curve_obj = curves_obj[i][0]
                        x = current_curve_obj[1]
                        y = current_curve_obj[2]
                        current_curve = numpy.concatenate((x, y), axis=1)
                        # if (current_curve.shape[0] >= 2000) and (current_curve.shape[0] <= 3000):
                        current_curves.append(current_curve)
                elif file_ext == '.png':
                    digits_re = re.compile('[0-9]+')
                    match = re.search(digits_re, file_name)
                    if match:
                        index = int(match.group())
                        indices.append(index)

            datasets.append(current_dataset)

        print(f'    - Reading curve files... Done.\r', end="")
        print('\r')
        print(f'    - Merging datasets...\r', end="")
        curves = []
        for dataset in datasets:
            dataset_curves = dataset['curves']
            dataset_indices = dataset['indices']
            selected_curves = [dataset_curves[i - 1] for i in dataset_indices]
            curves.extend(selected_curves)

        print(f'    - Merging datasets... Done.\r', end="")

        print('\r')
        print(f'    - Shuffling curves...\r', end="")
        random.shuffle(curves)
        print(f'    - Shuffling curves... Done.\r', end="")
        print('\r')

        self._raw_curves = curves

        raw_curves_count = len(self._raw_curves)

        print(f'    - {raw_curves_count} raw curves loaded.')

    @staticmethod
    def _generate_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor, positive):
        pairs = numpy.empty(shape=[curves_count * pairs_per_curve, 7]).astype(int)
        pairs_keys = {}
        for curve_index in range(curves_count):
            for curve_pair_index in range(pairs_per_curve):
                pair_index = pairs_per_curve * curve_index + curve_pair_index
                print(f'        - Creating pair #{pair_index}\r', end="")
                while True:
                    if positive is True:
                        factors = numpy.array([rotation_factor - 1, sampling_factor - 1])
                        curve1_indices = numpy.concatenate((numpy.array([curve_index]), numpy.round(numpy.random.rand(2) * factors))).astype(int)
                        curve2_indices = numpy.concatenate((numpy.array([curve_index]), numpy.round(numpy.random.rand(2) * factors))).astype(int)
                    else:
                        factors = numpy.array([curves_count - 1, rotation_factor - 1, sampling_factor - 1])
                        curve1_indices = numpy.round(numpy.random.rand(3) * factors).astype(int)
                        curve2_indices = numpy.round(numpy.random.rand(3) * factors).astype(int)

                    if numpy.all(curve1_indices == curve2_indices):
                        continue

                    curve1_indices_str = numpy.array2string(curve1_indices, precision=0, separator=',')
                    curve2_indices_str = numpy.array2string(curve2_indices, precision=0, separator=',')

                    if curve1_indices_str > curve2_indices_str:
                        pair_key = f'{curve1_indices_str}_{curve2_indices_str}'
                    else:
                        pair_key = f'{curve2_indices_str}_{curve1_indices_str}'

                    if pair_key in pairs_keys:
                        continue

                    pairs_keys[pair_key] = {}
                    pairs[pair_index] = numpy.concatenate((numpy.array([int(positive)]), curve1_indices, curve2_indices)).astype(int)
                    break

        return pairs

    @staticmethod
    def _generate_positive_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor):
        return DatasetGenerator._generate_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor, True)

    @staticmethod
    def _generate_negative_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor):
        return DatasetGenerator._generate_pairs(curves_count, pairs_per_curve, rotation_factor, sampling_factor, False)

    @staticmethod
    def _save_dataset_metadata(dir_path, curves_count, pairs_per_curve, rotation_factor, sampling_factor, sample_points):

        metadata = {
            'curves_count': curves_count,
            'rotation_factor': rotation_factor,
            'sampling_factor': sampling_factor,
            'sample_points': sample_points,
            'pairs': None
        }

        print('    - Generating positive pairs:')
        positive_pairs = DatasetGenerator._generate_positive_pairs(
            curves_count=curves_count,
            pairs_per_curve=pairs_per_curve,
            rotation_factor=rotation_factor,
            sampling_factor=sampling_factor)

        print('\r')
        print('    - Generating negative pairs:')
        negative_pairs = DatasetGenerator._generate_negative_pairs(
            curves_count=curves_count,
            pairs_per_curve=pairs_per_curve,
            rotation_factor=rotation_factor,
            sampling_factor=sampling_factor)
        print('\r')

        print('    - Interweaving positive and negative pairs...\r', end="")
        metadata['pairs'] = numpy.empty(shape=[positive_pairs.shape[0] + negative_pairs.shape[0], 7]).astype(int)
        metadata['pairs'][0::2] = positive_pairs
        metadata['pairs'][1::2] = negative_pairs
        print('    - Interweaving positive and negative pairs... Done.\r', end="")
        print('\r')

        print('    - Saving metadata...\r', end="")
        numpy.save(os.path.normpath(os.path.join(dir_path, 'metadata.npy')), metadata)
        print('    - Saving metadata... Done.\r', end="")
        print('\r')

    @staticmethod
    def _process_curves(raw_curves, predicate, rotation_factor, sampling_factor, sample_points, limit=None, chunk_size=5):
        if limit is not None:
            raw_curves = raw_curves[:limit]

        extended_raw_curves = []
        for i, curve in enumerate(raw_curves):
            extended_raw_curves.append({
                'curve': curve,
                'curve_id': i,
                'rotation_factor': rotation_factor,
                'sampling_factor': sampling_factor,
                'sample_points': sample_points
            })

        extended_raw_curves_chunks = DatasetGenerator._chunks(extended_raw_curves, chunk_size)

        print('    - Creating pool...\r', end="")
        pool = Pool()
        print('    - Creating pool... Done.\r', end="")

        print('\r')
        print('    - Processing curves...\r', end="")
        for i, processed_curves_chunk in enumerate(pool.imap_unordered(DatasetGenerator._process_curves_chunk, extended_raw_curves_chunks, 1)):
            print('    - Processing curves... {0:.1%} Done.\r'.format((i+1) / len(extended_raw_curves_chunks)), end="")
            for curve in processed_curves_chunk:
                predicate(curve)
        print('\r')

    @staticmethod
    def _process_curves_chunk(extended_raw_curves_chunk):
        generated_curves = []
        for curve in extended_raw_curves_chunk:
            generated_curves.append(Curve(
                curve=curve['curve'],
                curve_id=curve['curve_id'],
                rotation_factor=curve['rotation_factor'],
                sampling_factor=curve['sampling_factor'],
                sample_points=curve['sample_points']))

        return generated_curves

    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    @staticmethod
    def _chunks(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]


class DistGenerator:

    # https://stackoverflow.com/questions/37411633/how-to-generate-a-random-normal-distribution-of-integers
    @staticmethod
    def generate_normal_dist(bins, loc, scale):
        is_even = bins % 2 == 0
        half_bins = math.floor(bins / 2)

        start, stop = -half_bins, half_bins
        if not is_even:
            stop = stop + 1

        x = numpy.arange(start, stop) + half_bins
        x_lower, x_upper = x - 0.5, x + 0.5

        cfd_lower = ss.norm.cdf(x_lower, loc=loc, scale=scale)
        cfd_upper = ss.norm.cdf(x_upper, loc=loc, scale=scale)
        dist = cfd_upper - cfd_lower

        # normalize the distribution bins so their sum is 1
        dist = dist / dist.sum()

        return dist

    @staticmethod
    def generate_random_normal_dist(bins):
        mean = bins / 2

        truncated_normal = DistGenerator.generate_truncated_normal(mean=0.5, sd=0.25, low=0, upp=1)
        var = truncated_normal.rvs(1) * 10*numpy.sqrt(bins)
        dist = DistGenerator.generate_normal_dist(bins, mean, var)

        return numpy.roll(dist, int(numpy.round(numpy.random.rand(1) * bins)))

    @staticmethod
    def generate_random_dist(bins):
        truncated_normal = DistGenerator.generate_truncated_normal(mean=17, sd=4, low=2, upp=30)
        count = 15

        dist = None
        for _ in range(count):
            current_dist = DistGenerator.generate_random_normal_dist(bins)
            if dist is None:
                dist = current_dist
            else:
                dist = dist + current_dist

        dist = dist / dist.sum()
        return dist

    # https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
    @staticmethod
    def generate_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
