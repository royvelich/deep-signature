from multiprocessing import Pool
import os
import scipy.io
import numpy
import re
import math
import random
import deep_signature.dist


class CurveSample:
    def __init__(self, curve_sample_id, curve, sample_points):
        self._curve_sample_id = curve_sample_id
        self._bins = curve.shape[0]
        self._dist = deep_signature.dist.DistGenerator.generate_random_dist(self._bins)

        density_threshold = 1.0 / sample_points
        accumulated_density = 0
        self._indices = [0]
        for i in range(self._bins):
            accumulated_density = accumulated_density + self._dist[i]
            if accumulated_density > density_threshold:
                accumulated_density = accumulated_density - density_threshold
                self._indices.append(i)

        self._sorted_indices = numpy.sort(self._indices)
        self._sampled_curve = curve[self._sorted_indices]
        self._sampled_curve = self._sampled_curve[:sample_points]

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

    def save(self, dir_path, pairs_per_curve, rotation_factor, sampling_factor, sample_points, chunk_size=5):

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
    def _save_dataset_metadata(dir_path, curves_count, pairs_per_curve, rotation_factor, sampling_factor, sample_points):

        metadata = {}

        metadata['curves_count'] = curves_count
        metadata['rotation_factor'] = rotation_factor
        metadata['sampling_factor'] = sampling_factor
        metadata['sample_points'] = sample_points
        metadata['pairs'] = numpy.empty(shape=[2*curves_count*pairs_per_curve, 7]).astype(int)

        pairing_dict = {'positive': [], 'negative': []}
        pair_index = 0

        factors = numpy.array([rotation_factor, sampling_factor])
        for curve_index in range(curves_count):
            for _ in range(pairs_per_curve):
                print(f'    - Creating positive pair #{pair_index}\r', end="")
                while True:
                    curve1_indices = numpy.concatenate((numpy.array([curve_index]), numpy.round(numpy.random.rand(2) * factors))).astype(int)
                    curve2_indices = numpy.concatenate((numpy.array([curve_index]), numpy.round(numpy.random.rand(2) * factors))).astype(int)
                    if numpy.all(curve1_indices == curve2_indices):
                        continue

                    curve1_indices_str = numpy.array2string(curve1_indices, precision=0, separator=',')
                    curve2_indices_str = numpy.array2string(curve2_indices, precision=0, separator=',')

                    if curve1_indices_str > curve2_indices_str:
                        pair_key = f'{curve1_indices_str}_{curve2_indices_str}'
                    else:
                        pair_key = f'{curve2_indices_str}_{curve1_indices_str}'

                    if pair_key in pairing_dict['positive']:
                        continue

                    metadata['pairs'][pair_index] = numpy.concatenate((numpy.array([1]), curve1_indices, curve2_indices)).astype(int)
                    pair_index = pair_index + 1
                    break

        print('\r')

        factors = numpy.array([curves_count, rotation_factor, sampling_factor])
        for curve_index in range(curves_count):
            for _ in range(pairs_per_curve):
                print(f'    - Creating negative pair #{pair_index}\r', end="")
                while True:
                    curve1_indices = numpy.round(numpy.random.rand(3) * factors).astype(int)
                    curve2_indices = numpy.round(numpy.random.rand(3) * factors).astype(int)

                    if numpy.all(curve1_indices[0] == curve2_indices[0]):
                        continue

                    curve1_indices_str = numpy.array2string(curve1_indices, precision=0, separator=',')
                    curve2_indices_str = numpy.array2string(curve2_indices, precision=0, separator=',')

                    if curve1_indices_str > curve2_indices_str:
                        pair_key = f'{curve1_indices_str}_{curve2_indices_str}'
                    else:
                        pair_key = f'{curve2_indices_str}_{curve1_indices_str}'

                    if pair_key in pairing_dict['negative']:
                        continue

                    metadata['pairs'][pair_index] = numpy.concatenate((numpy.array([0]), curve1_indices, curve2_indices)).astype(int)
                    pair_index = pair_index + 1
                    break

        print('\r')
        print(f'    - Saving metadata...\r', end="")
        numpy.save(os.path.normpath(os.path.join(dir_path, 'metadata.npy')), metadata)
        print(f'    - Saving metadata... Done.\r', end="")
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

        print(f'    - Creating pool...\r', end="")
        pool = Pool()
        print(f'    - Creating pool... Done.\r', end="")

        print('\r')
        print('     - Processing curves...')
        for i, processed_curves_chunk in enumerate(pool.imap_unordered(DatasetGenerator._process_curves_chunk, extended_raw_curves_chunks, 1)):
            print('     - Processing curves... {0:.1%} Done.\r'.format((i+1) / len(extended_raw_curves_chunks)), end="")
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