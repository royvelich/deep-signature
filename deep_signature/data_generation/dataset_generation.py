# python peripherals
import os
import multiprocessing
import pathlib

# numpy
import numpy

# deep_signature
from deep_signature.data_manipulation import curve_sampling, curve_processing
from deep_signature.data_generation.curve_generation import CurvesGenerator
from deep_signature.utils import utils
from deep_signature.linalg import euclidean_transform, affine_transform


class CurveManager:
    def __init__(self, curve):
        self._curve = curve_processing.translate_curve(curve=curve, offset=-numpy.mean(curve, axis=0))
        self._curve_points_count = self._curve.shape[0]
        self._curvature = curve_processing.calculate_euclidean_curvature(self._curve)

    @property
    def curve(self):
        return self._curve

    @property
    def curvature(self):
        return self._curvature


class TuplesDatasetGenerator:
    _file_name = 'tuples'
    _label = 'tuples'

    @classmethod
    def load_tuples(cls, dir_path):
        return numpy.load(file=os.path.normpath(os.path.join(dir_path, f'{cls._file_name}.npy')), allow_pickle=True)

    @classmethod
    def generate_tuples(cls, dir_path, curves_dir_path, chunksize, **kwargs):
        tuples = []
        curves = CurvesGenerator.load_curves(curves_dir_path)
        iterable = cls._zip_iterable(curves=curves, **kwargs)

        def reduce_func(tuple):
            if tuple is not None:
                tuples.append(tuple)

        utils.par_proc(
            map_func=cls._map_func,
            reduce_func=reduce_func,
            iterable=iterable,
            chunksize=chunksize,
            label=cls._label
        )

        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        numpy.save(file=os.path.normpath(os.path.join(dir_path, f'{cls._file_name}.npy')), arr=numpy.array(tuples, dtype=object))

    @classmethod
    def _map_func(cls, kwargs):
        return cls._generate_tuple(**kwargs)

    @classmethod
    def _zip_iterable(cls, curves, sections_density, **kwargs):
        center_point_indices_pack = []
        curve_indices_pack = []

        for i, curve in enumerate(curves):
            sections_count = int(curve.shape[0] * sections_density)
            center_point_indices = numpy.linspace(
                start=0,
                stop=curve.shape[0],
                num=sections_count,
                endpoint=False,
                dtype=int)

            curve_indices_pack.extend([i] * sections_count)
            center_point_indices_pack.extend(center_point_indices)

        items_count = len(curve_indices_pack)
        curves_pack = [curves] * items_count

        names, data = cls._generate_zip_data(items_count, **kwargs)
        data = [curves_pack, curve_indices_pack, center_point_indices_pack] + data
        names = ['curves', 'curve_index', 'center_point_index'] + names
        zipped_data = zip(*data)
        iterable = [dict(zip(names, values)) for values in zipped_data]
        return iterable

    @staticmethod
    def _generate_tuple():
        raise NotImplemented

    @staticmethod
    def _generate_zip_data():
        raise NotImplemented


class CurvatureTupletsDatasetGenerator(TuplesDatasetGenerator):
    _file_name = 'tuplets'
    _label = 'tuplets'

    @classmethod
    def _generate_tuple(cls, curves, curve_index, center_point_index, negative_examples_count, supporting_points_count, max_offset):
        input = []
        tuplet = {
            'input': input
        }

        curve = curves[curve_index]
        transform = cls._generate_curve_transform()
        for _ in range(2):
            sample = curve_sampling.sample_curve_point_neighbourhood(
                curve=curve,
                center_point_index=center_point_index,
                supporting_point_count=supporting_points_count,
                max_offset=max_offset)
            sample = curve_processing.transform_curve(curve=sample, transform=transform)
            sample = curve_processing.normalize_curve(curve=sample)
            input.append(sample)

        flipped_anchor = numpy.flip(m=input[0], axis=0).copy()
        sample = curve_processing.normalize_curve(curve=flipped_anchor)
        input.append(sample)

        rng = numpy.random.default_rng()
        indices_pool = numpy.arange(start=0, stop=len(curves))
        indices_pool = numpy.delete(indices_pool, curve_index)
        indices = rng.choice(a=indices_pool, size=negative_examples_count, replace=False)
        for index in indices:
            current_curve = curves[index]
            sample = curve_sampling.sample_curve_point_neighbourhood(
                curve=current_curve,
                center_point_index=int(numpy.random.randint(current_curve.shape[0])),
                supporting_point_count=supporting_points_count,
                max_offset=max_offset)
            transform = cls._generate_curve_transform()
            sample = curve_processing.transform_curve(curve=sample, transform=transform)
            sample = curve_processing.normalize_curve(curve=sample)
            input.append(sample)

        return tuplet

    @staticmethod
    def _generate_zip_data(items_count, negative_examples_count, supporting_points_count, max_offset=None):
        negative_examples_count_pack = [negative_examples_count] * items_count
        supporting_points_count_pack = [supporting_points_count] * items_count
        max_offset_pack = [max_offset] * items_count
        data = [negative_examples_count_pack, supporting_points_count_pack, max_offset_pack]
        names = ['negative_examples_count', 'supporting_points_count', 'max_offset']
        return names, data

    @staticmethod
    def _generate_curve_transform():
        raise NotImplemented


class EuclideanCurvatureTupletsDatasetGenerator(CurvatureTupletsDatasetGenerator):
    _file_name = 'tuplets'
    _label = 'tuplets'

    @staticmethod
    def _generate_curve_transform():
        return euclidean_transform.identity_2d()


class AffineCurvatureTupletsDatasetGenerator(CurvatureTupletsDatasetGenerator):
    _file_name = 'tuplets'
    _label = 'tuplets'

    @staticmethod
    def _generate_curve_transform():
        return affine_transform.random_equiaffine_transform_2d()


class ArcLengthTupletsDatasetGenerator(TuplesDatasetGenerator):
    _file_name = 'tuplets'
    _label = 'tuplets'

    @staticmethod
    def _sample_curve_section(curve, supporting_points_count, start_point_index, end_point_index):
        sample = curve_sampling.sample_curve_section(
            curve=curve,
            supporting_points_count=supporting_points_count,
            start_point_index=start_point_index,
            end_point_index=end_point_index)
        sample = curve_processing.normalize_curve(curve=sample, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)

        # value = int(numpy.random.randint(0, 2, size=1))
        # if value == 1:
        #     sample = numpy.flip(m=sample, axis=0)

        return sample

    @staticmethod
    def _generate_tuple(curves, curve_index, center_point_index, negative_examples_count, supporting_points_count, min_perturbation, max_perturbation, min_offset, max_offset):
        input = []
        factors = []

        curve = curves[curve_index]
        if curve.shape[0] < 2500:
            return None

        tuplet = {
            'input': input,
            'factors': factors,
            # 'curve': curve
        }

        raw_offset = numpy.random.randint(max_offset, size=2)
        offset = numpy.maximum(raw_offset, [min_offset] * 2)
        start_point_index = numpy.mod(center_point_index - offset[0], curve.shape[0])
        end_point_index = numpy.mod(center_point_index + offset[1], curve.shape[0])

        # anchor
        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve,
            supporting_points_count=supporting_points_count,
            start_point_index=start_point_index,
            end_point_index=end_point_index)
        input.append(sample)
        factors.append(1)

        # tuplet['anchor_indices'] = [start_point_index, end_point_index]

        # positive example
        sample1 = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve,
            supporting_points_count=supporting_points_count,
            start_point_index=start_point_index,
            end_point_index=center_point_index)

        # tuplet['positive_indices1'] = [start_point_index, center_point_index]

        sample2 = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve,
            supporting_points_count=supporting_points_count,
            start_point_index=center_point_index,
            end_point_index=end_point_index)

        # tuplet['positive_indices2'] = [center_point_index, end_point_index]
        input.append(sample1)
        input.append(sample2)
        factors.append(1)

        # negative examples
        for _ in range(negative_examples_count):
            perturbation = numpy.random.randint(low=min_perturbation, high=max_perturbation, size=2)

            negative_example_type = numpy.random.choice(['longer', 'shorter'])
            if negative_example_type == 'longer':
                perturbation[0] = -perturbation[0]
                factors.append(-1)
            else:
                perturbation[1] = -perturbation[1]
                factors.append(1)

            current_start_point_index = numpy.mod(start_point_index + perturbation[0], curve.shape[0])
            current_end_point_index = numpy.mod(end_point_index + perturbation[1], curve.shape[0])

            sample1 = ArcLengthTupletsDatasetGenerator._sample_curve_section(
                curve=curve,
                supporting_points_count=supporting_points_count,
                start_point_index=current_start_point_index,
                end_point_index=center_point_index)

            sample2 = ArcLengthTupletsDatasetGenerator._sample_curve_section(
                curve=curve,
                supporting_points_count=supporting_points_count,
                start_point_index=center_point_index,
                end_point_index=current_end_point_index)
            input.append(sample1)
            input.append(sample2)

        return tuplet

    @staticmethod
    def _generate_zip_data(items_count, negative_examples_count, supporting_points_count, min_perturbation, max_perturbation, min_offset, max_offset=None):
        negative_examples_count_pack = [negative_examples_count] * items_count
        supporting_points_count_pack = [supporting_points_count] * items_count
        min_offset_pack = [min_offset] * items_count
        max_offset_pack = [max_offset] * items_count
        min_perturbation_pack = [min_perturbation] * items_count
        max_perturbation_pack = [max_perturbation] * items_count
        data = [negative_examples_count_pack, supporting_points_count_pack, min_perturbation_pack, max_perturbation_pack, min_offset_pack, max_offset_pack]
        names = ['negative_examples_count', 'supporting_points_count', 'min_perturbation', 'max_perturbation', 'min_offset', 'max_offset']
        return data, names