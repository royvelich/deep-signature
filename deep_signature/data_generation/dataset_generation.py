# python peripherals
import os
import multiprocessing
import pathlib
import random

# numpy
import numpy

# deep_signature
from deep_signature.data_manipulation import curve_sampling, curve_processing
from deep_signature.stats import discrete_distribution
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
    def generate_tuples(cls, pool, curves, curves_dir_path, chunksize, **kwargs):
        tuples = []
        # curves = CurvesGenerator.load_curves(curves_dir_path)
        iterable = cls._zip_iterable(curves=curves, **kwargs)

        def reduce_func(tuple):
            if tuple is not None:
                tuples.append(tuple)

        utils.par_proc(
            map_func=cls._map_func,
            reduce_func=reduce_func,
            iterable=iterable,
            label=cls._label,
            pool=pool,
            chunksize=chunksize)

        return tuples

    @classmethod
    def save_tuples(cls, pool, dir_path, curves_dir_path, chunksize, **kwargs):
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
            label=cls._label,
            pool=pool,
            chunksize=chunksize)

        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        numpy.save(file=os.path.normpath(os.path.join(dir_path, f'{cls._file_name}.npy')), arr=numpy.array(tuples, dtype=object))

    @classmethod
    def _map_func(cls, kwargs):
        return cls._generate_tuple(**kwargs)

    @classmethod
    def _zip_iterable(cls, curves, tuples_count, **kwargs):
        center_point_indices_pack = []
        curve_indices_pack = []

        rng = numpy.random.default_rng()
        indices_pool = numpy.arange(start=0, stop=len(curves))
        indices = rng.choice(a=indices_pool, size=tuples_count, replace=False)
        curves_pack = [curves] * tuples_count

        # names, data = cls._generate_zip_data(items_count, **kwargs)
        data = [curves_pack, indices, center_point_indices_pack]
        names = ['curves', 'curve_index']
        zipped_data = zip(*data)
        iterable = [dict(zip(names, values)) for values in zipped_data]
        return iterable

    @staticmethod
    def _generate_tuple():
        raise NotImplemented

    @staticmethod
    def _generate_zip_data():
        raise NotImplemented


class EuclideanTransform:
    @staticmethod
    def _generate_curve_transform():
        return euclidean_transform.generate_random_euclidean_transform_2d()


class EquiaffineTransform:
    @staticmethod
    def _generate_curve_transform():
        return affine_transform.generate_random_equiaffine_transform_2d()


class AffineTransform:
    @staticmethod
    def _generate_curve_transform():
        return affine_transform.generate_random_affine_transform_2d()


class CurvatureTupletsDatasetGenerator(TuplesDatasetGenerator):
    _file_name = 'tuplets'
    _label = 'tuplets'

    @classmethod
    def generate_tuple(cls, curves, curve_index, sampling_ratio, multimodality, offset_length, supporting_points_count):
        input = []
        tuplet = {
            'input': input
        }

        dist_index = 0
        curve = curve_processing.center_curve(curve=curves[curve_index])
        curve_points_count = curve.shape[0]
        sampling_points_count = int(sampling_ratio * curve_points_count)
        max_density = 1 / sampling_points_count
        discrete_distribution_pack = discrete_distribution.random_discrete_dist(bins=curve_points_count, multimodality=multimodality, max_density=1, count=4)
        center_point_index = int(numpy.random.randint(curve.shape[0], size=1))
        for i in range(2):
            transformed_curve = curve
            if i == 1:
                transform = cls._generate_curve_transform()
                transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)

            sample = curve_sampling.sample_curve_section_with_dist(
                curve=transformed_curve,
                center_point_index=center_point_index,
                dist=discrete_distribution_pack[dist_index],
                sampling_points_count=sampling_points_count,
                supporting_points_count=supporting_points_count)

            sample = curve_processing.normalize_curve(curve=sample)
            input.append(sample)
            dist_index = dist_index + 1

        flipped_anchor = numpy.flip(m=input[0], axis=0).copy()
        sample = curve_processing.normalize_curve(curve=flipped_anchor)
        input.append(sample)

        for i in range(2):
            while True:
                center_point_index_offset = int(numpy.random.randint(offset_length, size=1)) - int(offset_length/2)
                if center_point_index_offset != 0:
                    break
                # if numpy.abs(center_point_index_offset) > 5:
                #     break

            transform = cls._generate_curve_transform()
            transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)

            sample = curve_sampling.sample_curve_section_with_dist(
                curve=transformed_curve,
                center_point_index=numpy.mod(center_point_index + center_point_index_offset, transformed_curve.shape[0]),
                dist=discrete_distribution_pack[dist_index],
                sampling_points_count=sampling_points_count,
                supporting_points_count=supporting_points_count)

            sample = curve_processing.normalize_curve(curve=sample)
            input.append(sample)
            dist_index = dist_index + 1

        return tuplet

    @staticmethod
    def _generate_zip_data(items_count, negative_examples_count, supporting_points_count, max_offset=None):
        negative_examples_count_pack = [negative_examples_count] * items_count
        supporting_points_count_pack = [supporting_points_count] * items_count
        max_offset_pack = [max_offset] * items_count
        data = [negative_examples_count_pack, supporting_points_count_pack, max_offset_pack]
        names = ['negative_examples_count', 'supporting_points_count', 'max_offset']
        return names, data


class EuclideanCurvatureTupletsDatasetGenerator(CurvatureTupletsDatasetGenerator, EuclideanTransform):
    pass


class EquiaffineCurvatureTupletsDatasetGenerator(CurvatureTupletsDatasetGenerator, EquiaffineTransform):
    pass


class AffineCurvatureTupletsDatasetGenerator(CurvatureTupletsDatasetGenerator, AffineTransform):
    pass


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

        # flip_sample = bool(random.getrandbits(1))
        # if flip_sample is True:
        #     sample = numpy.flip(m=sample, axis=0).copy()

        sample = curve_processing.normalize_curve(curve=sample, force_ccw=False, force_end_point=True, index1=0, index2=1, center_index=0)
        return sample

    @classmethod
    def _generate_tuple(cls, curves, curve_index, center_point_index, exact_examples_count, inexact_examples_count, supporting_points_count, min_perturbation, max_perturbation, min_offset, max_offset):
        input = []
        factors = []

        curve = curves[curve_index]
        if curve.shape[0] < 2500:
            return None

        tuplet = {
            'input': input,
            'factors': factors
        }

        raw_offset = numpy.random.randint(max_offset, size=4)
        offset = numpy.maximum(raw_offset, [min_offset] * 4)
        index1 = center_point_index
        index2 = numpy.mod(index1 + offset[0], curve.shape[0])
        index3 = numpy.mod(index2 + offset[1], curve.shape[0])
        index4 = numpy.mod(index3 + offset[2], curve.shape[0])
        index5 = numpy.mod(index4 + offset[3], curve.shape[0])

        transform = cls._generate_curve_transform()
        transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)

        swap = bool(random.getrandbits(1))
        if swap is True:
            curve1 = curve
            curve2 = transformed_curve
        else:
            curve1 = transformed_curve
            curve2 = curve

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            supporting_points_count=supporting_points_count,
            start_point_index=index1,
            end_point_index=index3)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            supporting_points_count=supporting_points_count,
            start_point_index=index2,
            end_point_index=index4)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            supporting_points_count=supporting_points_count,
            start_point_index=index3,
            end_point_index=index5)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            supporting_points_count=supporting_points_count,
            start_point_index=index1,
            end_point_index=index4)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            supporting_points_count=supporting_points_count,
            start_point_index=index2,
            end_point_index=index5)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            supporting_points_count=supporting_points_count,
            start_point_index=index1,
            end_point_index=index5)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve2,
            supporting_points_count=supporting_points_count,
            start_point_index=index1,
            end_point_index=index2)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve2,
            supporting_points_count=supporting_points_count,
            start_point_index=index2,
            end_point_index=index3)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve2,
            supporting_points_count=supporting_points_count,
            start_point_index=index3,
            end_point_index=index4)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve2,
            supporting_points_count=supporting_points_count,
            start_point_index=index4,
            end_point_index=index5)
        input.append(sample)

        # exact examples
        # for i in range(exact_examples_count):
        #     transform = cls._generate_curve_transform()
        #     transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
        #
        #     positive_sample1 = ArcLengthTupletsDatasetGenerator._sample_curve_section(
        #         curve=transformed_curve,
        #         supporting_points_count=supporting_points_count,
        #         start_point_index=start_point_index,
        #         end_point_index=center_point_index)
        #
        #     positive_sample2 = ArcLengthTupletsDatasetGenerator._sample_curve_section(
        #         curve=transformed_curve,
        #         supporting_points_count=supporting_points_count,
        #         start_point_index=center_point_index,
        #         end_point_index=end_point_index)
        #
        #     input.append(positive_sample1)
        #     input.append(positive_sample2)
        #     factors.append(1)

        # inexact examples
        # for _ in range(inexact_examples_count):
        #     perturbation = numpy.random.randint(low=min_perturbation, high=max_perturbation, size=2)
        #
        #     negative_example_type = numpy.random.choice(['longer', 'shorter'])
        #     if negative_example_type == 'longer':
        #         perturbation[0] = -perturbation[0]
        #         factors.append(-1)
        #     else:
        #         perturbation[1] = -perturbation[1]
        #         factors.append(1)
        #
        #     current_start_point_index = numpy.mod(start_point_index + perturbation[0], curve.shape[0])
        #     current_end_point_index = numpy.mod(end_point_index + perturbation[1], curve.shape[0])
        #
        #     transform = cls._generate_curve_transform()
        #     transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
        #     sample1 = ArcLengthTupletsDatasetGenerator._sample_curve_section(
        #         curve=transformed_curve,
        #         supporting_points_count=supporting_points_count,
        #         start_point_index=current_start_point_index,
        #         end_point_index=center_point_index)
        #
        #     sample2 = ArcLengthTupletsDatasetGenerator._sample_curve_section(
        #         curve=transformed_curve,
        #         supporting_points_count=supporting_points_count,
        #         start_point_index=center_point_index,
        #         end_point_index=current_end_point_index)
        #     input.append(sample1)
        #     input.append(sample2)

        return tuplet

    @staticmethod
    def _generate_zip_data(items_count, exact_examples_count, inexact_examples_count, supporting_points_count, min_perturbation, max_perturbation, min_offset, max_offset=None):
        exact_examples_count_pack = [exact_examples_count] * items_count
        inexact_examples_count_pack = [inexact_examples_count] * items_count
        supporting_points_count_pack = [supporting_points_count] * items_count
        min_offset_pack = [min_offset] * items_count
        max_offset_pack = [max_offset] * items_count
        min_perturbation_pack = [min_perturbation] * items_count
        max_perturbation_pack = [max_perturbation] * items_count
        data = [exact_examples_count_pack, inexact_examples_count_pack, supporting_points_count_pack, min_perturbation_pack, max_perturbation_pack, min_offset_pack, max_offset_pack]
        names = ['exact_examples_count', 'inexact_examples_count', 'supporting_points_count', 'min_perturbation', 'max_perturbation', 'min_offset', 'max_offset']
        return names, data


class EuclideanArcLengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, EuclideanTransform):
    pass


class EquiaffineArcLengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, EquiaffineTransform):
    pass