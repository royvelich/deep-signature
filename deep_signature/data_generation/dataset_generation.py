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
    def generate_tuple(cls, curves, sampling_ratio, multimodality, supporting_points_count, offset_length):
        input = []
        tuplet = {
            'input': input
        }

        dist_index = 0
        curve_index = int(numpy.random.randint(curves.shape[0], size=1)),
        curve = curve_processing.center_curve(curve=curves[curve_index])
        curve_points_count = curve.shape[0]
        sampling_points_count = int(sampling_ratio * curve_points_count)
        discrete_distribution_pack = discrete_distribution.random_discrete_dist(bins=curve_points_count, multimodality=multimodality, max_density=1, count=11)
        center_point_index = int(numpy.random.randint(curve.shape[0], size=1))
        for i in range(2):
            transformed_curve = curve
            if i == 1:
                transform = cls._generate_curve_transform()
                transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)

            indices_pool = discrete_distribution.sample_discrete_dist(dist=discrete_distribution_pack[dist_index], sampling_points_count=sampling_points_count)
            sample = curve_sampling.sample_curve_neighborhood(
                curve=transformed_curve,
                center_point_index=center_point_index,
                indices_pool=indices_pool,
                supporting_points_count=supporting_points_count)

            sample = curve_processing.normalize_curve(curve=sample)
            input.append(sample)
            dist_index = dist_index + 1

        flipped_anchor = numpy.flip(m=input[0], axis=0).copy()
        sample = curve_processing.normalize_curve(curve=flipped_anchor)
        input.append(sample)

        for i in range(9):
            while True:
                center_point_index_offset = int(numpy.random.randint(offset_length, size=1)) - int(offset_length/2)
                if center_point_index_offset != 0:
                    break

            transform = cls._generate_curve_transform()
            transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)

            indices_pool = discrete_distribution.sample_discrete_dist(dist=discrete_distribution_pack[dist_index], sampling_points_count=sampling_points_count)
            sample = curve_sampling.sample_curve_neighborhood(
                curve=transformed_curve,
                center_point_index=numpy.mod(center_point_index + center_point_index_offset, transformed_curve.shape[0]),
                indices_pool=indices_pool,
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


# class ArcLengthTupletsDatasetGenerator(TuplesDatasetGenerator):
#     _file_name = 'tuplets'
#     _label = 'tuplets'
#
#     @classmethod
#     def generate_tuple(cls, curves, sampling_ratio, multimodality, section_points_count):
#         factors = []
#         input1 = []
#         input2 = []
#         input3 = []
#         input4 = []
#
#         curve_index = int(numpy.random.randint(curves.shape[0], size=1)),
#         curve = curves[curve_index]
#
#         tuplet = {
#             'input1': input1,
#             'input2': input2,
#             'input3': input3,
#             'input4': input4,
#             'factors': factors
#         }
#
#         dist_index = 0
#         point_type = 'start'
#         point_index = int(numpy.random.randint(curve.shape[0], size=1))
#         transform = cls._generate_curve_transform()
#         curve = curve_processing.center_curve(curve=curves[curve_index])
#         transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
#
#         curve_points_count = curve.shape[0]
#         sampling_points_count = int(sampling_ratio * curve_points_count)
#         discrete_distribution_pack = discrete_distribution.random_discrete_dist(bins=curve_points_count, multimodality=multimodality, max_density=1, count=1)
#         indices_pool = discrete_distribution.sample_discrete_dist(dist=discrete_distribution_pack[dist_index], sampling_points_count=sampling_points_count)
#
#         for offset in range(section_points_count):
#             curve_indices1, curve_indices2 = curve_sampling.sample_overlapping_curve_sections_indices(
#                 point_index=point_index,
#                 point_type=point_type,
#                 indices_pool=indices_pool,
#                 section_points_count=section_points_count,
#                 offset=offset)
#
#             input1.append(curve_processing.normalize_curve(curve=curve[curve_indices1]))
#             input2.append(curve_processing.normalize_curve(curve=transformed_curve[curve_indices1]))
#
#             if offset < section_points_count - 1:
#                 input3.append(curve_processing.normalize_curve(curve=curve[curve_indices2]))
#                 input4.append(curve_processing.normalize_curve(curve=transformed_curve[curve_indices2]))
#
#         return tuplet
#
#     @staticmethod
#     def _generate_zip_data(items_count, exact_examples_count, inexact_examples_count, section_points_count, min_perturbation, max_perturbation, min_offset, max_offset=None):
#         exact_examples_count_pack = [exact_examples_count] * items_count
#         inexact_examples_count_pack = [inexact_examples_count] * items_count
#         section_points_count_pack = [section_points_count] * items_count
#         min_offset_pack = [min_offset] * items_count
#         max_offset_pack = [max_offset] * items_count
#         min_perturbation_pack = [min_perturbation] * items_count
#         max_perturbation_pack = [max_perturbation] * items_count
#         data = [exact_examples_count_pack, inexact_examples_count_pack, section_points_count_pack, min_perturbation_pack, max_perturbation_pack, min_offset_pack, max_offset_pack]
#         names = ['exact_examples_count', 'inexact_examples_count', 'section_points_count', 'min_perturbation', 'max_perturbation', 'min_offset', 'max_offset']
#         return names, data
#
#
# class EuclideanArclengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, EuclideanTransform):
#     pass
#
#
# class EquiaffineArclengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, EquiaffineTransform):
#     pass
#
#
# class AffineArclengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, AffineTransform):
#     pass


class ArcLengthTupletsDatasetGenerator(TuplesDatasetGenerator):
    _file_name = 'tuplets'
    _label = 'tuplets'

    @staticmethod
    def _sample_curve_section(curve, supporting_points_count, start_point_index, end_point_index):
        sample = curve_sampling.sample_curve_section_old(
            curve=curve,
            # dist=dist,
            supporting_points_count=supporting_points_count,
            start_point_index=start_point_index,
            end_point_index=end_point_index)

        # flip_sample = bool(random.getrandbits(1))
        # if flip_sample is True:
        #     sample = numpy.flip(m=sample, axis=0).copy()

        sample = curve_processing.normalize_curve(curve=sample)
        return sample

    @classmethod
    def generate_tuple(cls, curves, multimodality, supporting_points_count, min_offset, max_offset):
        input = []
        factors = []

        curve_index = int(numpy.random.randint(curves.shape[0], size=1))
        curve = curves[curve_index]

        center_point_index = int(numpy.random.randint(curve.shape[0], size=1))

        tuplet = {
            'input': input,
            'factors': factors
        }

        # discrete_distribution_pack = discrete_distribution.random_discrete_dist(bins=curve.shape[0], multimodality=multimodality, max_density=1, count=10)

        raw_offset = numpy.random.randint(max_offset, size=4)
        offset = numpy.maximum(raw_offset, [min_offset] * 4)
        index1 = center_point_index
        index2 = int(numpy.mod(index1 + offset[0], curve.shape[0]))
        index3 = int(numpy.mod(index2 + offset[1], curve.shape[0]))
        index4 = int(numpy.mod(index3 + offset[2], curve.shape[0]))
        index5 = int(numpy.mod(index4 + offset[3], curve.shape[0]))

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
            # dist=discrete_distribution_pack[0],
            supporting_points_count=supporting_points_count,
            start_point_index=index1,
            end_point_index=index3)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            # dist=discrete_distribution_pack[1],
            supporting_points_count=supporting_points_count,
            start_point_index=index2,
            end_point_index=index4)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            # dist=discrete_distribution_pack[2],
            supporting_points_count=supporting_points_count,
            start_point_index=index3,
            end_point_index=index5)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            # dist=discrete_distribution_pack[3],
            supporting_points_count=supporting_points_count,
            start_point_index=index1,
            end_point_index=index4)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            # dist=discrete_distribution_pack[4],
            supporting_points_count=supporting_points_count,
            start_point_index=index2,
            end_point_index=index5)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve1,
            # dist=discrete_distribution_pack[5],
            supporting_points_count=supporting_points_count,
            start_point_index=index1,
            end_point_index=index5)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve2,
            # dist=discrete_distribution_pack[6],
            supporting_points_count=supporting_points_count,
            start_point_index=index1,
            end_point_index=index2)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve2,
            # dist=discrete_distribution_pack[7],
            supporting_points_count=supporting_points_count,
            start_point_index=index2,
            end_point_index=index3)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve2,
            # dist=discrete_distribution_pack[8],
            supporting_points_count=supporting_points_count,
            start_point_index=index3,
            end_point_index=index4)
        input.append(sample)

        sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
            curve=curve2,
            # dist=discrete_distribution_pack[9],
            supporting_points_count=supporting_points_count,
            start_point_index=index4,
            end_point_index=index5)
        input.append(sample)

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


class AffineArcLengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, AffineTransform):
    pass