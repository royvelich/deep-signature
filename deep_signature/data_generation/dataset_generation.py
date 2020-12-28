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


class CurveManager:
    def __init__(self, curve):
        self._curve = curve_processing.translate_curve(curve=curve, offset=-numpy.mean(curve, axis=0))
        self._curve_points_count = self._curve.shape[0]
        self._curvature = curve_processing.calculate_curvature(self._curve)

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

    @staticmethod
    def _generate_tuples():
        raise NotImplemented

    @staticmethod
    def _zip_iterable():
        raise NotImplemented

# class PairsDatasetGenerator(TuplesDatasetGenerator):
#     @staticmethod
#     def _zip_iterable(curves, sections_per_curve, pairs_per_section):
#         center_point_indices_pack = []
#         curves_pack = []
#         for curve in curves:
#             center_point_indices = numpy.linspace(
#                 start=0,
#                 stop=curve.shape[0],
#                 num=sections_per_curve,
#                 endpoint=False,
#                 dtype=int)
#
#             center_point_indices_pack.extend([center_point_indices] * pairs_per_section)
#             curves_pack.extend([curve] * sections_per_curve * pairs_per_section)
#
#         entry_names = ['curve', 'center_point_index']
#         iterable = [dict(zip(entry_names, values)) for values in zip(curves_pack, center_point_indices_pack)]
#         return iterable
#
#
# class PositiveSectionPairsDatasetGenerator(PairsDatasetGenerator):
#     _file_name = 'positive_pairs'
#     _label = 'positive pairs'
#
#     @staticmethod
#     def _generate_tuple():
#         raise NotImplemented
#
#
# class NegativeSectionPairsDatasetGenerator(PairsDatasetGenerator):
#     _file_name = 'negative_pairs'
#     _label = 'negative pairs'
#
#     @staticmethod
#     def _generate_tuple():
#         raise NotImplemented


class TupletsDatasetGenerator(TuplesDatasetGenerator):
    _file_name = 'tuplets'
    _label = 'tuplets'

    @staticmethod
    def _generate_tuple(curves, curve_index, center_point_index, negative_examples_count, supporting_points_count, max_offset):
        tuplet = []
        curve = curves[curve_index]
        for _ in range(2):
            sample = curve_sampling.sample_curve(
                curve=curve,
                center_point_index=center_point_index,
                supporting_point_count=supporting_points_count,
                max_offset=max_offset)
            sample = curve_processing.normalize_curve(curve=sample)
            tuplet.append(sample)

        # rng = numpy.random.default_rng()
        # indices_pool = numpy.arange(start=0, stop=curve.shape[0])
        # indices_pool = numpy.delete(indices_pool, center_point_index)
        # indices = rng.choice(a=indices_pool, size=negative_examples_count, replace=False)
        # for index in indices:
        #     sample = curve_sampling.sample_curve(
        #         curve=curve,
        #         center_point_index=index,
        #         supporting_point_count=supporting_points_count,
        #         max_offset=max_offset)
        #     sample = curve_processing.normalize_curve(curve=sample)
        #     tuplet.append(sample)

        rng = numpy.random.default_rng()
        indices_pool = numpy.arange(start=0, stop=len(curves))
        indices_pool = numpy.delete(indices_pool, curve_index)
        indices = rng.choice(a=indices_pool, size=negative_examples_count, replace=False)
        for index in indices:
            current_curve = curves[index]
            sample = curve_sampling.sample_curve(
                curve=current_curve,
                center_point_index=int(numpy.random.randint(current_curve.shape[0])),
                supporting_point_count=supporting_points_count,
                max_offset=max_offset)
            sample = curve_processing.normalize_curve(curve=sample)
            tuplet.append(sample)

        return tuplet

    @staticmethod
    def _zip_iterable(curves, sections_density, negative_examples_count, supporting_points_count, max_offset=None):
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
        negative_examples_count_pack = [negative_examples_count] * items_count
        supporting_points_count_pack = [supporting_points_count] * items_count
        max_offset_pack = [max_offset] * items_count
        zipped_data = zip(curves_pack, curve_indices_pack, center_point_indices_pack, negative_examples_count_pack, supporting_points_count_pack, max_offset_pack)
        entry_names = ['curves', 'curve_index', 'center_point_index', 'negative_examples_count', 'supporting_points_count', 'max_offset']
        iterable = [dict(zip(entry_names, values)) for values in zipped_data]
        return iterable
