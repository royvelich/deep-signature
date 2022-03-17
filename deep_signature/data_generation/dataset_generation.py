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
from deep_signature.linalg import transformations
from utils import settings


class TuplesDatasetGenerator:
    pass


class EuclideanTransform:
    @staticmethod
    def _generate_curve_transform():
        return transformations.generate_random_transform_2d_training(transform_type='euclidean')


class SimilarityTransform:
    @staticmethod
    def _generate_curve_transform():
        return transformations.generate_random_transform_2d_training(transform_type='similarity')


class EquiaffineTransform:
    @staticmethod
    def _generate_curve_transform():
        return transformations.generate_random_transform_2d_training(transform_type='equiaffine')


class AffineTransform:
    @staticmethod
    def _generate_curve_transform():
        return transformations.generate_random_transform_2d_training(transform_type='affine')


class CurvatureTupletsDatasetGenerator(TuplesDatasetGenerator):
    @classmethod
    def generate_tuple(cls, curves, sampling_ratio, multimodality, supporting_points_count, offset_length, negative_examples_count):
        input = []
        tuplet = {
            'input': input
        }

        dist_index = 0
        curve_index = int(numpy.random.randint(curves.shape[0], size=1))
        curve = curve_processing.center_curve(curve=curves[curve_index])
        curve_points_count = curve.shape[0]
        sampling_points_count = int(sampling_ratio * curve_points_count)
        discrete_distribution_pack = discrete_distribution.random_discrete_dist(bins=curve_points_count, multimodality=multimodality, max_density=1, count=negative_examples_count+2)
        center_point_index = int(numpy.random.randint(curve.shape[0], size=1))
        for i in range(2):
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

        for i in range(negative_examples_count):
            while True:
                center_point_index_offset = int(numpy.random.randint(offset_length, size=1)) - int(offset_length/2)
                if center_point_index_offset != 0:
                    break

            transform = cls._generate_curve_transform()
            transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)

            negative_center_point_index = numpy.mod(center_point_index + center_point_index_offset, transformed_curve.shape[0])
            # negative_center_point_index = int(numpy.random.randint(transformed_curve.shape[0], size=1))
            indices_pool = discrete_distribution.sample_discrete_dist(dist=discrete_distribution_pack[dist_index], sampling_points_count=sampling_points_count)
            sample = curve_sampling.sample_curve_neighborhood(
                curve=transformed_curve,
                center_point_index=negative_center_point_index,
                indices_pool=indices_pool,
                supporting_points_count=supporting_points_count)

            sample = curve_processing.normalize_curve(curve=sample)
            input.append(sample)
            dist_index = dist_index + 1

        return tuplet


class EuclideanCurvatureTupletsDatasetGenerator(CurvatureTupletsDatasetGenerator, EuclideanTransform):
    pass


class SimilarityCurvatureTupletsDatasetGenerator(CurvatureTupletsDatasetGenerator, SimilarityTransform):
    pass


class EquiaffineCurvatureTupletsDatasetGenerator(CurvatureTupletsDatasetGenerator, EquiaffineTransform):
    pass


class AffineCurvatureTupletsDatasetGenerator(CurvatureTupletsDatasetGenerator, AffineTransform):
    pass


class ArcLengthTupletsDatasetGenerator(TuplesDatasetGenerator):
    @staticmethod
    def _sample_curve_section(curve, start_point_index, end_point_index, multimodality, supporting_points_count):
        sample = curve_sampling.sample_curve_section(
            curve=curve,
            start_point_index=start_point_index,
            end_point_index=end_point_index,
            multimodality=multimodality,
            supporting_points_count=supporting_points_count,
            uniform=True)

        sample = curve_processing.normalize_curve(curve=sample)
        return sample

    @classmethod
    def generate_tuple(cls, curves, min_offset, max_offset, multimodality, supporting_points_count, anchor_points_count):
        tuplets = []

        curve_index = int(numpy.random.randint(curves.shape[0], size=1))
        curve = curve_processing.center_curve(curve=curves[curve_index])

        starting_point_index = int(numpy.random.randint(curve.shape[0], size=1))

        # anchor_points_count = 5
        raw_offset = numpy.random.randint(max_offset, size=anchor_points_count-1)
        offset = numpy.maximum(raw_offset, [min_offset] * (anchor_points_count-1))
        indices = [starting_point_index]
        for i in range(anchor_points_count - 1):
            indices.append(int(numpy.mod(indices[i] + offset[i], curve.shape[0])))

        transform1 = cls._generate_curve_transform()
        transform2 = cls._generate_curve_transform()
        transformed_curve1 = curve_processing.transform_curve(curve=curve, transform=transform1)
        transformed_curve2 = curve_processing.transform_curve(curve=curve, transform=transform2)

        for i in range(2, anchor_points_count):
            for index1, index2 in zip(indices, indices[i:]):
                sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
                    curve=transformed_curve1,
                    start_point_index=index1,
                    end_point_index=index2,
                    multimodality=multimodality,
                    supporting_points_count=supporting_points_count)
                tuplets.append(sample)

        for index1, index2 in zip(indices, indices[1:]):
            sample = ArcLengthTupletsDatasetGenerator._sample_curve_section(
                curve=transformed_curve2,
                start_point_index=index1,
                end_point_index=index2,
                multimodality=multimodality,
                supporting_points_count=supporting_points_count)
            tuplets.append(sample)

        return tuplets


class EuclideanArcLengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, EuclideanTransform):
    pass


class SimilarityArcLengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, SimilarityTransform):
    pass


class EquiaffineArcLengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, EquiaffineTransform):
    pass


class AffineArcLengthTupletsDatasetGenerator(ArcLengthTupletsDatasetGenerator, AffineTransform):
    pass