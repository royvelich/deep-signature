# numpy
import numpy

# deep signature
from deep_signature.stats import discrete_distribution

# utils
from utils import common as common_utils


def sample_curve_neighborhood_indices(center_point_index, indices_pool, supporting_points_count):
    modified_indices_pool = common_utils.insert_sorted(indices_pool, numpy.array([center_point_index]))
    meta_index = numpy.where(modified_indices_pool == center_point_index)[0]
    meta_indices = numpy.arange(start=meta_index - supporting_points_count, stop=meta_index + supporting_points_count + 1)
    indices = modified_indices_pool[numpy.mod(meta_indices, modified_indices_pool.shape[0])]
    return indices


def sample_curve_neighborhood(curve, center_point_index, indices_pool, supporting_points_count):
    indices = sample_curve_neighborhood_indices(
        center_point_index=center_point_index,
        indices_pool=indices_pool,
        supporting_points_count=supporting_points_count)

    return curve[indices]


# def sample_curve_section_indices(point_index, point_type, indices_pool, section_points_count, offset):
#     modified_indices_pool = utils.insert_sorted(indices_pool, numpy.array([point_index]))
#     meta_index = numpy.where(modified_indices_pool == point_index)[0] + offset
#
#     meta_indices = None
#     if point_type == 'start':
#         meta_indices = numpy.arange(start=meta_index, stop=meta_index + section_points_count)
#     elif point_type == 'end':
#         meta_indices = numpy.arange(start=meta_index - section_points_count + 1, stop=meta_index + 1)
#
#     indices = modified_indices_pool[numpy.mod(meta_indices, modified_indices_pool.shape[0])]
#     return indices
#
#
# def sample_curve_section(curve, point_index, point_type, indices_pool, section_points_count, offset):
#     indices = sample_curve_section_indices(
#         point_index=point_index,
#         point_type=point_type,
#         indices_pool=indices_pool,
#         section_points_count=section_points_count,
#         offset=offset)
#
#     return curve[indices]
#
#
# def sample_overlapping_curve_sections_indices(point_index, point_type, indices_pool, section_points_count, offset=0):
#     indices2 = sample_curve_section_indices(point_index=point_index, point_type=point_type, indices_pool=indices_pool, section_points_count=section_points_count + 1, offset=offset)
#     indices1 = indices2[:-1]
#
#     return indices1, indices2
#
#
# def sample_overlapping_curve_sections(curve, point_index, point_type, indices_pool, section_points_count, offset=0):
#     indices1, indices2 = sample_overlapping_curve_sections_indices(point_index=point_index, point_type=point_type, indices_pool=indices_pool, section_points_count=section_points_count, offset=offset)
#
#     return curve[indices1], curve[indices2]


def sample_curve_section_indices(curve, start_point_index, end_point_index, multimodality, supporting_points_count=None, uniform=False, rng=None):
    if rng is None:
        rng = numpy.random.default_rng()

    curve_points_count = curve.shape[0]

    if start_point_index > end_point_index:
        start_point_index = -(curve_points_count - start_point_index)
        # end_point_index = -end_point_index

    indices_pool = numpy.mod(numpy.array(list(range(start_point_index + 1, end_point_index))), curve.shape[0])
    bins = int(numpy.abs(end_point_index - start_point_index) + 1) - 2
    supporting_points_count = supporting_points_count - 2 if supporting_points_count is not None else bins

    # uniform = False
    if uniform is False:
        dist = discrete_distribution.random_discrete_dist(bins=bins, multimodality=multimodality, max_density=1, count=1)[0]
        meta_indices = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=supporting_points_count)
    else:
        meta_indices = list(range(end_point_index-start_point_index-1))
        meta_indices = numpy.sort(rng.choice(a=meta_indices, size=supporting_points_count, replace=False))

    indices = indices_pool[meta_indices]
    indices = numpy.concatenate(([numpy.mod(start_point_index, curve.shape[0])], indices, [numpy.mod(end_point_index, curve.shape[0])]))

    return indices


def sample_curve_section(curve, start_point_index, end_point_index, multimodality, supporting_points_count, uniform):
    indices = sample_curve_section_indices(
        curve=curve,
        start_point_index=start_point_index,
        end_point_index=end_point_index,
        multimodality=multimodality,
        supporting_points_count=supporting_points_count,
        uniform=uniform)
    return curve[indices]
