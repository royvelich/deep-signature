import numpy


def sample_indices(curve, starting_index, indices_pool, count, increment_delta):
    sampled_indices = []
    i = starting_index
    while True:
        if len(sampled_indices) == count:
            break
        i_mod = numpy.mod(i, curve.shape[0])
        if i_mod in indices_pool:
            sampled_indices.append(i_mod)
        i += increment_delta

    return sampled_indices


def sample_supporting_point_indices(curve, center_point_index, indices_pool, supporting_point_count):
    right_supporting_points_indices = sample_indices(
        curve=curve,
        starting_index=center_point_index+1,
        indices_pool=indices_pool,
        count=supporting_point_count,
        increment_delta=1)

    left_supporting_points_indices = sample_indices(
        curve=curve,
        starting_index=center_point_index-1,
        indices_pool=indices_pool,
        count=supporting_point_count,
        increment_delta=-1)

    supporting_points_indices = numpy.concatenate((left_supporting_points_indices, right_supporting_points_indices))

    return supporting_points_indices


# def sample_supporting_points_indices(curve, center_point_index, indices_pool, supporting_points_count):
#     right_supporting_points_indices = sample_indices(
#         curve=curve,
#         starting_index=center_point_index+1,
#         indices_pool=indices_pool,
#         count=supporting_points_count,
#         increment_delta=1)
#
#     left_supporting_points_indices = sample_indices(
#         curve=curve,
#         starting_index=center_point_index-1,
#         indices_pool=indices_pool,
#         count=supporting_points_count,
#         increment_delta=-1)
#
#     supporting_points_indices = numpy.concatenate((left_supporting_points_indices, numpy.array([center_point_index]), right_supporting_points_indices))
#
#     return supporting_points_indices
