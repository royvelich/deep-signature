import numpy


def sample_curve(curve, supporting_point_count, center_point_index=None, max_offset=None):
    rng = numpy.random.default_rng()
    curve_points_count = curve.shape[0]

    if max_offset is None:
        max_offset = curve_points_count - 1

    if center_point_index is None:
        center_point_index = int(numpy.random.randint(low=0, high=curve_points_count - 1))

    indices_pool_low = numpy.arange(start=center_point_index - max_offset, stop=center_point_index)
    indices_pool_high = numpy.arange(start=center_point_index + 1, stop=center_point_index + max_offset + 1)

    indices_low = numpy.mod(numpy.sort(rng.choice(a=indices_pool_low, size=supporting_point_count, replace=False)), curve_points_count)
    indices_high = numpy.mod(numpy.sort(rng.choice(a=indices_pool_high, size=supporting_point_count, replace=False)), curve_points_count)
    indices = numpy.concatenate((indices_low, numpy.array([center_point_index]), indices_high))
    return curve[indices]


# def sample_indices(curve, starting_index, indices_pool, count, increment_delta):
#     sampled_indices = []
#     i = starting_index
#     while True:
#         if len(sampled_indices) == count:
#             break
#         i_mod = numpy.mod(i, curve.shape[0])
#         if i_mod in indices_pool:
#             sampled_indices.append(i_mod)
#         i += increment_delta
#
#     return sampled_indices


# def sample_supporting_point_indices(curve, center_point_index, indices_pool, supporting_point_count):
#     right_supporting_points_indices = sample_indices(
#         curve=curve,
#         starting_index=center_point_index+1,
#         indices_pool=indices_pool,
#         count=supporting_point_count,
#         increment_delta=1)
#
#     left_supporting_points_indices = sample_indices(
#         curve=curve,
#         starting_index=center_point_index-1,
#         indices_pool=indices_pool,
#         count=supporting_point_count,
#         increment_delta=-1)
#
#     supporting_points_indices = numpy.concatenate((left_supporting_points_indices, right_supporting_points_indices))
#
#     return supporting_points_indices


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
