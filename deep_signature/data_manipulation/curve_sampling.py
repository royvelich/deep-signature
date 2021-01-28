import numpy


def sample_curve_point_neighbourhood(curve, supporting_point_count, center_point_index=None, max_offset=None):
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


def sample_curve_section_indices(curve, supporting_points_count, start_point_index, end_point_index):
    rng = numpy.random.default_rng()
    curve_points_count = curve.shape[0]

    if start_point_index > end_point_index:
        start_point_index = -(curve_points_count - start_point_index)
        # end_point_index = -end_point_index

    indices_pool = numpy.linspace(
        start=start_point_index,
        stop=end_point_index,
        num=int(numpy.abs(end_point_index - start_point_index) + 1),
        endpoint=True,
        dtype=int)

    inner_indices_pool = indices_pool[1:-1]
    indices = numpy.mod(numpy.sort(rng.choice(a=inner_indices_pool, size=supporting_points_count - 2, replace=False)), curve_points_count)
    indices = numpy.concatenate((
        numpy.mod(numpy.array([indices_pool[0]]), curve_points_count),
        indices,
        numpy.mod(numpy.array([indices_pool[-1]]), curve_points_count)))

    return indices


def sample_curve_section(curve, supporting_points_count, start_point_index, end_point_index):
    indices = sample_curve_section_indices(
        curve=curve,
        supporting_points_count=supporting_points_count,
        start_point_index=start_point_index,
        end_point_index=end_point_index)
    return curve[indices]


def sample_curve_section2(curve, supporting_points_count, start_point_index, end_point_index):
    rng = numpy.random.default_rng()
    curve_points_count = curve.shape[0]

    if start_point_index > end_point_index:
        end_point_index = -end_point_index

    indices_pool = numpy.linspace(
        start=start_point_index,
        stop=end_point_index,
        num=int(numpy.abs(end_point_index - start_point_index) + 1),
        endpoint=True,
        dtype=int)

    inner_indices_pool = indices_pool[1:-1]
    indices = numpy.mod(numpy.sort(rng.choice(a=inner_indices_pool, size=supporting_points_count - 2, replace=False)), curve_points_count)
    indices = numpy.concatenate((
        numpy.mod(numpy.array([indices_pool[0]]), curve_points_count),
        indices,
        numpy.mod(numpy.array([indices_pool[-1]]), curve_points_count)))

    # print(indices)

    return curve[indices]