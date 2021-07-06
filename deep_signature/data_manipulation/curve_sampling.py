import numpy

from deep_signature.stats import discrete_distribution


def sample_curve_point_neighborhood_indices(curve, supporting_points_count, center_point_index=None, max_offset=None):
    rng = numpy.random.default_rng()
    curve_points_count = curve.shape[0]

    if max_offset is None:
        max_offset = curve_points_count - 1

    if center_point_index is None:
        center_point_index = int(numpy.random.randint(low=0, high=curve_points_count - 1))

    indices_pool_low = numpy.arange(start=center_point_index - max_offset, stop=center_point_index)
    indices_pool_high = numpy.arange(start=center_point_index + 1, stop=center_point_index + max_offset + 1)

    indices_low = numpy.mod(numpy.sort(rng.choice(a=indices_pool_low, size=supporting_points_count, replace=False)), curve_points_count)
    indices_high = numpy.mod(numpy.sort(rng.choice(a=indices_pool_high, size=supporting_points_count, replace=False)), curve_points_count)
    indices = numpy.concatenate((indices_low, numpy.array([center_point_index]), indices_high))
    return indices


def sample_curve_point_neighborhood_indices_from_curve_sections(curve, section1_indices, section2_indices, supporting_points_count, max_offset=None):
    rng = numpy.random.default_rng()
    indices = numpy.concatenate((section1_indices, section2_indices[1:]))
    center_point_index = section1_indices.shape[0] - 1
    max_offset = supporting_points_count

    indices_pool_low = numpy.arange(start=center_point_index - max_offset, stop=center_point_index)
    indices_pool_high = numpy.arange(start=center_point_index + 1, stop=center_point_index + max_offset + 1)

    indices_low = numpy.sort(rng.choice(a=indices_pool_low, size=supporting_points_count, replace=False))
    indices_high = numpy.sort(rng.choice(a=indices_pool_high, size=supporting_points_count, replace=False))
    sampled_indices = numpy.concatenate((indices_low, numpy.array([center_point_index]), indices_high))
    return indices[sampled_indices]


def sample_curve_point_neighborhood(curve, supporting_points_count, center_point_index=None, max_offset=None):
    indices = sample_curve_point_neighborhood_indices(
        curve=curve,
        supporting_points_count=supporting_points_count,
        center_point_index=center_point_index,
        max_offset=max_offset)
        
    return curve[indices]


def sample_curve_point_neighborhood_from_curve_sections(curve, section1_indices, section2_indices, supporting_points_count, max_offset=None):
    indices = sample_curve_point_neighborhood_indices_from_curve_sections(
        curve=curve,
        section1_indices=section1_indices,
        section2_indices=section2_indices,
        supporting_points_count=supporting_points_count,
        max_offset=max_offset)

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
    indices = numpy.concatenate((numpy.mod(numpy.array([indices_pool[0]]), curve_points_count), indices, numpy.mod(numpy.array([indices_pool[-1]]), curve_points_count)))

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
    indices = numpy.concatenate((numpy.mod(numpy.array([indices_pool[0]]), curve_points_count), indices, numpy.mod(numpy.array([indices_pool[-1]]), curve_points_count)))

    return curve[indices]


def sample_curve_section_indices_with_dist(curve, center_point_index, dist, sampling_points_count, supporting_points_count):
    def concatenate_indices1(index):
        indices_pool_low = numpy.arange(start=index - supporting_points_count, stop=index)
        indices_pool_high = numpy.arange(start=index + 1, stop=index + supporting_points_count + 1)
        return indices_pool_low, indices_pool_high

    def concatenate_indices2(index1, index2):
        indices_pool_low = numpy.arange(start=index1 - supporting_points_count + 1, stop=index1 + 1)
        indices_pool_high = numpy.arange(start=index2, stop=index2 + supporting_points_count)
        return indices_pool_low, indices_pool_high

    sampled_indices = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=sampling_points_count)
    sampled_indices_count = sampled_indices.shape[0]
    found = False
    for i in range(sampled_indices_count):
        index1 = sampled_indices[numpy.mod(i - 1, sampled_indices_count)]
        index2 = sampled_indices[numpy.mod(i, sampled_indices_count)]
        if index1 <= center_point_index <= index2:
            if center_point_index == index1:
                indices_pool_low, indices_pool_high = concatenate_indices1(index=i-1)
                found = True
            elif center_point_index == index2:
                indices_pool_low, indices_pool_high = concatenate_indices1(index=i)
                found = True
            else:
                indices_pool_low, indices_pool_high = concatenate_indices2(index1=i-1, index2=i)
                found = True
            break

    if found is False:
        index1 = sampled_indices[-1]
        index2 = 0
        if center_point_index == index1:
            indices_pool_low, indices_pool_high = concatenate_indices1(index=sampled_indices_count-1)
        elif center_point_index == index2:
            indices_pool_low, indices_pool_high = concatenate_indices1(index=0)
        else:
            indices_pool_low, indices_pool_high = concatenate_indices2(index1=sampled_indices_count-1, index2=0)

    indices_low = sampled_indices[numpy.mod(indices_pool_low, sampled_indices_count)]
    indices_high = sampled_indices[numpy.mod(indices_pool_high, sampled_indices_count)]
    indices = numpy.concatenate((indices_low, [center_point_index], indices_high))

    print(indices)

    return indices


def sample_curve_section_with_dist(curve, center_point_index, dist, sampling_points_count, supporting_points_count):
    indices = sample_curve_section_indices_with_dist(
        curve=curve,
        center_point_index=center_point_index,
        dist=dist,
        sampling_points_count=sampling_points_count,
        supporting_points_count=supporting_points_count)

    return curve[indices]
