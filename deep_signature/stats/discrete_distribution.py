# numpy
import numpy

# scipy
import scipy.io
import scipy.signal
import scipy.stats

# skimage
from skimage.util.shape import view_as_windows


# https://stackoverflow.com/questions/37411633/how-to-generate-a-random-normal-distribution-of-integers
def normal_discrete_dist(bins, loc, scale):
    is_even = bins % 2 == 0
    half_bins = numpy.floor(bins / 2)

    start, stop = -half_bins, half_bins
    if not is_even:
        stop = stop + 1

    x = numpy.arange(start, stop)
    x_lower, x_upper = x - 0.5, x + 0.5

    cfd_lower = scipy.stats.norm.cdf(x_lower[:, None], loc=loc, scale=scale).transpose()
    cfd_upper = scipy.stats.norm.cdf(x_upper[:, None], loc=loc, scale=scale).transpose()
    dist = cfd_upper - cfd_lower
    dist = dist / dist.sum()

    return dist


def random_normal_discrete_dist(bins, count=1):

    # https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    def strided_indexing_roll(a, r):
        # Concatenate with sliced to cover all rolls
        a_ext = numpy.concatenate((a, a[:, :-1]), axis=1)

        # Get sliding windows; use advanced-indexing to select appropriate ones
        n = a.shape[1]
        return view_as_windows(a_ext, (1, n))[numpy.arange(len(r)), (n - r) % n, 0]

    truncnorm_loc = 0.5
    truncnorm_scale = 0.5
    truncnorm_a = 0
    truncnorm_b = 1

    # https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
    truncated_normal = scipy.stats.truncnorm(
        a=(truncnorm_a - truncnorm_loc) / truncnorm_scale,
        b=(truncnorm_b - truncnorm_loc) / truncnorm_scale,
        loc=truncnorm_loc,
        scale=truncnorm_scale)

    # scale = numpy.maximum(truncated_normal.rvs(count) * bins * 0.3, bins * 0.03)
    scale = truncated_normal.rvs(count) * bins * 0.1
    loc = [0] * count
    dist = normal_discrete_dist(bins, loc, scale)

    dist_sum = dist.sum(axis=1)
    dist = dist / dist_sum[:, None]

    return strided_indexing_roll(a=dist, r=numpy.round(numpy.random.rand(count) * bins).astype(int))


def random_discrete_dist(bins, multimodality, max_density, count=1):
    valid_dists = None
    while True:
        # Generate "count * multimodality" single modal normal distribution with k bins (when k = "bins")
        dists = random_normal_discrete_dist(bins=bins, count=count * multimodality)

        # Reshape the tensor so every k single modal distributions will be packed together (when k = "multimodality")
        dists = dists.reshape(count, multimodality, bins)

        # Sum each pack of k distributions into a single multimodal distribution (when k = "multimodality")
        dists = dists.sum(axis=1)

        # Normalize each multimodal distribution by dividing its entries by its sum
        dist_sum = dists.sum(axis=1)
        dists = dists / dist_sum[:, None]

        dists_max_density = numpy.max(dists, axis=1)
        # dists_min_density = numpy.min(dists, axis=1)
        # dists_ratio_density = (dists_max_density - dists_min_density) / max_density
        # dists_validity_flags = numpy.logical_and(dists_max_density <= max_density, dists_ratio_density > 0.1)
        dists_validity_flags = dists_max_density <= max_density
        current_valid_dists = dists[numpy.where(dists_validity_flags == True)]
        if valid_dists is None:
            valid_dists = current_valid_dists
        else:
            valid_dists = numpy.vstack((valid_dists, current_valid_dists))

        if valid_dists.shape[0] >= count:
            break

    valid_dists = valid_dists[range(count)]

    # Return result
    return valid_dists


def sample_discrete_dist(dist, sampling_points_count):
    density_threshold = 1.0 / sampling_points_count
    accumulated_density = 0
    sampled_indices = []
    missed_indices = []
    for i in range(len(dist)):
        accumulated_density = accumulated_density + dist[i]
        if accumulated_density >= density_threshold:
            sampled_indices.append(i)
            accumulated_density = accumulated_density - density_threshold
        else:
            missed_indices.append({
                'index': i,
                'delta': density_threshold - accumulated_density
            })

    missing_indices_count = sampling_points_count - len(sampled_indices)
    if missing_indices_count > 0:
        missed_indices = sorted(missed_indices, key=lambda x: x['delta'], reverse=True)
        for i in range(missing_indices_count):
            sampled_indices.append(missed_indices[i]['index'])

    sampled_indices.sort()

    return numpy.array(sampled_indices)