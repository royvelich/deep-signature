import os
import scipy.io
import numpy
import re
import scipy.stats as ss
from scipy.stats import truncnorm
import math


class DistGenerator:

    # https://stackoverflow.com/questions/37411633/how-to-generate-a-random-normal-distribution-of-integers
    @staticmethod
    def generate_normal_dist(bins, loc, scale):
        is_even = bins % 2 == 0
        half_bins = math.floor(bins / 2)

        start, stop = -half_bins, half_bins
        if not is_even:
            stop = stop + 1

        x = numpy.arange(start, stop) + half_bins
        x_lower, x_upper = x - 0.5, x + 0.5

        cfd_lower = ss.norm.cdf(x_lower, loc=loc, scale=scale)
        cfd_upper = ss.norm.cdf(x_upper, loc=loc, scale=scale)
        dist = cfd_upper - cfd_lower

        # normalize the distribution bins so their sum is 1
        dist = dist / dist.sum()

        return dist

    @staticmethod
    def generate_random_normal_dist(bins):
        mean = bins / 2

        truncated_normal = DistGenerator.generate_truncated_normal(mean=0.5, sd=0.25, low=0, upp=1)
        var = truncated_normal.rvs(1) * 10*numpy.sqrt(bins)
        # params = numpy.random.rand(2)
        # params = params * numpy.array([bins, bins/3])
        dist = DistGenerator.generate_normal_dist(bins, mean, var)

        return numpy.roll(dist, int(numpy.round(numpy.random.rand(1) * bins)))

    @staticmethod
    def generate_random_dist(bins):
        truncated_normal = DistGenerator.generate_truncated_normal(mean=17, sd=4, low=2, upp=30)
        count = 15

        dist = None
        for _ in range(count):
            current_dist = DistGenerator.generate_random_normal_dist(bins)
            if dist is None:
                dist = current_dist
            else:
                dist = dist + current_dist

        dist = dist / dist.sum()
        return dist

    # https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
    @staticmethod
    def generate_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
