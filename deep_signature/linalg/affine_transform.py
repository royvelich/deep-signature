# numpy
import numpy


def random_equiaffine_transform_2d(max_scale=2):
    scale = numpy.random.uniform(low=1, high=max_scale, size=2)
    coeffs = numpy.random.random(size=2)
    entries = scale * coeffs
    L = numpy.array([[1, 0], [entries[0], 1]])
    U = numpy.array([[1, entries[1]], [0, 1]])
    return numpy.matmul(L, U)
