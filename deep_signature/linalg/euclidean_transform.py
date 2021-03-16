# numpy
import numpy


def rotation_2d(radians):
    c, s = numpy.cos(radians), numpy.sin(radians)
    return numpy.array([[c, s], [-s, c]])


def horizontal_reflection_2d():
    return numpy.array([[1, 0], [0, -1]])


def vertical_reflection_2d():
    return numpy.array([[-1, 0], [0, 1]])


def identity_2d():
    return numpy.array([[1, 0], [0, 1]])


def random_euclidean_transform_2d():
    radians = numpy.random.uniform(low=0, high=2*numpy.pi, size=1)
    return rotation_2d(float(radians))
