# numpy
import numpy


def generate_rotation_transform_2d(radians):
    c, s = numpy.cos(radians), numpy.sin(radians)
    return numpy.array([[c, s], [-s, c]])


def generate_horizontal_reflection_transform_2d():
    return numpy.array([[1, 0], [0, -1]])


def generate_vertical_reflection_transform_2d():
    return numpy.array([[-1, 0], [0, 1]])


def generate_identity_transform_2d():
    return numpy.array([[1, 0], [0, 1]])


def generate_random_euclidean_transform_2d():
    radians = numpy.random.uniform(low=0, high=2*numpy.pi, size=1)
    return generate_rotation_transform_2d(float(radians))
