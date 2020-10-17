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
