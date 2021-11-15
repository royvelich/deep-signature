# numpy
import numpy


def _validate_eigen_values_ratio(A, min_cond, max_cond):
    w, v = numpy.linalg.eig(A)
    cond = numpy.linalg.cond(A)
    return ((w[0] > 0) and (w[1] > 0)) and (min_cond < cond < max_cond)


def generate_random_equiaffine_transform_2d(max_scale=1, min_cond=1.2, max_cond=2):
    while True:
        scale = numpy.random.uniform(low=0, high=max_scale, size=2)
        coeffs = numpy.random.random(size=2)
        entries = scale * coeffs
        L = numpy.array([[1, 0], [entries[0], 1]])
        U = numpy.array([[1, entries[1]], [0, 1]])
        A = numpy.matmul(L, U)
        if _validate_eigen_values_ratio(A=A, min_cond=min_cond, max_cond=max_cond):
            return A


def generate_random_affine_transform_2d(max_scale=1, min_cond=1.2, max_cond=2):
    while True:
        A = numpy.random.uniform(low=0, high=max_scale, size=(2,2))
        if _validate_eigen_values_ratio(A=A, min_cond=min_cond, max_cond=max_cond):
            return A
