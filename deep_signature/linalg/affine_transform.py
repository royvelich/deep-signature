# numpy
import numpy


def _validate_eigen_values_ratio(A, min_eig_value_ratio, max_eig_value_ratio):
    w, v = numpy.linalg.eig(A)
    r1 = w[0] / w[1]
    r2 = w[1] / w[0]
    r = numpy.maximum(r1, r2)
    return ((r > min_eig_value_ratio) and (r < max_eig_value_ratio))


def generate_random_equiaffine_transform_2d(max_scale=1, min_eig_value_ratio=1.5, max_eig_value_ratio=3):
    while True:
        scale = numpy.random.uniform(low=0, high=max_scale, size=2)
        coeffs = numpy.random.random(size=2)
        entries = scale * coeffs
        L = numpy.array([[1, 0], [entries[0], 1]])
        U = numpy.array([[1, entries[1]], [0, 1]])
        A = numpy.matmul(L, U)
        if _validate_eigen_values_ratio(A=A, min_eig_value_ratio=min_eig_value_ratio, max_eig_value_ratio=max_eig_value_ratio):
            return A


def generate_random_affine_transform_2d(max_scale=1, min_eig_value_ratio=1.5, max_eig_value_ratio=3):
    while True:
        A = numpy.random.uniform(low=0, high=max_scale, size=(2,2))
        if _validate_eigen_values_ratio(A=A, min_eig_value_ratio=min_eig_value_ratio, max_eig_value_ratio=max_eig_value_ratio):
            return A
