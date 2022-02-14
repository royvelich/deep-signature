# numpy
import numpy


def _validate_condition_number(A, min_cond, max_cond):
    w, v = numpy.linalg.eig(A)
    cond = numpy.linalg.cond(A)
    return ((w[0] > 0) and (w[1] > 0)) and (min_cond < cond < max_cond)


def generate_identity_transform_2d():
    return numpy.array([[1, 0], [0, 1]])


def generate_rotation_transform_2d(radians):
    c, s = numpy.cos(radians), numpy.sin(radians)
    return numpy.array([[c, s], [-s, c]])


def generate_horizontal_reflection_transform_2d():
    return numpy.array([[1, 0], [0, -1]])


def generate_vertical_reflection_transform_2d():
    return numpy.array([[-1, 0], [0, 1]])


def generate_random_euclidean_transform_2d():
    radians = numpy.random.uniform(low=0, high=2*numpy.pi, size=1)
    return generate_rotation_transform_2d(float(radians))


def generate_random_similarity_transform_2d(min_scale=0.5, max_scale=3):
    B = generate_random_euclidean_transform_2d()
    scale = numpy.random.uniform(low=min_scale, high=max_scale, size=1)
    S = scale * generate_identity_transform_2d()
    A = numpy.matmul(S, B)
    return A


def generate_random_equiaffine_transform_2d(max_scale=3, min_cond=2, max_cond=4):
    while True:
        scale = numpy.random.uniform(low=0, high=max_scale, size=2)
        coeffs = numpy.random.random(size=2)
        entries = scale * coeffs
        L = numpy.array([[1, 0], [entries[0], 1]])
        U = numpy.array([[1, entries[1]], [0, 1]])
        A = numpy.matmul(L, U)
        if _validate_condition_number(A=A, min_cond=min_cond, max_cond=max_cond):
            return A


def generate_random_affine_transform_2d(max_scale=3, min_cond=1.1, max_cond=3, min_det=1.5):
    while True:
        A = numpy.random.uniform(low=0, high=max_scale, size=(2,2))
        det = numpy.linalg.det(A)
        if _validate_condition_number(A=A, min_cond=min_cond, max_cond=max_cond) and (det > min_det):
            return A
