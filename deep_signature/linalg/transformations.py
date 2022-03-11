# numpy
import numpy

# deep-signature
from utils import settings

def _validate_condition_number(A, min_cond, max_cond):
    w, v = numpy.linalg.eig(A)
    cond = numpy.linalg.cond(A)
    return ((w[0] > 0) and (w[1] > 0)) and (min_cond < cond < max_cond)


def _validate_determinant(A, min_det, max_det):
    det = numpy.linalg.det(A)
    return min_det < det < max_det


def generate_identity_transform_2d():
    return numpy.array([[1, 0], [0, 1]])


def generate_rotation_transform_2d(radians):
    c, s = numpy.cos(radians), numpy.sin(radians)
    return numpy.array([[c, s], [-s, c]])


def generate_horizontal_reflection_transform_2d():
    return numpy.array([[1, 0], [0, -1]])


def generate_vertical_reflection_transform_2d():
    return numpy.array([[-1, 0], [0, 1]])


def generate_random_euclidean_transform_2d(rotation=True):
    if rotation is True:
        radians = numpy.random.uniform(low=0, high=2*numpy.pi, size=1)
    else:
        radians = 0
    return generate_rotation_transform_2d(float(radians))
    # return generate_rotation_transform_2d(float(numpy.pi / 4))


def generate_random_similarity_transform_2d(min_scale=0.5, max_scale=3):
    B = generate_random_euclidean_transform_2d()
    scale = numpy.random.uniform(low=min_scale, high=max_scale, size=1)
    S = scale * generate_identity_transform_2d()
    A = numpy.matmul(S, B)
    return A


def generate_random_equiaffine_transform_2d(min_cond, max_cond, rotation=True):
    return generate_random_affine_transform_2d(min_cond=min_cond, max_cond=max_cond, min_det=1, max_det=1, rotation=rotation)
    # B = generate_rotation_transform_2d(float(numpy.pi / 3))
    # return numpy.matmul(numpy.array([[2, 0], [0, 1/2]]), B)
    # return numpy.array([[2, 0], [0, 1/2]])
    # while True:
    #     scale = numpy.random.uniform(low=0, high=max_scale, size=2)
    #     coeffs = numpy.random.random(size=2)
    #     entries = scale * coeffs
    #     L = numpy.array([[1, 0], [entries[0], 1]])
    #     U = numpy.array([[1, entries[1]], [0, 1]])
    #     A = numpy.matmul(L, U)
    #     if _validate_condition_number(A=A, min_cond=min_cond, max_cond=max_cond):
    #         return A


def generate_random_affine_transform_2d(max_scale=3, min_cond=1.1, max_cond=3, min_det=2):
    while True:
        A = numpy.random.uniform(low=0, high=max_scale, size=(2,2))
        det = numpy.linalg.det(A)
        if _validate_condition_number(A=A, min_cond=min_cond, max_cond=max_cond) and (det > min_det):
            return A

    # B = generate_rotation_transform_2d(float(numpy.pi / 5))
    # return numpy.matmul(numpy.array([[1.3, 0], [0, 2.3]]), B)
    # return numpy.array([[1.5, 0], [0, 2.2]])
    # while True:
    #     A = numpy.random.uniform(low=0, high=max_scale, size=(2, 2))
    #     if (_validate_condition_number(A=A, min_cond=min_cond, max_cond=max_cond) is True) and (_validate_determinant(A=A, min_det=min_det, max_det=max_det) is True):
    #         return A


def generate_random_transform_2d(transform_type, min_cond, max_cond, min_det, max_det, rotation=True):
    return generate_random_affine_transform_2d()


def generate_random_transform_2d_training(transform_type, rotation=True):
    return generate_random_affine_transform_2d()


def generate_random_transform_2d_evaluation(transform_type, rotation=True):
    return generate_random_affine_transform_2d()
