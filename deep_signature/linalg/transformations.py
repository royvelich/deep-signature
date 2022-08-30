import random

# numpy
import numpy

# deep-signature
from utils import settings

def _validate_condition_number(A, min_cond, max_cond):
    w, v = numpy.linalg.eig(A)
    cond = numpy.linalg.cond(A)
    # return ((w[0] > 0) and (w[1] > 0)) and (min_cond < cond < max_cond)
    return ((w[0] > 0) and (w[1] > 0)) and (cond < max_cond)

def _validate_determinant(A, min_det, max_det):
    det = numpy.linalg.det(A)
    return det < max_det
    # return min_det < det < max_det


def generate_identity_transform_2d():
    return numpy.array([[1, 0], [0, 1]])


def generate_rotation_transform_2d(radians):
    c, s = numpy.cos(radians), numpy.sin(radians)
    return numpy.array([[c, s], [-s, c]]).astype('float32')


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


# def generate_random_similarity_transform_2d(min_det, max_det, rotation=False):
#     A = generate_random_affine_transform_2d(min_cond=1, max_cond=1, min_det=min_det, max_det=max_det, rotation=False)
#     return A

def generate_random_similarity_transform_2d(min_det, max_det, min_scale=1, max_scale=3):
    B = generate_random_euclidean_transform_2d()
    scale = numpy.random.uniform(low=min_scale, high=max_scale, size=1)
    S = scale * generate_identity_transform_2d()
    A = numpy.matmul(S, B)
    return A


def generate_random_similarity_transform_2d_eval(min_det, max_det, rotation=False):
    A = generate_random_affine_transform_2d(min_cond=1, max_cond=1, min_det=min_det, max_det=max_det, rotation=False)
    return A


def generate_random_equiaffine_transform_2d(min_cond, max_cond, rotation=True):
    return generate_random_affine_transform_2d(min_cond=min_cond, max_cond=max_cond, min_det=1, max_det=1, rotation=rotation)


def generate_random_affine_transform_2d(min_cond, max_cond, min_det, max_det, rotation=True):
    U = generate_random_euclidean_transform_2d(rotation=rotation)
    V = generate_random_euclidean_transform_2d(rotation=rotation)
    det = float(numpy.random.uniform(low=min_det, high=max_det, size=1))
    cond = float(numpy.random.uniform(low=min_cond, high=max_cond, size=1))
    s1 = numpy.sqrt(det / cond)
    s2 = det / s1
    if bool(random.getrandbits(1)):
        S = numpy.array([[s1, 0], [0, s2]])
    else:
        S = numpy.array([[s2, 0], [0, s1]])
    A = numpy.matmul(numpy.matmul(U, S), V)
    return A


def generate_random_transform_2d(transform_type, min_cond, max_cond, min_det, max_det, rotation=True):
    if transform_type == 'euclidean':
        return generate_random_euclidean_transform_2d(rotation=rotation)
    elif transform_type == 'similarity':
        return generate_random_similarity_transform_2d(min_det=min_det, max_det=max_det)
    elif transform_type == 'equiaffine':
        return generate_random_equiaffine_transform_2d(min_cond=min_cond, max_cond=max_cond, rotation=rotation)
    elif transform_type == 'affine':
        return generate_random_affine_transform_2d(min_cond=min_cond, max_cond=max_cond, min_det=min_det, max_det=max_det, rotation=rotation)


def generate_random_transform_2d_training(transform_type, rotation=True):
    if transform_type == 'euclidean':
        return generate_random_euclidean_transform_2d(rotation=rotation).astype('float32')
    elif transform_type == 'similarity':
        return generate_random_similarity_transform_2d(min_det=settings.affine_min_det_training, max_det=settings.affine_max_det_training).astype('float32')
    elif transform_type == 'equiaffine':
        return generate_random_equiaffine_transform_2d(min_cond=settings.equiaffine_min_cond_training, max_cond=settings.equiaffine_max_cond_training, rotation=rotation).astype('float32')
    elif transform_type == 'affine':
        return generate_random_affine_transform_2d(min_cond=settings.affine_min_cond_training, max_cond=settings.affine_max_cond_training, min_det=settings.affine_min_det_training, max_det=settings.affine_max_det_training, rotation=rotation).astype('float32')


def generate_random_transform_2d_evaluation(transform_type, rotation=True):
    if transform_type == 'euclidean':
        return generate_random_euclidean_transform_2d(rotation=rotation).astype('float32')
    elif transform_type == 'similarity':
        return generate_random_similarity_transform_2d_eval(min_det=settings.affine_min_det_evaluation, max_det=settings.affine_max_det_evaluation).astype('float32')
    elif transform_type == 'equiaffine':
        return generate_random_equiaffine_transform_2d(min_cond=settings.equiaffine_min_cond_evaluation, max_cond=settings.equiaffine_max_cond_evaluation, rotation=rotation).astype('float32')
    elif transform_type == 'affine':
        return generate_random_affine_transform_2d(min_cond=settings.affine_min_cond_evaluation, max_cond=settings.affine_max_cond_evaluation, min_det=settings.affine_min_det_evaluation, max_det=settings.affine_max_det_evaluation, rotation=rotation).astype('float32')
