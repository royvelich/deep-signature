# numpy
import numpy


def validate_condition_number(a: numpy.ndarray, min_cond: float, max_cond: float) -> bool:
    w, v = numpy.linalg.eig(a)
    cond = numpy.linalg.cond(a)
    return ((w[0] > 0) and (w[1] > 0)) and (min_cond < cond < max_cond)


def validate_determinant(a: numpy.ndarray, min_det: float, max_det: float) -> bool:
    det = numpy.linalg.det(a)
    return min_det < det < max_det


def generate_identity_transform_2d() -> numpy.ndarray:
    return numpy.array([[1, 0], [0, 1]])


def generate_rotation_transform_2d(radians: float) -> numpy.ndarray:
    c, s = numpy.cos(radians).squeeze(), numpy.sin(radians).squeeze()
    return numpy.array([[c, s], [-s, c]])


def generate_horizontal_reflection_transform_2d() -> numpy.ndarray:
    return numpy.array([[1, 0], [0, -1]])


def generate_affine_transform_2d(det: float, cond: float, radians_u: float, radians_v: float) -> numpy.ndarray:
    U = generate_rotation_transform_2d(radians=radians_u)
    V = generate_rotation_transform_2d(radians=radians_v)
    s1 = numpy.sqrt(det / cond)
    s2 = det / s1
    S = numpy.array([[s1, 0], [0, s2]])
    A = numpy.matmul(numpy.matmul(U, S), V)
    return A
