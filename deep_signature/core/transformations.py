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


def generate_vertical_reflection_transform_2d() -> numpy.ndarray:
    return numpy.array([[-1, 0], [0, 1]])


def generate_affine_transform_2d(det: float, cond: float, radians_u: float, radians_v: float) -> numpy.ndarray:
    U = generate_rotation_transform_2d(radians=radians_u)
    V = generate_rotation_transform_2d(radians=radians_v)
    s1 = numpy.sqrt(det / cond)
    s2 = det / s1
    S = numpy.array([[s1, 0], [0, s2]])
    A = numpy.matmul(numpy.matmul(U, S), V)
    return A

# # https://gist.github.com/bstellato/23322fe5d87bb71da922fbc41d658079
# def generate_affine_transform_2d(det: float, cond: float, radians_u: float, radians_v: float) -> numpy.ndarray:
#     U = generate_rotation_transform_2d(radians=radians_u)
#     V = generate_rotation_transform_2d(radians=radians_v)
#
#     # n = 2
#     # log_cond_P = numpy.log(cond)
#     # exp_vec = numpy.arange(-log_cond_P / 4., log_cond_P * (n + 1) / (4 * (n - 1)), log_cond_P / (2. * (n - 1)))
#     # s = numpy.exp(exp_vec)
#     # S = numpy.diag(s)
#     # U, _ = numpy.linalg.qr((numpy.random.rand(n, n) - 5.) * 200)
#     # V, _ = numpy.linalg.qr((numpy.random.rand(n, n) - 5.) * 200)
#     # P = U.dot(S).dot(V.T)
#     # P = P.dot(P.T) * numpy.sqrt(det)
#     # A, S, B = numpy.linalg.svd(P)
#     # return P
#
#     # s1 = numpy.random.uniform(low=0.1, high=5)
#     # s2 = s1 / cond
#     #
#     # # s1 = numpy.sqrt(det / cond)
#     # # s2 = det / s1
#     # S = numpy.array([[s1, 0], [0, s2]])
#     # A = numpy.matmul(numpy.matmul(U, S), V)
#     #
#     # current_det = numpy.linalg.det(A)
#     # A = A / numpy.sqrt(numpy.linalg.det(A))
#     # A = A * numpy.sqrt(det)
#     # N, S2, M = numpy.linalg.svd(A)
#     #
#     # return A
#
#     while True:
#         A = numpy.random.uniform(low=0.1, high=2, size=(2,2))
#         det_A = numpy.linalg.det(A)
#         cond_A = numpy.linalg.cond(A)
#         if 1 <= det_A <= 3 and 1 <= cond_A <= 3:
#             return A
