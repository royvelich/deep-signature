# numpy
import numpy

# scipy
import scipy.signal

# sklearn
import sklearn.preprocessing

# itertools
from itertools import combinations

# deep_signature
from deep_signature.linalg import euclidean_transform

# -------------------------------------------------
# curve padding
# -------------------------------------------------
def pad_curve(curve, padding):
    return numpy.pad(curve, ((padding, padding), (0, 0)), 'wrap')


def unpad_curve(curve, padding):
    x = curve[:, 0]
    y = curve[:, 1]
    range = numpy.arange(padding, curve.shape[0] - padding)
    unpadded_x = x[range]
    unpadded_y = y[range]
    return numpy.hstack((unpadded_x[:, numpy.newaxis], unpadded_y[:, numpy.newaxis]))


# -------------------------------------------------
# partial derivatives
# -------------------------------------------------
def calculate_dx_dt(curve):
    return numpy.gradient(curve[:, 0])


def calculate_dy_dt(curve):
    return numpy.gradient(curve[:, 1])


# -------------------------------------------------
# tangent and normal
# -------------------------------------------------
def calculate_tangent(curve):
    tangent = numpy.empty_like(curve)
    tangent[:, 0] = calculate_dx_dt(curve)
    tangent[:, 1] = calculate_dy_dt(curve)
    return tangent


def calculate_normal(curve):
    normal = numpy.empty_like(curve)
    normal[:, 0] = calculate_dy_dt(curve)
    normal[:, 1] = -calculate_dx_dt(curve)
    return normal


# -------------------------------------------------
# euclidean curvature and arclength
# -------------------------------------------------
def calculate_euclidean_curvature(curve, padding=True):
    if padding is True:
        padded_curve = pad_curve(curve=curve, padding=2)
    else:
        padded_curve = curve

    tangent = calculate_tangent(padded_curve)
    dtangent_dt = calculate_tangent(tangent)
    dx_dt = tangent[:, 0]
    dy_dt = tangent[:, 1]
    d2x_dt2 = dtangent_dt[:, 0]
    d2y_dt2 = dtangent_dt[:, 1]
    kappa = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    kappa = kappa[2:-2]
    return kappa


def calculate_euclidean_arclength(curve):
    adjacent_diff = numpy.diff(a=curve, n=1, axis=0)
    diff_norm = numpy.linalg.norm(x=adjacent_diff, ord=2, axis=1)
    s = numpy.cumsum(a=numpy.concatenate((numpy.array([0]), diff_norm), axis=0))
    return s


# -------------------------------------------------
# curve smoothing
# -------------------------------------------------
def smooth_curve(curve, iterations, window_length, poly_order):
    smoothed_curve = numpy.copy(curve)
    for _ in range(iterations):
        x = smoothed_curve[:, 0]
        y = smoothed_curve[:, 1]
        smoothed_curve[:, 0] = scipy.signal.savgol_filter(x=x, window_length=window_length, polyorder=poly_order, mode='wrap')
        smoothed_curve[:, 1] = scipy.signal.savgol_filter(x=y, window_length=window_length, polyorder=poly_order, mode='wrap')

    return smoothed_curve


# -------------------------------------------------
# curve evolution
# -------------------------------------------------
def evolve_curve(curve, evolution_iterations, evolution_dt, smoothing_window_length, smoothing_poly_order, smoothing_iterations):
    evolved_curve = numpy.copy(curve)
    for _ in range(evolution_iterations):
        kappa = calculate_euclidean_curvature(evolved_curve)
        padded_curve = pad_curve(curve=curve, padding=1)
        normal = calculate_normal(curve=padded_curve)
        normal = unpad_curve(curve=normal, padding=1)
        normal = sklearn.preprocessing.normalize(X=normal, axis=1, norm='l2')
        delta = normal * kappa[:, numpy.newaxis]
        evolved_curve = evolved_curve + evolution_dt * delta
        evolved_curve = smooth_curve(curve=evolved_curve, iterations=smoothing_iterations, window_length=smoothing_window_length, poly_order=smoothing_poly_order)

    return evolved_curve


# -------------------------------------------------
# curve transformation
# -------------------------------------------------
def normalize_curve(curve, force_ccw=False, force_end_point=False, index1=None, index2=None, center_index=None):
    if center_index is None:
        center_index = get_middle_index(curve)
    normalized_curve = translate_curve(curve=curve, offset=-curve[center_index])

    if force_ccw is True:
        if not is_ccw(curve=normalized_curve):
            normalized_curve = numpy.flip(m=normalized_curve, axis=0)

    radians = calculate_secant_angle(curve=normalized_curve, index1=index1, index2=index2)
    normalized_curve = rotate_curve(curve=normalized_curve, radians=radians)

    if force_end_point is True:
        end_point = normalized_curve[-1]
        if end_point[0] < 0:
            normalized_curve = normalized_curve * numpy.array([[-1,1]] * curve.shape[0])

        if end_point[1] < 0:
            normalized_curve = normalized_curve * numpy.array([[1,-1]] * curve.shape[0])

    return normalized_curve


def center_curve(curve):
    return translate_curve(curve=curve, offset=-numpy.mean(curve, axis=0))


def translate_curve(curve, offset):
    translated_curve = curve + offset
    return translated_curve


def rotate_curve(curve, radians):
    rotation_transform = euclidean_transform.generate_rotation_transform_2d(radians)
    transformed_curve = curve.dot(rotation_transform)
    return transformed_curve


def transform_curve(curve, transform):
    transformed_curve = curve.dot(transform)
    return transformed_curve


# -------------------------------------------------
# helpers
# -------------------------------------------------
def match_curve_sample_tangents(curve_sample1, curve_sample2, index1, index2):
    tangent1 = curve_sample1[index1] - curve_sample1[index2]
    tangent2 = curve_sample2[index1] - curve_sample2[index2]

    tangent1 = tangent1 / numpy.linalg.norm(tangent1)
    tangent2 = tangent2 / numpy.linalg.norm(tangent2)
    normal2 = numpy.array([-tangent2[1], tangent2[0]])

    distance = numpy.dot(tangent1, normal2)
    cosine = numpy.dot(tangent1, tangent2)
    radians = numpy.arccos(cosine)
    rotation_transform = euclidean_transform.generate_rotation_transform_2d(-numpy.sign(distance) * numpy.abs(radians))

    return curve_sample1.dot(rotation_transform), curve_sample2


def is_ccw(curve, index1=None, index2=None, index3=None):
    if index1 is None:
        index1 = get_leftmost_index(curve)

    if index2 is None:
        index2 = get_middle_index(curve)

    if index3 is None:
        index3 = get_rightmost_index(curve)

    eps = 1e-9
    pointer01 = curve[index1] - curve[index2]
    pointer12 = curve[index3] - curve[index2]
    pointer12 = pointer12 / (numpy.linalg.norm(pointer12) + eps)
    normal01 = numpy.array([-pointer01[1], pointer01[0]])
    normal01 = normal01 / (numpy.linalg.norm(normal01) + eps)
    return numpy.dot(pointer12, normal01) < 0


def calculate_secant_angle(curve, index1=None, index2=None):
    if index1 is None:
        index1 = get_leftmost_index(curve)

    if index2 is None:
        index2 = get_middle_index(curve)

    eps = 1e-9
    secant = curve[index1] - curve[index2]
    secant = secant / (numpy.linalg.norm(secant) + eps)
    axis = numpy.array([1, 0])
    axis = axis / numpy.linalg.norm(axis)
    normal = numpy.array([-axis[1], axis[0]])
    radians = -numpy.sign(numpy.dot(secant, normal)) * numpy.arccos(numpy.dot(secant, axis))
    return radians


def get_leftmost_index(curve):
    return 0


def get_middle_index(curve):
    return int(numpy.floor(curve.shape[0] / 2))


def get_rightmost_index(curve):
    return curve.shape[0] - 1


# -------------------------------------------------
# equiaffine curvature and arclength approximation
# https://link.springer.com/article/10.1023/A:1007992709392
# -------------------------------------------------
def calculate_signed_parallelogram_area(curve, indices):
    if indices.shape[0] == 4:
        return calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 1, 3]]) - calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 1, 2]])

    points = curve[indices]
    points = numpy.hstack((points, numpy.ones((points.shape[0], 1), dtype=points.dtype)))
    return numpy.linalg.det(points)


def calculate_T(curve, indices):
    T = 1
    for item in combinations(indices, 3):
        sorted_indices = sorted(item)
        T = T * calculate_signed_parallelogram_area(curve=curve, indices=numpy.array(sorted_indices))
    return T / 4


def calculate_S(curve, indices):
    c013 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 1, 3]])
    c024 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 2, 4]])
    c1234 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[1, 2, 3, 4]])
    c012 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 1, 2]])
    c034 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 3, 4]])
    c1324 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[1, 3, 2, 4]])
    c123 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[1, 2, 3]])
    c234 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[2, 3, 4]])
    c124 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[1, 2, 4]])
    c134 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[1, 3, 4]])

    t1 = numpy.square(c013) * numpy.square(c024) * numpy.square(c1234)
    t2 = numpy.square(c012) * numpy.square(c034) * numpy.square(c1324)
    t3 = 2 * c012 * c034 * c013 * c024 * ((c123 * c234) + (c124 * c134))

    return (t1 + t2 - t3) / 4


def calculate_N(curve, indices):
    c123 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[1, 2, 3]])
    c134 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[1, 3, 4]])
    c023 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 2, 3]])
    c014 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 1, 4]])
    c1234 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[1, 2, 3, 4]])
    c012 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 1, 2]])
    c034 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[0, 3, 4]])
    c2314 = calculate_signed_parallelogram_area(curve=curve, indices=indices[[2, 3, 1, 4]])

    t1 = -c123 * c134
    t2 = numpy.square(c023) * numpy.square(c014) * c1234
    t3 = numpy.square(c012) * numpy.square(c034) * c2314
    t4 = c012 * c023 * c014 * c034 * (c134 - c123)

    return (t1 * (t2 + t3 + t4)) / 4


def calculate_elliptic_area(curve, indices):
    S = calculate_S(curve=curve, indices=indices)
    N = calculate_N(curve=curve, indices=indices)
    return N / (2 * S)


def calculate_equiaffine_curvature_at_point(curve, indices):
    T = calculate_T(curve=curve, indices=indices)
    S = calculate_S(curve=curve, indices=indices)
    return S / numpy.power(T, 2/3)


def calculate_equiaffine_curvature(curve):
    k = numpy.zeros(curve.shape[0])
    for i in range(curve.shape[0]):
        indices = numpy.array(list(range((i - 2), (i + 3))))
        indices = numpy.mod(indices, curve.shape[0])
        k[i] = calculate_equiaffine_curvature_at_point(curve=curve, indices=indices)
    return k


def calculate_equiaffine_arclength_for_quintuple(curve, indices):
    kappa = calculate_equiaffine_curvature_at_point(curve=curve, indices=indices)
    area = calculate_elliptic_area(curve=curve, indices=indices)
    return 2 * area * numpy.abs(kappa)


def calculate_equiaffine_arclength(curve):
    s = numpy.zeros(curve.shape[0])
    for i in numpy.arange(start=1, stop=curve.shape[0] - 1, step=3):
        indices = numpy.arange(start=i-1, stop=i+4)
        s[i] = s[i - 1] + calculate_equiaffine_arclength_for_quintuple(curve=curve, indices=indices)
    return s


def calculate_equiaffine_arclength_by_euclidean_metrics(curve):
    s = calculate_euclidean_arclength(curve=curve[2:-2])
    k = numpy.abs(calculate_euclidean_curvature(curve=curve, padding=False))
    ds = numpy.diff(a=s, n=1, axis=0)
    ds = numpy.concatenate((numpy.array([0]), ds))
    k_cbrt = numpy.cbrt(k)
    ds_affine = k_cbrt * ds
    s_affine = numpy.cumsum(a=ds_affine, axis=0)
    return s_affine
