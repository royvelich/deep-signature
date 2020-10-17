# numpy
import numpy

# scipy
import scipy.signal

# sklearn
import sklearn.preprocessing

# deep_signature
from deep_signature import euclidean_transform


def pad_curve(curve, padding):
    return numpy.pad(curve, ((padding, padding), (0, 0)), 'wrap')


def unpad_curve(curve, padding):
    x = curve[:, 0]
    y = curve[:, 1]
    range = numpy.arange(padding, curve.shape[0] - padding)
    unpadded_x = x[range]
    unpadded_y = y[range]
    return numpy.hstack((unpadded_x[:, numpy.newaxis], unpadded_y[:, numpy.newaxis]))


def calculate_dx_dt(curve):
    return numpy.gradient(curve[:, 0])


def calculate_dy_dt(curve):
    return numpy.gradient(curve[:, 1])


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


def calculate_curvature(curve):
    padded_curve = pad_curve(curve=curve, padding=2)
    tangent = calculate_tangent(padded_curve)
    dtangent_dt = calculate_tangent(tangent)
    dx_dt = tangent[:, 0]
    dy_dt = tangent[:, 1]
    d2x_dt2 = dtangent_dt[:, 0]
    d2y_dt2 = dtangent_dt[:, 1]
    kappa = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    kappa = kappa[2:-2]
    return kappa


def smooth_curve(curve, iterations, window_length, poly_order):
    smoothed_curve = numpy.copy(curve)
    for _ in range(iterations):
        x = smoothed_curve[:, 0]
        y = smoothed_curve[:, 1]
        smoothed_curve[:, 0] = scipy.signal.savgol_filter(x=x, window_length=window_length, polyorder=poly_order, mode='wrap')
        smoothed_curve[:, 1] = scipy.signal.savgol_filter(x=y, window_length=window_length, polyorder=poly_order, mode='wrap')

    return smoothed_curve


def evolve_curve(curve, evolution_iterations, evolution_dt, smoothing_window_length, smoothing_poly_order, smoothing_iterations):
    evolved_curve = numpy.copy(curve)
    for _ in range(evolution_iterations):
        kappa = calculate_curvature(evolved_curve)
        padded_curve = pad_curve(curve=curve, padding=1)
        normal = calculate_normal(curve=padded_curve)
        normal = unpad_curve(curve=normal, padding=1)
        normal = sklearn.preprocessing.normalize(X=normal, axis=1, norm='l2')
        delta = normal * kappa[:, numpy.newaxis]
        evolved_curve = evolved_curve + evolution_dt * delta
        evolved_curve = smooth_curve(curve=evolved_curve, iterations=smoothing_iterations, window_length=smoothing_window_length, poly_order=smoothing_poly_order)

    return evolved_curve


def translate_curve(curve, offset):
    translated_curve = curve + offset
    return translated_curve


def transform_curve(curve, radians, reflection):
    reflection_transform = euclidean_transform.identity_2d()
    if reflection == 'horizontal':
        reflection_transform = euclidean_transform.horizontal_reflection_2d()
    elif reflection == 'vertical':
        reflection_transform = euclidean_transform.vertical_reflection_2d()

    rotation_transform = euclidean_transform.rotation_2d(radians)
    transform = rotation_transform.dot(reflection_transform)
    transformed_curve = curve.dot(transform)

    return transformed_curve
