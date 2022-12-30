# python peripherals
from typing import Callable, Optional
from math import sqrt, pow, cos, sin, asin

# numpy
import numpy

# numba
import numba


# @numba.jit(nopython=True, fastmath=True)
def _get_min_distance(point: numpy.ndarray, subset: numpy.ndarray, max_distance: float, distance_function: Callable[[numpy.ndarray, numpy.ndarray], float]) -> Optional[float]:
	min_distance = numpy.inf
	for i in range(subset.shape[0]):
		distance = distance_function(subset[i, :], point)
		if distance < min_distance:
			min_distance = distance
		if min_distance < max_distance:
			return None

	return min_distance


# @numba.jit(nopython=True, fastmath=True)
def _get_max_min_distance(subset1: numpy.ndarray, subset2: numpy.ndarray, max_distance: Optional[float], distance_function: Callable[[numpy.ndarray, numpy.ndarray], float]) -> float:
	if max_distance is None:
		max_distance = 0.0

	for i in range(subset1.shape[0]):
		min_distance = _get_min_distance(point=subset1[i, :], subset=subset2, max_distance=max_distance, distance_function=distance_function)
		if min_distance is not None and min_distance > max_distance:
			max_distance = min_distance

	return max_distance


# @numba.jit(nopython=True, fastmath=True)
def manhattan(subset1: numpy.ndarray, subset2: numpy.ndarray) -> float:
	n = subset1.shape[0]
	ret = 0.
	for i in range(n):
		ret += abs(subset1[i] - subset2[i])
	return ret


# @numba.jit(nopython=True, fastmath=True)
def euclidean(subset1: numpy.ndarray, subset2: numpy.ndarray) -> float:
	n = subset1.shape[0]
	ret = 0.
	for i in range(n):
		ret += (subset1[i] - subset2[i]) ** 2
	return sqrt(ret)


# @numba.jit(nopython=True, fastmath=True)
def chebyshev(subset1: numpy.ndarray, subset2: numpy.ndarray) -> float:
	n = subset1.shape[0]
	ret = -1*numpy.inf
	for i in range(n):
		d = abs(subset1[i] - subset2[i])
		if d > ret:
			ret = d
	return ret


# @numba.jit(nopython=True, fastmath=True)
def cosine(subset1: numpy.ndarray, subset2: numpy.ndarray) -> float:
	n = subset1.shape[0]
	xy_dot = 0.
	x_norm = 0.
	y_norm = 0.
	for i in range(n):
		xy_dot += subset1[i] * subset2[i]
		x_norm += subset1[i] * subset1[i]
		y_norm += subset2[i] * subset2[i]
	return 1.-xy_dot/(sqrt(x_norm)*sqrt(y_norm))


# @numba.jit(nopython=True, fastmath=True)
def haversine(array_x: numpy.ndarray, array_y: numpy.ndarray) -> float:
	R = 6378.0
	radians = numpy.pi / 180.0
	lat_x = radians * array_x[0]
	lon_x = radians * array_x[1]
	lat_y = radians * array_y[0]
	lon_y = radians * array_y[1]
	dlon = lon_y - lon_x
	dlat = lat_y - lat_x
	a = (pow(sin(dlat/2.0), 2.0) + cos(lat_x) * cos(lat_y) * pow(sin(dlon/2.0), 2.0))
	return R * 2 * asin(sqrt(a))


# @numba.jit(nopython=True, fastmath=True)
def hausdorff_distance(subset1: numpy.ndarray, subset2: numpy.ndarray, distance_function: Callable[[numpy.ndarray, numpy.ndarray], float]) -> float:
	max_min_distance1 = _get_max_min_distance(subset1=subset1, subset2=subset2, max_distance=None, distance_function=distance_function)
	max_min_distance2 = _get_max_min_distance(subset1=subset2, subset2=subset1, max_distance=max_min_distance1, distance_function=distance_function)
	return max_min_distance2
