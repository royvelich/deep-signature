# python peripherals
from abc import ABC, abstractmethod

# numpy
import numpy

# deep-signature
import deep_signature.transformations


class Group(ABC):
    def __init__(self, min_det: float, max_det: float, min_cond: float, max_cond: float, name: str):
        self._min_det = min_det
        self._max_det = max_det
        self._min_cond = min_cond
        self._max_cond = max_cond
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate_random_group_action(self) -> numpy.ndarray:
        radians_u = numpy.random.uniform(low=0, high=2*numpy.pi, size=1)
        radians_v = numpy.random.uniform(low=0, high=2*numpy.pi, size=1)
        det = float(numpy.random.uniform(low=self._min_det, high=self._max_det, size=1))
        cond = float(numpy.random.uniform(low=self._min_cond, high=self._max_cond, size=1))
        return deep_signature.transformations.generate_affine_transform_2d(det=det, cond=cond, radians_u=radians_u, radians_v=radians_v)


class AffineGroup(Group):
    def __init__(self, min_det: float, max_det: float, min_cond: float, max_cond: float):
        super().__init__(min_det=min_det, max_det=max_det, min_cond=min_cond, max_cond=max_cond, name='affine')


class SimilarityGroup(Group):
    def __init__(self, min_det: float, max_det: float):
        super().__init__(min_det=min_det, max_det=max_det, min_cond=1, max_cond=1, name='similarity')


class EquiaffineGroup(Group):
    def __init__(self, min_cond: float, max_cond: float):
        super().__init__(min_det=1, max_det=1, min_cond=min_cond, max_cond=max_cond, name='equiaffine')


class EuclideanGroup(Group):
    def __init__(self):
        super().__init__(min_det=1, max_det=1, min_cond=1, max_cond=1, name='euclidean')
