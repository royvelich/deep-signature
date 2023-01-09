# python peripherals
from __future__ import annotations
from abc import ABC
from typing import Optional, Union

# numpy
import numpy

# deep-signature
import deep_signature.core.transformations
from deep_signature.core.base import SeedableObject, SeedType


class Group(ABC, SeedableObject):
    def __init__(self, min_det: float, max_det: float, min_cond: float, max_cond: float, name: str, seed: Union[Optional[int], SeedType]):
        super().__init__(seed=seed)
        self._min_det = min_det
        self._max_det = max_det
        self._min_cond = min_cond
        self._max_cond = max_cond
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate_random_group_action(self, radians_u: Optional[float] = None, radians_v: Optional[float] = None) -> numpy.ndarray:
        if radians_u is None:
            radians_u = self._rng.uniform(low=0, high=2*numpy.pi, size=1)

        if radians_v is None:
            radians_v = self._rng.uniform(low=0, high=2*numpy.pi, size=1)

        det = float(self._rng.uniform(low=self._min_det, high=self._max_det, size=1))
        cond = float(self._rng.uniform(low=self._min_cond, high=self._max_cond, size=1))
        return deep_signature.core.transformations.generate_affine_transform_2d(det=det, cond=cond, radians_u=radians_u, radians_v=radians_v)

    @staticmethod
    def from_group_name(name: str, min_det: Optional[float] = None, max_det: Optional[float] = None, min_cond: Optional[float] = None, max_cond: Optional[float] = None) -> Group:
        if name == 'euclidean':
            return EuclideanGroup()
        elif name == 'similarity':
            return SimilarityGroup(min_det=min_det, max_det=max_det)
        elif name == 'equiaffine':
            return EquiaffineGroup(min_cond=min_cond, max_cond=max_cond)
        elif name == 'affine':
            return AffineGroup(min_det=min_det, max_det=max_det, min_cond=min_cond, max_cond=max_cond)


class AffineGroup(Group):
    def __init__(self, min_det: float, max_det: float, min_cond: float, max_cond: float, seed: Union[Optional[int], SeedType] = SeedType.Global):
        super().__init__(min_det=min_det, max_det=max_det, min_cond=min_cond, max_cond=max_cond, seed=seed, name='affine')


class SimilarityGroup(Group):
    def __init__(self, min_det: float, max_det: float, seed: Union[Optional[int], SeedType] = SeedType.Global):
        super().__init__(min_det=min_det, max_det=max_det, min_cond=1, max_cond=1, seed=seed, name='similarity')


class EquiaffineGroup(Group):
    def __init__(self, min_cond: float, max_cond: float, seed: Union[Optional[int], SeedType] = SeedType.Global):
        super().__init__(min_det=1, max_det=1, min_cond=min_cond, max_cond=max_cond, seed=seed, name='equiaffine')


class EuclideanGroup(Group):
    def __init__(self, seed: Union[Optional[int], SeedType] = SeedType.Global):
        super().__init__(min_det=1, max_det=1, min_cond=1, max_cond=1, seed=seed, name='euclidean')
