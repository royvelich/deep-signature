# python peripherals
from typing import Union

# numpy
import numpy


class SeedableObject:
    seed: Union[None, int] = None

    def __init__(self):
        self._rng = numpy.random.default_rng(seed=SeedableObject.seed)

    @property
    def rng(self) -> numpy.random.Generator:
        return self._rng
