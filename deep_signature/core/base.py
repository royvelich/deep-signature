# python peripherals
from typing import Union
import logging
from pathlib import Path

# numpy
import numpy

# deep-signature
from deep_signature.core import utils


class SeedableObject:
    _rng = numpy.random.default_rng()

    @staticmethod
    def set_seed(seed: int):
        SeedableObject._rng = numpy.random.default_rng(seed=seed)

    def __init__(self):
        self._rng = SeedableObject._rng

    @property
    def rng(self) -> numpy.random.Generator:
        return self._rng


class LoggerObject:
    def __init__(self, name: str, log_dir_path: Path, level: int = logging.DEBUG, **kw: object):
        self._name = name
        self._log_dir_path = log_dir_path
        self._log_name = f'{self.__class__.__name__} [{self._name}]'
        self._log_file_path = log_dir_path / f'{name}.log'
        self._log_dir_path.mkdir(parents=True, exist_ok=True)
        self._logger = utils.create_logger(log_file_path=self._log_file_path, name=self._log_name, level=level)
        super(LoggerObject, self).__init__(**kw)
