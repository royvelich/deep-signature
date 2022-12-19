# python peripherals
import random
import logging
from pathlib import Path
from typing import Optional

# numpy
import numpy

# deep-signature
from deep_signature.core import utils

# torch
import torch


class SeedableObject:
    _rng = numpy.random.default_rng()

    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        SeedableObject._rng = numpy.random.default_rng(seed=seed)

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            self._rng = SeedableObject._rng
        else:
            self._rng = numpy.random.default_rng(seed=seed)

    @property
    def rng(self) -> numpy.random.Generator:
        return self._rng


class LoggerObject:
    def __init__(self, log_dir_path: Path, level: int = logging.DEBUG, **kw: object):
        self._log_dir_path = log_dir_path
        self._log_name = self.__class__.__name__
        self._log_file_path = log_dir_path / 'log.txt'
        self._log_dir_path.mkdir(parents=True, exist_ok=True)
        self._logger = utils.create_logger(log_file_path=self._log_file_path, name=self._log_name, level=level)
        super(LoggerObject, self).__init__(**kw)
