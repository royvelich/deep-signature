# python peripherals
from abc import ABC, abstractmethod

# numpy
import numpy

# scipy
import scipy.stats

# matplotlib
import matplotlib.axes

# deep-signature
import deep_signature.manifolds.planar_curves.visualization
from deep_signature.core.base import SeedableObject


class DiscreteDistribution(ABC, SeedableObject):
    def __init__(self, bins_count: int):
        super(ABC, self).__init__()
        super(SeedableObject, self).__init__()
        self._bins_count = bins_count
        self._pdf = self._generate_pdf()

    @property
    def pdf(self) -> numpy.ndarray:
        return self._pdf

    def sample_pdf(self, samples_count: int) -> numpy.ndarray:
        return self._rng.choice(a=self._bins_count, size=samples_count, replace=False, p=self._pdf)

    def plot_dist(self, ax: matplotlib.axes.Axes, line_width: float = 2, alpha: float = 1.0, cmap: str = 'hsv'):
        x = numpy.array(list(range(self._pdf.shape[0])))
        y = self._pdf
        deep_signature.manifolds.planar_curves.visualization.plot_multicolor_line(x=x, y=y, ax=ax, line_width=line_width, alpha=alpha, cmap=cmap)

    @abstractmethod
    def _generate_pdf(self) -> numpy.ndarray:
        pass


class MultimodalGaussianDiscreteDistribution(DiscreteDistribution):
    def __init__(self, bins_count: int, multimodality: int):
        self._multimodality = multimodality
        super().__init__(bins_count=bins_count)

    def _generate_pdf(self) -> numpy.ndarray:
        scales = self._generate_scales()
        weight = 1 / self._multimodality

        is_even = self._bins_count % 2 == 0
        half_bins = numpy.floor(self._bins_count / 2)
        start, stop = -half_bins, half_bins
        if not is_even:
            stop = stop + 1

        quantiles = numpy.linspace(start=start, stop=stop, num=self._bins_count)
        pdf = numpy.zeros_like(quantiles)

        for scale in scales:
            current_pdf = scipy.stats.norm.pdf(quantiles, loc=0, scale=scale)
            shift = self._rng.integers(low=0, high=self._bins_count)
            current_pdf = numpy.roll(a=current_pdf, shift=shift)
            pdf += current_pdf * weight

        # pdf = pdf + 1e-5
        pdf = pdf / numpy.sum(pdf)
        return pdf

    def _generate_scales(self) -> numpy.ndarray:
        # mean = self._bins_count * 0.25
        # sd = self._bins_count * 0.25
        # lower = self._bins_count * 0.01
        # upper = self._bins_count * 0.5
        # truncated_normal = MultimodalGaussianDiscreteDistribution._generate_truncated_normal(mean=mean, sd=sd, lower=lower, upper=upper)
        # return truncated_normal.rvs(self._multimodality)
        return self._rng.integers(low=self._bins_count//16, high=self._bins_count//8, size=self._multimodality)


    # https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
    @staticmethod
    def _generate_truncated_normal(mean: float, sd: float, lower: float, upper: float):
        return scipy.stats.truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)


class UniformDiscreteDistribution(DiscreteDistribution):
    def __init__(self, bins_count: int):
        super().__init__(bins_count=bins_count)

    def _generate_pdf(self) -> numpy.ndarray:
        return numpy.array([1 / self._bins_count] * self._bins_count)
