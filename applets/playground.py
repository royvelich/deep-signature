import numpy
import torch

from deep_signature.core.discrete_distributions import MultimodalGaussianDiscreteDistribution
from deep_signature.manifolds.planar_curves.implementation import PlanarCurve
import deep_signature.core.transformations

def uniform_proposal(x, delta=2.0):
    return numpy.random.uniform(x - delta, x + delta)

def metropolis_sampler(p, nsamples, proposal=uniform_proposal):
    x = 1 # start somewhere

    for i in range(nsamples):
        trial = proposal(x) # random neighbour from the proposal distribution
        acceptance = p(trial)/p(x)

        # accept the move conditionally
        if numpy.random.uniform() < acceptance:
            x = trial

        yield x

def gaussian(x, mu, sigma):
    return 1. / sigma / numpy.sqrt(2 * numpy.pi) * numpy.exp(-((x - mu) ** 2) / 2. / sigma / sigma)


class MyClass:
    def myprop(self):
        return 5


if __name__ == '__main__':

    bla = numpy.array([[1,1],[2,2],[3,3],[4,4],[5,5]])


    bla = torch.tensor([1,2,3])

    a = [1,2,3,4,5]
    b = a[:2]

    bla = MyClass()
    foo = dict()
    h = f'{bla.myprop=}'.split('=')[0]


    min_radius = 1
    max_radius = 10
    sampling_density = 100
    # radius = float(numpy.random.uniform(low=min_radius, high=max_radius, size=1))
    radius = 1
    circumference = 2 * radius * numpy.pi
    points_count = int(numpy.round(sampling_density * circumference))
    radians_delta = 2 * numpy.pi / points_count
    pointer = numpy.array([radius, 0])
    circle = numpy.empty((points_count, 2))
    for i in range(points_count):
        circle[i] = numpy.matmul(deep_signature.core.transformations.generate_rotation_transform_2d(radians=i * radians_delta), pointer)
    planar_curve = PlanarCurve(points=circle, closed=True)
    planar_curve.center_curve()


    k_equiaffine = planar_curve.calculate_equiaffine_k()
    k_equiaffine_cleaned = k_equiaffine[~numpy.isnan(k_equiaffine)]
    equiaffine_std = numpy.std(k_equiaffine_cleaned)

    import numpy.random
    import matplotlib.pyplot as plt

    dist = MultimodalGaussianDiscreteDistribution(bins_count=2000, multimodality=20)
    plt.plot(dist.pdf)
    plt.show()

    bla = dist.sample_pdf(samples_count=500)
    h = 5

    # for _ in range(2000):
    #     # Set-up.
    #     n = 10000
    #     numpy.random.seed(0x5eed)
    #     # Parameters of the mixture components
    #     norm_params = numpy.array([[5, 1],
    #                             [4, 2],
    #                             [9, 3.3]])
    #     n_components = norm_params.shape[0]
    #     # # Weight of each component, in this case all of them are 1/3
    #     weights = numpy.ones(n_components, dtype=np.float64) / 3.0
    #     # # A stream of indices from which to choose the component
    #     # mixture_idx = numpy.random.choice(len(weights), size=n, replace=True, p=weights)
    #     # # y is the mixture sample
    #     # y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
    #     #                    dtype=np.float64)
    #
    #     # Theoretical PDF plotting -- generate the x and y plotting positions
    #     xs = np.linspace(-5, 5, 5000)
    #     ys = np.zeros_like(xs)
    #
    #     for (l, s), w in zip(norm_params, weights):
    #         ys += ss.norm.pdf(xs, loc=l, scale=s) * w
    #
    #     ys = ys / numpy.sum(ys)
    #
    #     plt.plot(xs, ys)
    #
    #     bla = numpy.sum(ys)
    #
    #     # plt.hist(y, bins="fd")
    #     # plt.xlabel("x")
    #     # plt.ylabel("f(x)")
    #     # plt.show()