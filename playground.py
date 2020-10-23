import numpy
import torch
import sklearn.preprocessing
import math
import scipy.stats
import itertools
import random
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from deep_signature.data_generation import CurveDatasetGenerator
from deep_signature.data_generation import CurveDataGenerator
from skimage.util.shape import view_as_windows


# https://stackoverflow.com/questions/36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar
def colorline(ax, x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = numpy.array([z])

    z = numpy.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    # ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_dist(ax, dist):
    x = numpy.array(range(dist.shape[0]))
    y = dist
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    colorline(ax=ax, x=x, y=y, cmap='hsv')


def plot_curve_section_sample(ax, curve, curve_section_sample, dist, sampling_indices, point_size=10):
    x = curve_section_sample[:,0]
    y = curve_section_sample[:,1]
    c = numpy.linspace(0.0, 1.0, dist.shape[0])

    ax.scatter(
        x=x,
        y=y,
        c=c[sampling_indices],
        s=point_size,
        cmap='hsv')


def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    a_ext = numpy.concatenate((a,a[:,:-1]),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return view_as_windows(a_ext,(1,n))[numpy.arange(len(r)), (n-r)%n,0]


if __name__ == '__main__':

    # arr1 = numpy.array([1,2,3])
    # arr2 = numpy.array([4,5,6])
    # arr3 = numpy.vstack((arr1, arr2))
    # arr4 = numpy.vstack((arr2, arr3))
    #
    # h = 5

    # # momo = numpy.mod(-1, 300)
    #
    # arr1 = numpy.array([1,2,3])
    # arr2 = numpy.array([4,5,6])
    # arr = numpy.concatenate((arr1, arr2))
    #
    # bla = numpy.array([[1,2,3],[4,5,6],[7,8,9]])
    #
    # shift = numpy.round(numpy.random.rand(3) * 3).astype(int)
    # # bla2 = numpy.roll(a=bla, axis=(1,1,1), shift=(2,0,1))
    #
    # bla2 = strided_indexing_roll(bla, shift)


    rotation_factor = 10
    sectioning_factor = 20
    sampling_factor = 4
    multimodality_factor = 20
    sampling_points_count = 400
    supporting_points_count = 10

    curve_dataset_generator = CurveDatasetGenerator()
    curves = curve_dataset_generator.load_curves(file_path="C:/deep-signature-data/curves/curves.npy")

    for i, curve in enumerate(curves[:3]):
        curve_data_generator = CurveDataGenerator(
            curve=curve,
            rotation_factor=rotation_factor,
            sectioning_factor=sectioning_factor,
            sampling_factor=sampling_factor,
            multimodality_factor=multimodality_factor,
            sampling_points_count=sampling_points_count,
            supporting_points_count=supporting_points_count)

        negative_pairs = curve_data_generator.generate_negative_pairs()

        for negative_pair_index, negative_pair in enumerate(negative_pairs[:3]):
            fig, ax = plt.subplots(1, 2, figsize=(80, 40))

            for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
                label.set_fontsize(30)

            plot_dist(ax=ax[0], dist=negative_pair['dist'])
            plt.show()



    # arr = numpy.array([[2,4,6],[3,6,9],[5,10,15],[6,12,18],[7,14,21],[8,16,24]])
    #
    # arr = arr.reshape(-1,3,2)
    #
    # arr2 = arr.sum(axis=1)
    # arr3 = arr2.sum(axis=1)
    #
    # div = numpy.array([2,3,5])
    #
    # res = arr / div[:,None]
    #






    # a = numpy.arange(6,20)
    # it = numpy.nditer(a)
    # while not it.finished:
    #     bubu = it[0].item()
    #     it.iternext()





    # center = 17
    # delta = 6
    #
    # bibi = numpy.arange(center - delta, center + delta + 1)
    # bibi2 = numpy.mod(bibi, 20)
    #
    # itemindex = numpy.where(bibi2 == 17).item()
    # bla = bibi2[itemindex-3:itemindex+3+1]
    #
    # bla = [1,2,3,4,5,6,7,8]
    #
    # random.shuffle(bla)
    #
    # bla2 = itertools.combinations(bla, 3)
    #
    # for x in bla2:
    #     bla3 = x
    #
    # bla.extend([5,6,7])
    # j = 6






    # bins = 500
    # is_even = bins % 2 == 0
    # half_bins = math.floor(bins / 2)
    #
    # start, stop = -half_bins, half_bins
    # if not is_even:
    #     stop = stop + 1
    #
    # x = numpy.arange(start, stop)
    # x_lower, x_upper = x - 0.5, x + 0.5
    #
    # loc = 0.5
    # scale = 0.25
    # a = 0
    # b = 1
    # truncated_normal = scipy.stats.truncnorm(a=(a - loc) / scale, b=(b - loc) / scale, loc=loc, scale=scale)
    #
    # scales = truncated_normal.rvs(bins) * 10 * numpy.sqrt(bins)
    # locs = [0] * bins
    #
    # # blu = x_lower[:,None]
    #
    # dist_count = 1000
    #
    # cfd_lower = scipy.stats.norm.cdf(x_lower[:,None], loc=[0] * dist_count, scale=[15] * dist_count).transpose()
    # cfd_upper = scipy.stats.norm.cdf(x_upper[:,None], loc=[0] * dist_count, scale=[15] * dist_count).transpose()
    #
    # dist = cfd_upper - cfd_lower
    #
    # du = numpy.sum(dist, axis=1)
    #
    # dist2 = dist / du[:,None]







    # m = 5




    # numpy.seterr(all='raise')
    # x0 = numpy.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]])
    # padded_curve = numpy.pad(x0, ((2, 2), (0, 0)), 'wrap')
    #
    # bla = 2

    # x1 = numpy.array([1,2,3,4,5,6])
    # x2 = x1[:, numpy.newaxis]
    #
    # x3 = x0 * x2
    #
    # bla = 5

    # x1 = sklearn.preprocessing.normalize(x0, axis=1, norm='l2')
    #
    # x2 = numpy.pad(x1, ((1,1),(0,0)), 'wrap')
    # x3 = x2[1:-1]

    # bla = [1,2,3,7,8,12,20]
    # var = numpy.sqrt(numpy.var(bla))
    # mean = numpy.mean(bla)


    # dataset_generator = DatasetGenerator()
    # generated_curves = dataset_generator.generate_curves(dir_path="C:/deep-signature-data/images", plot_curves=False)
    # dataset_generator.save_curves(dir_path="C:/deep-signature-data/curves")
    # loaded_curves = dataset_generator.load_curves(file_path="C:/deep-signature-data/curves/curves.npy")
    # h = 5
    # a = numpy.array([[0., 0.], [0.3, 0.], [1.25, -0.1],
    #               [2.1, -0.9], [2.85, -2.3], [3.8, -3.95],
    #               [5., -5.75], [6.4, -7.8], [8.05, -9.9],
    #               [9.9, -11.6], [12.05, -12.85], [14.25, -13.7],
    #               [16.5, -13.8], [19.25, -13.35], [21.3, -12.2],
    #               [22.8, -10.5], [23.55, -8.15], [22.95, -6.1],
    #               [21.35, -3.95], [19.1, -1.9]])
    #
    # dx_dt = numpy.gradient(a[:, 0])
    # dy_dt = numpy.gradient(a[:, 1])
    # bla = numpy.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    #
    # bla2 = numpy.gradient(a, axis=0)
    #
    # h = 5


    # print(torch.__version__)
    #
    # t = torch.tensor([[2,1,2],[2,5,2],[1,1,4]], dtype=torch.float32)
    # norms = torch.norm(t)
    #
    #
    # t1 = torch.tensor([1,0,1,1,1,0,1,0,0,1], dtype=torch.float32)
    # t2 = torch.tensor([2,2,2,2,2,2,2,2,2,2], dtype=torch.float32)
    # t3 = 1 - t1
    # t4 = torch.tensor([2,-5,5,1,2,-1,-0.1,8,-7,6], dtype=torch.float32)
    #
    # bla = t1 * t2
    # bla2 = torch.sum(bla)
    #
    # bla3 = torch.max(torch.zeros_like(t1), t4)
    #
    #
    #
    #
    # metadata = numpy.load(file='./dataset/metadata.npy', allow_pickle=True)
    #
    # bla = metadata.item()
    #
    # curve = numpy.load(file='C:\\Users\\Roy\\Documents\\GitHub\\deep-signature\\dataset\\1268\\0\\8\\sample.npy', allow_pickle=True)
    #
    # g = 5
    #
    # arr1 = numpy.array([[1,2],[4,6],[7,5]])
    #
    # curve_torch = torch.from_numpy(curve)
    #
    # print(curve_torch[:, 0])
    #
    # # arr1_da = numpy.concatenate((arr1[-2:],arr1,arr1[:2]), axis=0)
    #
    # arr2 = numpy.array([[7,3],[9,6],[0,3]])
    # #
    # # print(numpy.concatenate((arr1, arr2), axis=1))
    #
    # # print(arr1)
    # bob = numpy.vstack((numpy.transpose(arr1), numpy.transpose(arr2)))
    # print(bob)
    # print(bob.shape)
    #
    # # print(arr[0])
    # # print(arr[1:4])
    # # print(arr[4:8])
