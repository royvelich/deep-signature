# python peripherals
import os

# numpy
import numpy

# skimage
import skimage.io
import skimage.color
import skimage.measure
from skimage import metrics

# deep signature
from deep_signature.data_manipulation import curve_processing
from deep_signature.linalg import transformations
from deep_signature.data_generation.curve_generation import LevelCurvesGenerator

# common
from utils import common as common_utils
from utils import evaluation as evaluation_utils
from utils import settings

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_graph(ax, x, y, linewidth=2, color='red', alpha=1, zorder=1, label=None):
    return ax.plot(x, y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder, label=label)


def plot_curve(ax, curve, linewidth=2, color='red', alpha=1, zorder=1, label=None):
    x = curve[:, 0]
    y = curve[:, 1]
    return plot_graph(ax=ax, x=x, y=y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder, label=label)


def plot_sample(ax, sample, color, zorder, point_size=10, alpha=1, x=None, y=None):
    if sample is not None:
        x = sample[:, 0]
        y = sample[:, 1]

    return ax.scatter(
        x=x,
        y=y,
        s=point_size,
        color=color,
        alpha=alpha,
        zorder=zorder)


def calculate_signature_curve(curve, transform_type, sampling_ratio, anchors_ratio, curvature_model, arclength_model, rng=None, plot=False, transform_curve=True):


    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
    # plot_curve(ax=ax, curve=curve, color='red', zorder=10)
    # plt.show()

    # curve = curve_processing.smooth_curve(
    #     curve=curve,
    #     iterations=2,
    #     window_length=5,
    #     poly_order=2)

    curve = curve_processing.center_curve(curve=curve)

    if transform_curve is True:
        transform = transformations.generate_random_transform_2d(
            transform_type=transform_type,
            min_cond=settings.arclength_min_cond_evaluation,
            max_cond=settings.arclength_max_cond_evaluation,
            min_det=settings.arclength_min_det_evaluation,
            max_det=settings.arclength_max_det_evaluation)

        transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
    else:
        transformed_curve = curve

    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
        plot_curve(ax=ax, curve=curve, color='red', zorder=10, linewidth=10)
        # plt.show()

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
        plot_curve(ax=ax, curve=transformed_curve, color='green', zorder=10, linewidth=10)
        plt.show()

    # transformed_curve = curve_processing.smooth_curve(
    #     curve=transformed_curve,
    #     iterations=50,
    #     window_length=99,
    #     poly_order=2)

    # if plot is True:
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
    #     plot_curve(ax=ax, curve=transformed_curve, color='red', zorder=10)
    #     plt.show()

    transformed_curve = curve_processing.center_curve(curve=transformed_curve)

    predicted_curve_invariants = evaluation_utils.predict_curve_invariants(
        curve=transformed_curve,
        arclength_model=arclength_model,
        curvature_model=curvature_model,
        sampling_ratio=sampling_ratio,
        anchors_ratio=anchors_ratio,
        neighborhood_supporting_points_count=settings.curvature_default_supporting_points_count,
        section_supporting_points_count=settings.arclength_default_supporting_points_count,
        rng=rng)
    signature_curve = predicted_curve_invariants['predicted_signature']
    return signature_curve


def calculate_hausdorff_distances(curve1, curve2):
    shift_distances = []
    for shift in range(curve2.shape[0]):
        shifted_curve2 = evaluation_utils.shift_signature_curve(curve=curve2, shift=shift)
        hausdorff_distance = evaluation_utils.calculate_hausdorff_distance(curve1=curve1, curve2=shifted_curve2, distance_type='euclidean')
        shift_distances.append(hausdorff_distance)
    return numpy.array(shift_distances)


if __name__ == '__main__':
    # rng = numpy.random.default_rng(seed=8)
    # indices1 = rng.choice(a=[1,2,3,4,5,6,7], size=3, replace=False)
    # indices2 = rng.choice(a=[1,2,3,4,5,6,7], size=3, replace=False)
    # indices3 = rng.choice(a=[1,2,3,4,5,6,7], size=3, replace=False)
    # indices4 = rng.choice(a=[1,2,3,4,5,6,7], size=3, replace=False)
    # indices5 = rng.choice(a=[1,2,3,4,5,6,7], size=3, replace=False)
    # indices6 = rng.choice(a=[1,2,3,4,5,6,7], size=3, replace=False)
    # indices7 = rng.choice(a=[1,2,3,4,5,6,7], size=3, replace=False)
    seed = 30
    rng = numpy.random.default_rng(seed=seed)
    numpy.random.seed(seed)
    sampling_ratio = 0.9
    anchors_ratio = None
    transform_type = 'affine'
    curvature_model, arclength_model = common_utils.load_models(transform_type=transform_type)

    # image_file_path = os.path.normpath('C:/Users/Roy/OneDrive - Technion/Thesis/1x/cats.png')
    # image = skimage.io.imread(image_file_path)
    # gray_image = skimage.color.rgb2gray(image)
    # contours = skimage.measure.find_contours(gray_image, 0.3)
    # contours.sort(key=lambda contour: contour.shape[0], reverse=True)
    # raw_curves = [contour for contour in contours if 1000 < contour.shape[0]]
    dataset_name = 'birds'
    curves = numpy.load(f'C:/deep-signature-data/level-curves/curves/test_raw/{dataset_name}.npy', allow_pickle=True)

    limit = None
    if dataset_name == 'butterflies':
        limit = 100
    elif dataset_name == 'cats':
        limit = 200
    elif dataset_name == 'dogs':
        limit = 700
    elif dataset_name == 'trees':
        limit = 200
    elif dataset_name == 'chickens':
        limit = 200
    elif dataset_name == 'birds':
        limit = 200
    elif dataset_name == 'leaves':
        limit = 100
    elif dataset_name == 'bears':
        limit = 200

    if limit is not None:
        curves = [curve for curve in curves if limit < curve.shape[0]]
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
    # plot_curve(ax=ax, curve=contours[0], color='red', zorder=10)
    # plt.show()
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
    # plot_curve(ax=ax, curve=contours[1], color='red', zorder=10)
    # plt.show()

    # curves = LevelCurvesGenerator.load_curves(dir_path=settings.level_curves_dir_path_test)
    # curves = [curve for curve in curves if 700 < curve.shape[0] < 1100]
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
    # plot_curve(ax=ax, curve=curves[0], color='red', zorder=10)
    # plt.show()
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
    # plot_curve(ax=ax, curve=curves[1], color='red', zorder=10)
    # plt.show()

    # raw_curves = raw_curves[:20]
    # curves = []
    # for i, curve in enumerate(curves):
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
    #     plot_curve(ax=ax, curve=curve, color='red', zorder=10)
    #     plt.show()

        # curve = curve_processing.smooth_curve(
        #     curve=curve,
        #     iterations=1,
        #     window_length=43,
        #     poly_order=2)
        # curves.append(curve)

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
        # plot_curve(ax=ax, curve=curve, color='red', zorder=10)
        # plt.show()

    correct = 0
    signatures = []
    for i, curve in enumerate(curves):
        signature_curve = calculate_signature_curve(
            curve=curve,
            transform_type=transform_type,
            sampling_ratio=sampling_ratio,
            anchors_ratio=anchors_ratio,
            curvature_model=curvature_model,
            arclength_model=arclength_model,
            rng=rng,
            transform_curve=False)

        signatures.append(signature_curve)

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
    # plot_sample(ax=ax, sample=signatures[0], color='red', zorder=10, point_size=300)
    # plt.show()
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 40))
    # shifted_curve = evaluation_utils.shift_signature_curve(curve=signatures[0], shift=90)
    # plot_sample(ax=ax, sample=shifted_curve, color='blue', zorder=10, point_size=300)
    # plt.show()

    distances = numpy.zeros((len(curves), len(curves)))
    for i, curve in enumerate(curves):
        anchor_signature_curve = calculate_signature_curve(
            curve=curve,
            transform_type=transform_type,
            sampling_ratio=sampling_ratio,
            anchors_ratio=anchors_ratio,
            curvature_model=curvature_model,
            arclength_model=arclength_model,
            rng=rng,
            plot=False)

        anchor_arc_length = anchor_signature_curve[-1, 0]
        for j, signature_curve in enumerate(signatures):
            current_arc_length = signature_curve[-1, 0]

            # shift_distances = calculate_hausdorff_distances(curve1=anchor_signature_curve, curve2=signature_curve)
            # # shift_distances2 = calculate_hausdorff_distances(curve1=signature_curve, curve2=anchor_signature_curve)
            # # distances[i, j] = numpy.min([numpy.min(shift_distances1), numpy.min(shift_distances2)])
            # distances[i, j] = numpy.min(shift_distances)

            # bla = numpy.abs(current_arc_length - anchor_arc_length) / anchor_arc_length
            # shift_distances = calculate_hausdorff_distances(curve1=anchor_signature_curve, curve2=signature_curve)
            # distances[i, j] = numpy.min(shift_distances) * bla

            if (numpy.abs(current_arc_length - anchor_arc_length) / anchor_arc_length) < 0.1:
                shift_distances = calculate_hausdorff_distances(curve1=anchor_signature_curve, curve2=signature_curve)
                distances[i, j] = numpy.min(shift_distances)
            else:
                distances[i, j] = numpy.inf

        curve_id = numpy.argmin(distances[i, :])
        if curve_id == i:
            correct = correct + 1
            print(f'curve #{i} correctly identified')
        else:
            print(f'curve #{i} failed to be identified')

    print(f'{correct} identifications out of {len(curves)}')
