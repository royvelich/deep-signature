# python peripherals
import os

# numpy
import numpy

# skimage
import skimage.io
import skimage.color
import skimage.measure

# deep signature
from deep_signature.data_manipulation import curve_processing
from deep_signature.linalg import transformations

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


if __name__ == '__main__':
    sampling_ratio = 0.7
    anchors_ratio = 1
    transform_type = 'euclidean'
    curvature_model, arclength_model = common_utils.load_models(transform_type=transform_type)

    image_file_path = os.path.normpath('C:/Users/Roy/OneDrive - Technion/Thesis/1x/cats.png')
    image = skimage.io.imread(image_file_path)
    gray_image = skimage.color.rgb2gray(image)
    contours = skimage.measure.find_contours(gray_image, 0.5)
    contours.sort(key=lambda contour: contour.shape[0], reverse=True)
    contours = [contour for contour in contours if contour.shape[0] > 1000]

    numpy.save('cats.npy', contours)

    comparison_curves = []
    for i, contour in enumerate(contours):
        if transform_type == 'euclidean':
            transform = transformations.generate_random_euclidean_transform_2d()
        elif transform_type == 'similarity':
            transform = transformations.generate_random_similarity_transform_2d()
        elif transform_type == 'equiaffine':
            transform = transformations.generate_random_equiaffine_transform_2d()
        elif transform_type == 'affine':
            transform = transformations.generate_random_affine_transform_2d()
        transformed_curve = curve_processing.transform_curve(curve=contour, transform=transform)
        comparison_curves.append({
            'curve': curve_processing.center_curve(curve=transformed_curve),
            'id': i
        })

    for i, contour in enumerate(contours):
        if transform_type == 'euclidean':
            transform = transformations.generate_random_euclidean_transform_2d()
        elif transform_type == 'similarity':
            transform = transformations.generate_random_similarity_transform_2d()
        elif transform_type == 'equiaffine':
            transform = transformations.generate_random_equiaffine_transform_2d()
        elif transform_type == 'affine':
            transform = transformations.generate_random_affine_transform_2d()

        transformed_anchor_curve = curve_processing.transform_curve(curve=contour, transform=transform)
        anchor_predicted_curve_invariants = evaluation_utils.predict_curve_invariants(
            curve=transformed_anchor_curve,
            arclength_model=arclength_model,
            curvature_model=curvature_model,
            sampling_ratio=sampling_ratio,
            anchors_ratio=anchors_ratio,
            neighborhood_supporting_points_count=settings.curvature_default_supporting_points_count,
            section_supporting_points_count=settings.arclength_default_supporting_points_count)

        anchor_signature_curve = anchor_predicted_curve_invariants['predicted_signature']

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40, 20))
        plot_sample(ax=ax, sample=anchor_signature_curve, color='red', zorder=10)
        plt.show()

        distances = numpy.zeros(len(contours))
        for comparison_curve_index, comparison_curve in enumerate(comparison_curves):
            predicted_curve_invariants = evaluation_utils.predict_curve_invariants(
                curve=comparison_curve['curve'],
                arclength_model=arclength_model,
                curvature_model=curvature_model,
                sampling_ratio=sampling_ratio,
                anchors_ratio=anchors_ratio,
                neighborhood_supporting_points_count=settings.curvature_default_supporting_points_count,
                section_supporting_points_count=settings.arclength_default_supporting_points_count)

            signature_curve = predicted_curve_invariants['predicted_signature']
            shift_distances = []
            for shift in range(signature_curve.shape[0]):
                shifted_signature_curve = evaluation_utils.shift_signature_curve(curve=signature_curve, shift=shift)
                hausdorff_distance = evaluation_utils.calculate_hausdorff_distance(curve1=anchor_signature_curve, curve2=shifted_signature_curve)
                shift_distances.append(hausdorff_distance[0])

            distances[comparison_curve_index] = numpy.min(shift_distances)

        bla = 5


    # for i in range(42, 52):
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    #     plot_curve(ax=ax, curve=contours[i])
    #     plt.show()
