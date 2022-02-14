# python peripherals
import os

# skimage
import skimage.io
import skimage.color
import skimage.measure

# deep signature
from deep_signature.data_manipulation import curve_processing
from deep_signature.linalg import transformations

# common
import utils


def plot_graph(ax, x, y, linewidth=2, color='red', alpha=1, zorder=1, label=None):
    return ax.plot(x, y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder, label=label)


def plot_curve(ax, curve, linewidth=2, color='red', alpha=1, zorder=1, label=None):
    x = curve[:, 0]
    y = curve[:, 1]
    return plot_graph(ax=ax, x=x, y=y, linewidth=linewidth, color=color, alpha=alpha, zorder=zorder, label=label)

if __name__ == '__main__':
    transform_type = 'euclidean'
    curvature_model, arclength_model = utils.common.load_models(transform_type=transform_type)

    image_file_path = os.path.normpath('C:/Users/Roy/OneDrive - Technion/Thesis/3x/cats@3x.png')
    image = skimage.io.imread(image_file_path)
    gray_image = skimage.color.rgb2gray(image)
    contours = skimage.measure.find_contours(gray_image, 0.5)
    contours.sort(key=lambda contour: contour.shape[0], reverse=True)
    contours = [contour for contour in contours if contour.shape[0] > 1000]

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

    for comparison_curve in comparison_curves:

    # for i in range(42, 52):
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    #     plot_curve(ax=ax, curve=contours[i])
    #     plt.show()
