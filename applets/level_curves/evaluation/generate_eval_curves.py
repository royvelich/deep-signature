# python peripherals
import os
from argparse import ArgumentParser
from os import walk
from pathlib import Path

# numpy
import numpy

# skimage
import skimage.io
import skimage.color
import skimage.measure

# common
from utils import common as common_utils
from utils import settings

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--images_dir", dest="images_dir", type=str)
    parser.add_argument("--curves_dir", dest="curves_dir", type=str)
    args = parser.parse_args()

    image_file_paths = []
    for (dir_path, dir_names, file_names) in walk(args.images_dir):
        for file_name in file_names:
            image_file_paths.append(os.path.normpath(os.path.join(dir_path, file_name)))
        break

    for image_file_path in image_file_paths:
        stem = Path(image_file_path).stem
        image = skimage.io.imread(image_file_path)
        gray_image = skimage.color.rgb2gray(image)
        curves = skimage.measure.find_contours(gray_image, 0.3)
        curves.sort(key=lambda curve: curve.shape[0], reverse=True)
        curves_file_path = os.path.normpath(os.path.join(args.curves_dir, f'{stem}.npy'))
        numpy.save(curves_file_path, curves)
