import os
import scipy.io
import numpy
import re
import scipy.stats as ss
import matplotlib.pyplot as plt
import deep_signature.dist
import deep_signature.data
import gzip


if __name__ == '__main__':
    dataset_generator = deep_signature.data.DatasetGenerator()
    dataset_generator.load_raw_curves(dir_path='C:\\Users\\Roy\\OneDrive - Technion\\deep-signature-raw-data\\raw-data-new')
    dataset_generator.save(dir_path='./dataset', pairs_per_curve=200, rotation_factor=10, sampling_factor=10, sample_points=500)
