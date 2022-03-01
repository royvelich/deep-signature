# python peripherals
import random
import pathlib
import os

# scipy
import scipy

# numpy
import numpy

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.ticker as ticker
import matplotlib.lines

# pytorch
import torch

# skimage
from skimage import metrics

# pandas
import pandas

# ipython
from IPython.display import display, HTML

# deep signature
from deep_signature.data_manipulation import curve_sampling
from deep_signature.data_manipulation import curve_processing
from deep_signature.linalg import euclidean_transform
from deep_signature.linalg import affine_transform
from deep_signature.linalg import transformations

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines

# plotly
from plotly.subplots import make_subplots
from plotly import graph_objects

# utils
from utils import common as common_utils
from utils import settings

# hausdorff
from hausdorff import hausdorff_distance

# https://stackoverflow.com/questions/36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar
from deep_signature.stats import discrete_distribution

# ---------------------
# GROUND TRUTH ROUTINES
# ---------------------
def calculate_arclength_by_index(curve, anchor_indices, transform_type, modifier=None):
    true_arclength = numpy.zeros(curve.shape[0])
    if transform_type == 'euclidean':
        true_arclength = curve_processing.calculate_euclidean_arclength(curve=curve)
    elif transform_type == 'equiaffine':
        if modifier == 'calabi':
            true_arclength = curve_processing.calculate_equiaffine_arclength(curve=curve)
        else:
            true_arclength = curve_processing.calculate_equiaffine_arclength_by_euclidean_metrics(curve=curve)

    indices = numpy.array(list(range(anchor_indices.shape[0])))
    values = true_arclength[anchor_indices]
    return numpy.vstack((indices, values)).transpose()


def calculate_curvature_by_index(curve, transform_type):
    true_curvature = numpy.zeros([curve.shape[0], 2])
    true_curvature[:, 0] = numpy.arange(curve.shape[0])

    if transform_type == 'euclidean':
        true_curvature[:, 1] = curve_processing.calculate_euclidean_curvature(curve=curve)
    elif transform_type == 'equiaffine':
        true_curvature[:, 1] = curve_processing.calculate_equiaffine_curvature(curve=curve)
    elif transform_type == 'similarity':
        true_curvature[:, 1] = 0
    elif transform_type == 'affine':
        true_curvature[:, 1] = 0

    return true_curvature


# -------------------
# PREDICTION ROUTINES
# -------------------
def predict_curvature_by_index(model, curve_neighborhoods, factor=-1):
    sampled_neighborhoods = curve_neighborhoods['sampled_neighborhoods']
    predicted_curvature = numpy.zeros([len(sampled_neighborhoods), 2])
    for point_index, sampled_neighborhood in enumerate(sampled_neighborhoods):
        for (indices, sample) in zip(sampled_neighborhood['indices'], sampled_neighborhood['samples']):
            sample = curve_processing.normalize_curve(curve=sample)
            curvature_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample).double(), dim=0), dim=0).cuda()
            with torch.no_grad():
                predicted_curvature[point_index, 0] = point_index
                predicted_curvature[point_index, 1] = torch.squeeze(model(curvature_batch_data), dim=0).cpu().detach().numpy() * factor
    return predicted_curvature


def predict_arclength_by_index(model, curve, indices_pool, supporting_points_count, anchor_indices=None, rng=None):
    anchor_indices = anchor_indices if anchor_indices is not None else indices_pool
    predicted_arclength = numpy.zeros(anchor_indices.shape[0])
    step = supporting_points_count - 1
    arclength_at_index = {}
    arclength_at_index[anchor_indices[0]] = 0
    for i, anchor_index in enumerate(anchor_indices[1:]):
        modified_indices_pool = common_utils.insert_sorted(indices_pool, numpy.array([anchor_indices[0], anchor_index]))
        sampled_curve = curve[modified_indices_pool]
        anchor_meta_index = int(numpy.where(modified_indices_pool == anchor_index)[0])
        max_index = max(arclength_at_index, key=arclength_at_index.get)
        # try:
        max_meta_index = int(numpy.where(modified_indices_pool == max_index)[0])
        # except:
        #     bla = 5
        anchor_arclength = arclength_at_index[max_index]
        for meta_index in range(max_meta_index, anchor_meta_index):
            start_meta_index = meta_index - step
            end_meta_index = meta_index
            end_meta_index2 = end_meta_index + 1

            sampled_indices1 = curve_sampling.sample_curve_section_indices(
                curve=sampled_curve,
                start_point_index=start_meta_index,
                end_point_index=end_meta_index,
                multimodality=settings.arclength_default_multimodality,
                supporting_points_count=supporting_points_count,
                uniform=True,
                rng=rng)

            sampled_indices2 = curve_sampling.sample_curve_section_indices(
                curve=sampled_curve,
                start_point_index=start_meta_index,
                end_point_index=end_meta_index2,
                multimodality=settings.arclength_default_multimodality,
                supporting_points_count=supporting_points_count,
                uniform=True,
                rng=rng)

            sampled_section1 = sampled_curve[sampled_indices1]
            sampled_section2 = sampled_curve[sampled_indices2]

            sample1 = curve_processing.normalize_curve(curve=sampled_section1)
            sample2 = curve_processing.normalize_curve(curve=sampled_section2)

            arclength_batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1).double(), dim=0), dim=0).cuda()
            arclength_batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2).double(), dim=0), dim=0).cuda()

            with torch.no_grad():
                anchor_arclength = float(anchor_arclength + numpy.abs(torch.squeeze(model(arclength_batch_data1), dim=0).cpu().detach().numpy() - torch.squeeze(model(arclength_batch_data2), dim=0).cpu().detach().numpy()))

            current_index = modified_indices_pool[end_meta_index2]
            # if current_index != anchor_index:
            arclength_at_index[current_index] = anchor_arclength

        predicted_arclength[i+1] = anchor_arclength

    indices = numpy.array(list(range(predicted_arclength.shape[0])))
    values = predicted_arclength
    return numpy.vstack((indices, values)).transpose()


def predict_curve_invariants(curve, arclength_model, curvature_model, sampling_ratio, anchors_ratio, neighborhood_supporting_points_count, section_supporting_points_count, reference_index=0, multimodality=30, anchor_indices=None, rng=None):
    curve_points_count = curve.shape[0]
    sampling_points_count = int(sampling_ratio * curve_points_count)
    dist = discrete_distribution.random_discrete_dist(bins=curve_points_count, multimodality=settings.curvature_default_multimodality, max_density=1, count=1)[0]
    indices_pool = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=sampling_points_count)
    modified_indices_pool = common_utils.insert_sorted(indices_pool, numpy.array([0]))

    predicted_arclength = predict_arclength_by_index(
        model=arclength_model,
        curve=curve,
        indices_pool=modified_indices_pool,
        supporting_points_count=section_supporting_points_count,
        rng=rng)

    curve_neighborhoods = extract_curve_neighborhoods(
        curve=curve,
        indices_pool=modified_indices_pool,
        supporting_points_count=neighborhood_supporting_points_count)

    predicted_curvature = predict_curvature_by_index(
        model=curvature_model,
        curve_neighborhoods=curve_neighborhoods)

    if anchor_indices is not None:
        predicted_arclength_with_anchors = predict_arclength_by_index(
            model=arclength_model,
            curve=curve,
            indices_pool=indices_pool,
            supporting_points_count=section_supporting_points_count,
            anchor_indices=anchor_indices)

        curve_neighborhoods_with_anchors = extract_curve_neighborhoods(
            curve=curve,
            indices_pool=modified_indices_pool,
            supporting_points_count=neighborhood_supporting_points_count,
            anchor_indices=anchor_indices)

        predicted_curvature_with_anchors = predict_curvature_by_index(
            model=curvature_model,
            curve_neighborhoods=curve_neighborhoods_with_anchors)

    predicted_signature = numpy.zeros((modified_indices_pool.shape[0], 2))
    predicted_signature[:, 0] = predicted_arclength[:, 1]
    predicted_signature[:, 1] = predicted_curvature[:, 1]

    sampled_curve = curve[indices_pool]
    anchors = curve[indices_pool]

    return {
        'predicted_arclength': predicted_arclength,
        'predicted_arclength_with_anchors': predicted_arclength_with_anchors if anchor_indices is not None else None,
        'predicted_curvature': predicted_curvature,
        'predicted_curvature_with_anchors': predicted_curvature_with_anchors if anchor_indices is not None else None,
        'predicted_signature': predicted_signature,
        'curve_neighborhoods': curve_neighborhoods,
        'sampling_points_count': sampling_points_count,
        'sampled_curve': sampled_curve,
        'anchors': anchors,
        'dist': dist
    }


def calculate_curve_invariants(curve, transform_type, anchor_indices=None):
    curve_points_count = curve.shape[0]
    if anchor_indices is None:
        anchor_indices = numpy.linspace(start=0, stop=curve_points_count, num=curve_points_count, endpoint=False, dtype=int)

    true_arclength = calculate_arclength_by_index(
        curve=curve,
        anchor_indices=anchor_indices,
        transform_type=transform_type)

    true_curvature = calculate_curvature_by_index(
        curve=curve,
        transform_type=transform_type)

    return {
        'true_arclength': true_arclength,
        'true_curvature': true_curvature,
    }

# --------------------------
# RECORD GENERATION ROUTINES
# --------------------------
def generate_curve_records(arclength_model, curvature_model, curves, factor_extraction_curves, transform_type, comparison_curves_count, sampling_ratio, anchors_ratio, neighborhood_supporting_points_count, section_supporting_points_count):
    curve_records = []
    for curve_index, curve in enumerate(curves):
        curve = curve_processing.enforce_cw(curve=curve)

        comparison_curves = []
        for i in range(comparison_curves_count):
            if transform_type == 'euclidean':
                transform = transformations.generate_random_euclidean_transform_2d()
            elif transform_type == 'similarity':
                transform = transformations.generate_random_similarity_transform_2d()
            elif transform_type == 'equiaffine':
                transform = transformations.generate_random_equiaffine_transform_2d()
            elif transform_type == 'affine':
                transform = transformations.generate_random_affine_transform_2d()

            if i == 0:
                transformed_curve = curve_processing.transform_curve(curve=curve, transform=transform)
            else:
                transformed_curve = curve

            comparison_curves.append(curve_processing.center_curve(curve=transformed_curve))

        curve_record = {
            'curve': curve_processing.center_curve(curve=curve),
            'comparisons': []
        }

        if anchors_ratio is not None:
            anchor_indices = numpy.linspace(start=0, stop=curve.shape[0], num=int(anchors_ratio * curve.shape[0]), endpoint=False, dtype=int)
        else:
            anchor_indices = None

        for i, comparison_curve in enumerate(comparison_curves):
            predicted_curve_invariants = predict_curve_invariants(
                curve=comparison_curve,
                arclength_model=arclength_model,
                curvature_model=curvature_model,
                sampling_ratio=sampling_ratio,
                anchors_ratio=anchors_ratio,
                neighborhood_supporting_points_count=neighborhood_supporting_points_count,
                section_supporting_points_count=section_supporting_points_count,
                anchor_indices=anchor_indices)

            true_curve_invariants = calculate_curve_invariants(
                curve=comparison_curve,
                transform_type=transform_type)

            arclength_comparison = {
                'true_arclength': true_curve_invariants['true_arclength'],
                'predicted_arclength': predicted_curve_invariants['predicted_arclength'],
                'predicted_arclength_with_anchors': predicted_curve_invariants['predicted_arclength_with_anchors']
            }

            curvature_comparison = {
                'curve_neighborhoods': predicted_curve_invariants['curve_neighborhoods'],
                'true_curvature': true_curve_invariants['true_curvature'],
                'predicted_curvature': predicted_curve_invariants['predicted_curvature'],
                'predicted_curvature_with_anchors': predicted_curve_invariants['predicted_curvature_with_anchors']
            }

            curve_record['comparisons'].append({
                'curve': comparison_curve,
                'sampled_curve': predicted_curve_invariants['sampled_curve'],
                'anchor_indices': anchor_indices,
                'anchors': predicted_curve_invariants['anchors'],
                'dist': predicted_curve_invariants['dist'],
                'arclength_comparison': arclength_comparison,
                'curvature_comparison': curvature_comparison,
                'predicted_signature': predicted_curve_invariants['predicted_signature'],
            })

        curve_records.append(curve_record)

    factors = []
    for curve_index, curve in enumerate(factor_extraction_curves):
        all_indices = numpy.array(list(range(curve.shape[0])))
        true_arclength = calculate_arclength_by_index(
            curve=curve,
            anchor_indices=all_indices,
            transform_type=transform_type)

        predicted_arclength = predict_arclength_by_index(
            model=arclength_model,
            curve=curve,
            indices_pool=all_indices,
            anchor_indices=all_indices,
            supporting_points_count=section_supporting_points_count)

        factor = numpy.mean(true_arclength[1:, 1] / predicted_arclength[1:, 1])
        factors.append(factor)

    if transform_type != 'affine' and transform_type != 'similarity':
        factor = numpy.mean(numpy.array(factors))
        for curve_record in curve_records:
            for comparison in curve_record['comparisons']:
                comparison['arclength_comparison']['predicted_arclength'][:, 1] *= factor
                if comparison['arclength_comparison']['predicted_arclength_with_anchors'] is not None:
                    comparison['arclength_comparison']['predicted_arclength_with_anchors'][:, 1] *= factor

    return curve_records


def extract_curve_neighborhoods(curve, indices_pool, supporting_points_count, anchor_indices=None):
    sampled_neighborhoods = []
    anchor_indices = anchor_indices if anchor_indices is not None else indices_pool
    for anchor_index in anchor_indices:
        sampled_indices = curve_sampling.sample_curve_neighborhood_indices(
            center_point_index=anchor_index,
            indices_pool=indices_pool,
            supporting_points_count=supporting_points_count)

        sampled_neighborhood = {
            'indices': [sampled_indices],
            'samples': [curve[sampled_indices]]
        }

        sampled_neighborhoods.append(sampled_neighborhood)

    return {
        'sampled_neighborhoods': sampled_neighborhoods,
        'curve': curve
    }


# ----------------
# METRICS ROUTINES
# ----------------
def calculate_signature_metrics(curve_records):
    curvature_offsets = numpy.array([])
    arclength_offsets = numpy.array([])
    for i, curve_record in enumerate(curve_records):
        comparisons = curve_record['comparisons']
        arclength_comparison_ref = comparisons[0]['arclength_comparison']
        curvature_comparison_ref = comparisons[0]['curvature_comparison']
        predicted_arclength_ref = arclength_comparison_ref['predicted_arclength'][1:, 1].squeeze()
        predicted_curvature_ref = curvature_comparison_ref['predicted_curvature'][:, 1].squeeze()
        for comparison in comparisons[1:]:
            arclength_comparison = comparison['arclength_comparison']
            curvature_comparison = comparison['curvature_comparison']
            predicted_arclength = arclength_comparison['predicted_arclength'][1:, 1].squeeze()
            predicted_curvature = curvature_comparison['predicted_curvature'][:, 1].squeeze()
            arclength_offset = numpy.abs(predicted_arclength - predicted_arclength_ref) / numpy.abs(predicted_arclength_ref)
            curvature_offset = numpy.abs(predicted_curvature - predicted_curvature_ref) / numpy.abs(predicted_curvature_ref)
            arclength_offsets = numpy.concatenate((arclength_offsets, arclength_offset))
            curvature_offsets = numpy.concatenate((curvature_offsets, curvature_offset))

    curvature_offsets.sort()
    print(curvature_offsets)

    return {
        'arclength_offset_mean': numpy.mean(arclength_offsets),
        'arclength_offset_std': numpy.std(arclength_offsets),
        'curvature_offset_mean': numpy.mean(curvature_offsets),
        'curvature_offset_std': numpy.std(curvature_offsets),
        'curvature_offset_min': numpy.min(curvature_offsets),
        'curvature_offset_max': numpy.max(curvature_offsets),
    }

def shift_signature_curve(curve, shift):
    arclength_diff = numpy.diff(a=curve[:, 0], axis=0)
    shifted_arclength_diff = numpy.roll(a=arclength_diff, shift=shift, axis=0)
    shifted_arclength = numpy.cumsum(shifted_arclength_diff)
    shifted_arclength = numpy.insert(shifted_arclength, 0, 0)
    shifted_curvature = numpy.roll(a=curve[:, 1], shift=shift, axis=0)
    return numpy.array([shifted_arclength, shifted_curvature]).transpose()

def calculate_hausdorff_distance(curve1, curve2, distance_type):
    # hausdorff_distance = scipy.spatial.distance.directed_hausdorff(u=curve1, v=curve2)
    # hausdorff_distance2 = metrics.hausdorff_distance(image0=curve1, image1=curve2)
    # return hausdorff_distance[0]
    distance = hausdorff_distance(XA=curve1, XB=curve2, distance=distance_type)
    return distance
