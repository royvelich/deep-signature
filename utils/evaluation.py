# numpy
import numpy

# pytorch
import torch

# deep signature
from deep_signature.data_manipulation import curve_sampling
from deep_signature.data_manipulation import curve_processing
from deep_signature.linalg import transformations

# utils
from utils import common as common_utils
from utils import settings

# hausdorff
from hausdorff import hausdorff_distance

# https://stackoverflow.com/questions/36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar
from deep_signature.stats import discrete_distribution

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


def calculate_ks_by_index(curve, transform_type):
    true_ks = numpy.zeros([curve.shape[0], 2])
    true_ks[:, 0] = numpy.arange(curve.shape[0])

    if transform_type == 'euclidean':
        true_ks[:, 1] = curve_processing.calculate_euclidean_ks(curve=curve)
    elif transform_type == 'equiaffine':
        true_ks[:, 1] = 0
    elif transform_type == 'similarity':
        true_ks[:, 1] = 0
    elif transform_type == 'affine':
        true_ks[:, 1] = 0

    return true_ks


# -------------------
# PREDICTION ROUTINES
# -------------------
def predict_curvature_by_index(model, curve_neighborhoods, device='cuda', factor=-1):
    sampled_neighborhoods = curve_neighborhoods['sampled_neighborhoods']
    predicted_curvature = numpy.zeros([len(sampled_neighborhoods), 2])
    for point_index, sampled_neighborhood in enumerate(sampled_neighborhoods):
        for (indices, sample) in zip(sampled_neighborhood['indices'], sampled_neighborhood['samples']):
            sample = curve_processing.normalize_curve(curve=sample)
            curvature_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample), dim=0), dim=0).to(device)
            with torch.no_grad():
                predicted_curvature[point_index, 0] = point_index
                predicted_curvature[point_index, 1] = torch.squeeze(model(curvature_batch_data), dim=0).cpu().detach().numpy() * factor
    return predicted_curvature


def predict_arclength_by_index(model, curve, indices_pool, supporting_points_count, device='cuda', rng=None):
    indices_count = indices_pool.shape[0]
    predicted_arclength = numpy.zeros(indices_count)
    step = supporting_points_count - 1
    arclength_at_index = {}
    arclength_at_index[0] = 0
    sampled_curve = curve[indices_pool]
    for i in range(indices_count - 1):
        start_meta_index = i - step
        end_meta_index = i
        end_meta_index2 = end_meta_index + 1

        sampled_indices1 = numpy.mod(numpy.linspace(start=start_meta_index, stop=end_meta_index, num=supporting_points_count, dtype=int, endpoint=True), indices_count)
        sampled_indices2 = numpy.mod(numpy.linspace(start=start_meta_index, stop=end_meta_index2, num=supporting_points_count+1, dtype=int, endpoint=True), indices_count)
        sampled_indices2 = numpy.delete(sampled_indices2, supporting_points_count-1)

        sampled_section1 = sampled_curve[sampled_indices1]
        sampled_section2 = sampled_curve[sampled_indices2]

        sample1 = curve_processing.normalize_curve(curve=sampled_section1)
        sample2 = curve_processing.normalize_curve(curve=sampled_section2)

        sample1 = curve_processing.append_curve_moments(curve=sample1)
        sample2 = curve_processing.append_curve_moments(curve=sample2)

        arclength_batch_data1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample1), dim=0), dim=0).to(device)
        arclength_batch_data2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample2), dim=0), dim=0).to(device)

        with torch.no_grad():
            predicted_arclength[i + 1] = float(predicted_arclength[i] + numpy.abs(torch.squeeze(model(arclength_batch_data1.to(torch.float32)), dim=0).cpu().detach().numpy() - torch.squeeze(model(arclength_batch_data2.to(torch.float32)), dim=0).cpu().detach().numpy()))

    indices = numpy.array(list(range(predicted_arclength.shape[0])))
    values = predicted_arclength
    return numpy.vstack((indices, values)).transpose()


def predict_differential_invariants_by_index(model, curve_neighborhoods, device='cuda'):
    sampled_neighborhoods = curve_neighborhoods['sampled_neighborhoods']
    predicted_differential_invariants = numpy.zeros([len(sampled_neighborhoods), 2])
    for point_index, sampled_neighborhood in enumerate(sampled_neighborhoods):
        for (indices, sample) in zip(sampled_neighborhood['indices'], sampled_neighborhood['samples']):
            sample = curve_processing.normalize_curve(curve=sample)
            curvature_batch_data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(sample), dim=0), dim=0).to(device)
            with torch.no_grad():
                # diff_invariants_net = model._model
                # if point_index == 0:
                #     print(diff_invariants_net(curvature_batch_data).cpu().detach().numpy())
                # j = 5

                # backbone_res = model._model.backbone(curvature_batch_data)
                # proj_res = model._model.projection_head(backbone_res)
                #
                # backbone_momentum_res = model._model.backbone_momentum(curvature_batch_data)
                # proj_momentum_res = model._model.projection_head_momentum(backbone_momentum_res)

                res = proj_res = model(curvature_batch_data)

                diff_invariants = torch.squeeze(torch.squeeze(res, dim=0), dim=0).cpu().detach().numpy()
                # diff_invariants2 = torch.squeeze(torch.squeeze(proj_momentum_res, dim=0), dim=0).cpu().detach().numpy()
                predicted_differential_invariants[point_index, 0] = diff_invariants[0]
                predicted_differential_invariants[point_index, 1] = diff_invariants[1]
                # predicted_differential_invariants[point_index, 2] = diff_invariants[2]
                # predicted_differential_invariants[point_index, 3] = diff_invariants[3]
                # predicted_differential_invariants[point_index, 4] = diff_invariants[4]
                # predicted_differential_invariants[point_index, 5] = diff_invariants[5]
                # predicted_differential_invariants[point_index, 6] = diff_invariants[6]
                # predicted_differential_invariants[point_index, 7] = diff_invariants[7]
    return predicted_differential_invariants


def predict_curve_invariants(curve, models, sampling_ratio, neighborhood_supporting_points_count, section_supporting_points_count, indices_shift=0, device='cuda', rng=None):
    curve_points_count = curve.shape[0]
    sampling_points_count = int(sampling_ratio * curve_points_count)
    dist = discrete_distribution.random_discrete_dist(bins=curve_points_count, multimodality=settings.default_multimodality_evaluation, max_density=1, count=1)[0]
    indices_pool = discrete_distribution.sample_discrete_dist(dist=dist, sampling_points_count=sampling_points_count)
    # modified_indices_pool = common_utils.insert_sorted(indices_pool, numpy.array([reference_index]))
    # meta_reference_index = int(numpy.where(modified_indices_pool == reference_index)[0])
    indices_pool = numpy.roll(indices_pool, shift=indices_shift, axis=0)

    predicted_arclength = None
    arclength_model = models['arclength']
    if models['arclength'] is not None:
        arclength_model.eval()
        predicted_arclength = predict_arclength_by_index(
            model=arclength_model,
            curve=curve,
            indices_pool=indices_pool,
            supporting_points_count=section_supporting_points_count,
            device=device,
            rng=rng)

    curve_neighborhoods = extract_curve_neighborhoods(
        curve=curve,
        indices_pool=indices_pool,
        supporting_points_count=neighborhood_supporting_points_count)

    predicted_curvature = None
    curvature_model = models['curvature']
    if curvature_model is not None:
        curvature_model.eval()
        predicted_curvature = predict_curvature_by_index(
            model=curvature_model,
            curve_neighborhoods=curve_neighborhoods,
            device=device)

    predicted_differential_invariants = None
    differential_invariants_model = models['diff_inv']
    if differential_invariants_model is not None:
        differential_invariants_model.eval()
        predicted_differential_invariants = predict_differential_invariants_by_index(
            model=differential_invariants_model,
            curve_neighborhoods=curve_neighborhoods,
            device=device)

    # if anchor_indices is not None:
    #     predicted_arclength_with_anchors = predict_arclength_by_index(
    #         model=arclength_model,
    #         curve=curve,
    #         indices_pool=indices_pool,
    #         supporting_points_count=section_supporting_points_count,
    #         anchor_indices=anchor_indices)
    #
    #     curve_neighborhoods_with_anchors = extract_curve_neighborhoods(
    #         curve=curve,
    #         indices_pool=indices_pool,
    #         supporting_points_count=neighborhood_supporting_points_count,
    #         anchor_indices=anchor_indices)
    #
    #     predicted_curvature_with_anchors = predict_curvature_by_index(
    #         model=curvature_model,
    #         curve_neighborhoods=curve_neighborhoods_with_anchors)

    if predicted_curvature is not None and predicted_arclength is not None:
        predicted_signature = numpy.zeros((indices_pool.shape[0], 2))
        predicted_signature[:, 0] = predicted_arclength[:, 1]
        predicted_signature[:, 1] = predicted_curvature[:, 1]
    else:
        predicted_signature = None

    sampled_curve = curve[indices_pool]
    anchors = curve[indices_pool]

    return {
        'predicted_arclength': predicted_arclength,
        # 'predicted_arclength_with_anchors': predicted_arclength_with_anchors if anchor_indices is not None else None,
        'predicted_curvature': predicted_curvature,
        'predicted_differential_invariants': predicted_differential_invariants,
        # 'predicted_curvature_with_anchors': predicted_curvature_with_anchors if anchor_indices is not None else None,
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

    true_ks = calculate_ks_by_index(
        curve=curve,
        transform_type=transform_type)

    return {
        'true_arclength': true_arclength,
        'true_curvature': true_curvature,
        'true_ks': true_ks
    }

# --------------------------
# RECORD GENERATION ROUTINES
# --------------------------
def generate_curve_records(models, curves, factor_extraction_curves, transform_type, comparison_curves_count, sampling_ratio, anchors_ratio, neighborhood_supporting_points_count, section_supporting_points_count, do_not_transform_first=True, do_not_downsample_first=False, rotation=True, first_curve_index=None):
    curve_records = []
    for curve_index, curve in enumerate(curves):
        curve = curve_processing.enforce_cw(curve=curve)

        comparison_curves = []
        for i in range(comparison_curves_count):
            current_curve = curve
            if i == 0 and first_curve_index is not None:
                current_curve = curve_processing.enforce_cw(curve=curves[first_curve_index])

            if i == 0:
                transform = transformations.generate_random_transform_2d_evaluation(transform_type=transform_type, rotation=rotation)
            else:
                transform = transformations.generate_random_transform_2d_evaluation(transform_type=transform_type)

            if i == 0 and do_not_transform_first is True:
                transformed_curve = current_curve
            else:
                transformed_curve = curve_processing.transform_curve(curve=current_curve, transform=transform)

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
                models=models,
                sampling_ratio=1 if (do_not_downsample_first is True and i == 0) else sampling_ratio,
                neighborhood_supporting_points_count=neighborhood_supporting_points_count,
                section_supporting_points_count=section_supporting_points_count)

            true_curve_invariants = calculate_curve_invariants(
                curve=comparison_curve,
                transform_type=transform_type)

            arclength_comparison = {
                'true_arclength': true_curve_invariants['true_arclength'],
                'predicted_arclength': predicted_curve_invariants['predicted_arclength'],
            }

            curvature_comparison = {
                'curve_neighborhoods': predicted_curve_invariants['curve_neighborhoods'],
                'true_curvature': true_curve_invariants['true_curvature'],
                'true_ks': true_curve_invariants['true_ks'],
                'predicted_curvature': predicted_curve_invariants['predicted_curvature'],
            }

            differential_invariants_comparison = {
                'curve_neighborhoods': predicted_curve_invariants['curve_neighborhoods'],
                'predicted_differential_invariants': predicted_curve_invariants['predicted_differential_invariants'],
                'true_curvature': true_curve_invariants['true_curvature'],
                'true_ks': true_curve_invariants['true_ks']
            }

            curve_record['comparisons'].append({
                'curve': comparison_curve,
                'sampled_curve': predicted_curve_invariants['sampled_curve'],
                'anchor_indices': anchor_indices,
                'anchors': predicted_curve_invariants['anchors'],
                'dist': predicted_curve_invariants['dist'],
                'arclength_comparison': arclength_comparison,
                'curvature_comparison': curvature_comparison,
                'differential_invariants_comparison': differential_invariants_comparison,
                'predicted_signature': predicted_curve_invariants['predicted_signature'],
            })

        curve_records.append(curve_record)

    arclength_model = models['arclength']
    if len(factor_extraction_curves) > 0 and arclength_model is not None:
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
                supporting_points_count=section_supporting_points_count)

            factor = numpy.mean(true_arclength[1:, 1] / predicted_arclength[1:, 1])
            factors.append(factor)

        if transform_type != 'affine' and transform_type != 'similarity':
            factor = numpy.mean(numpy.array(factors))
            for curve_record in curve_records:
                for comparison in curve_record['comparisons']:
                    comparison['arclength_comparison']['predicted_arclength'][:, 1] *= factor
                    comparison['predicted_signature'][:, 0] *= factor

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
