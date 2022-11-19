# python peripherals
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

# deep-signature
from deep_signature import manifolds
from deep_signature.manifolds.planar_curves import groups
from deep_signature.manifolds.planar_curves.generation import ShapeMatchingBenchmarkCurvesGenerator
from deep_signature.core import utils
from deep_signature.core.base import SeedableObject


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--curves-base-dir-path', type=str)
    parser.add_argument('--benchmark-base-dir-path', type=str)
    parser.add_argument('--sampling-ratios', nargs='+', type=float)
    parser.add_argument('--multimodalities', nargs='+', type=int)
    parser.add_argument('--groups', nargs='+', type=str)
    parser.add_argument('--min-det', type=float)
    parser.add_argument('--max-det', type=float)
    parser.add_argument('--min-cond', type=float)
    parser.add_argument('--max-cond', type=float)
    parser.add_argument('--fig-size', nargs='+', type=int)
    parser.add_argument('--point-size', type=float)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    utils.save_command_args(dir_path=args.benchmark_base_dir_path, args=args)

    SeedableObject.seed = args.seed

    group_list = []
    for group_name in args.groups:
        if group_name == 'euclidean':
            group_list.append(groups.EuclideanGroup())
        if group_name == 'equiaffine':
            group_list.append(groups.EquiaffineGroup(min_cond=args.min_cond, max_cond=args.max_cond))
        if group_name == 'similarity':
            group_list.append(groups.SimilarityGroup(min_det=args.min_det, max_det=args.max_det))
        if group_name == 'affine':
            group_list.append(groups.AffineGroup(min_det=args.min_det, max_det=args.max_det, min_cond=args.min_cond, max_cond=args.max_cond))

    shape_matching_benchmark_curves_generator = ShapeMatchingBenchmarkCurvesGenerator(
        num_workers=args.num_workers,
        curves_base_dir_path=args.curves_base_dir_path,
        benchmark_base_dir_path=args.benchmark_base_dir_path,
        sampling_ratios=args.sampling_ratios,
        multimodalities=args.multimodalities,
        groups=group_list,
        fig_size=tuple(args.fig_size),
        point_size=args.point_size)

    shape_matching_benchmark_curves_generator.process()
