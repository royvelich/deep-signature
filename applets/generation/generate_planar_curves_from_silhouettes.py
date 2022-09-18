# python peripherals
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

# deep-signature
from deep_signature import manifolds
from deep_signature import utils
from deep_signature.base import SeedableObject


if __name__ == '__main__':
    utils.fix_random_seeds()

    parser = ArgumentParser()
    parser.add_argument('--images-base-dir-path', type=str)
    parser.add_argument('--curves-base-dir-path', type=str)
    parser.add_argument('--min-points-count', type=int)
    parser.add_argument('--max-points-count', type=int)
    parser.add_argument('--contour-level', type=float)
    parser.add_argument('--workers-count', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    SeedableObject.seed = args.seed

    silhouette_level_curves_generator = manifolds.SilhouetteLevelCurvesGenerator(
        images_base_dir_path=args.images_base_dir_path,
        curves_base_dir_path=args.curves_base_dir_path,
        min_points_count=args.min_points_count,
        max_points_count=args.max_points_count,
        contour_level=args.contour_level)

    silhouette_level_curves_generator.process(workers_count=args.workers_count)

