# python peripherals
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

# deep-signature
from deep_signature.manifolds.planar_curves.generation import ImageLevelCurvesGenerator
from deep_signature.core import utils
from deep_signature.core.base import SeedableObject


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--images-base-dir-path', type=str)
    parser.add_argument('--curves-base-dir-path', type=str)
    parser.add_argument('--min-points-count', type=int)
    parser.add_argument('--max-points-count', type=int)
    parser.add_argument('--kernel-sizes', nargs='+', type=int)
    parser.add_argument('--contour-levels', nargs='+', type=float)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--min-equiaffine-std', type=float)
    parser.add_argument('--smoothing-iterations', type=int)
    parser.add_argument('--smoothing-window-length', type=int)
    parser.add_argument('--smoothing-poly-order', type=int)
    parser.add_argument('--flat-point-threshold', type=float)
    parser.add_argument('--max-flat-points-ratio', type=float)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    utils.save_command_args(dir_path=args.curves_base_dir_path, args=args)

    SeedableObject.seed = args.seed

    image_level_curves_generator = ImageLevelCurvesGenerator(
        num_workers=args.num_workers,
        images_base_dir_path=args.images_base_dir_path,
        curves_base_dir_path=args.curves_base_dir_path,
        min_points_count=args.min_points_count,
        max_points_count=args.max_points_count,
        kernel_sizes=args.kernel_sizes,
        contour_levels=args.contour_levels,
        flat_point_threshold=args.flat_point_threshold,
        max_flat_points_ratio=args.max_flat_points_ratio,
        min_equiaffine_std=args.min_equiaffine_std,
        smoothing_iterations=args.smoothing_iterations,
        smoothing_window_length=args.smoothing_window_length,
        smoothing_poly_order=args.smoothing_poly_order)

    image_level_curves_generator.process()
