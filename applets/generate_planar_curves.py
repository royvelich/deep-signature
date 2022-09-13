# python peripherals
from argparse import ArgumentParser
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# deep-signature
from deep_signature import manifolds

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--images-base-dir-path', type=str)
    parser.add_argument('--curves-file-path', type=str)
    parser.add_argument('--sigmas', nargs='+', type=int)
    parser.add_argument('--min-contour-level', type=float)
    parser.add_argument('--max-contour-level', type=float)
    parser.add_argument('--min-points-count', type=int)
    parser.add_argument('--max-points-count', type=int)
    parser.add_argument('--curves-count', type=int)
    parser.add_argument('--workers-count', type=int)
    parser.add_argument('--queue-maxsize', type=int)

    parser.add_argument('--min-equiaffine-std', type=float)
    parser.add_argument('--smoothing-iterations', type=int)
    parser.add_argument('--smoothing-window-length', type=int)
    parser.add_argument('--smoothing-poly-order', type=int)

    parser.add_argument('--flat-point-threshold', type=float)
    parser.add_argument('--max-flat-points-ratio', type=float)

    args = parser.parse_args()

    level_curves_generator = manifolds.LevelCurvesGenerator(
        curves_count=args.curves_count,
        images_base_dir_path=args.images_base_dir_path,
        sigmas=args.sigmas,
        min_contour_level=args.min_contour_level,
        max_contour_level=args.max_contour_level,
        min_points_count=args.min_points_count,
        max_points_count=args.max_points_count,
        flat_point_threshold=args.flat_point_threshold,
        max_flat_points_ratio=args.max_flat_points_ratio,
        min_equiaffine_std=args.min_equiaffine_std,
        smoothing_iterations=args.smoothing_iterations,
        smoothing_window_length=args.smoothing_window_length,
        smoothing_poly_order=args.smoothing_poly_order)

    level_curves_generator.process(items_count=args.curves_count, workers_count=args.workers_count, queue_maxsize=args.curves_count)
    level_curves_generator.save(items_file_path=args.curves_file_path)
