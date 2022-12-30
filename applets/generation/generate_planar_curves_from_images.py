# python peripherals
from typing import List, Optional
from pathlib import Path

# deep-signature
from applets.core.utils import AppArgumentParser, init_app_tap
from deep_signature.manifolds.planar_curves.generation import ImageLevelCurvesGenerator


class GeneratePlanarCurvesFromImagesArgumentParser(AppArgumentParser):
    num_workers: int
    images_base_dir_path: Path
    min_points_count: int
    max_points_count: int
    contour_levels: List[float]
    kernel_sizes: List[int]
    flat_point_threshold: float
    max_flat_points_ratio: float
    min_equiaffine_std: float
    smoothing_iterations: int
    smoothing_window_length: int
    smoothing_poly_order: int
    max_tasks: Optional[int] = None


if __name__ == '__main__':
    parser = GeneratePlanarCurvesFromImagesArgumentParser().parse_args()
    results_dir_path = init_app_tap(parser=parser)

    image_level_curves_generator = ImageLevelCurvesGenerator(
        log_dir_path=results_dir_path,
        num_workers=parser.num_workers,
        images_base_dir_path=parser.images_base_dir_path,
        curves_base_dir_path=results_dir_path,
        min_points_count=parser.min_points_count,
        max_points_count=parser.max_points_count,
        kernel_sizes=parser.kernel_sizes,
        contour_levels=parser.contour_levels,
        flat_point_threshold=parser.flat_point_threshold,
        max_flat_points_ratio=parser.max_flat_points_ratio,
        min_equiaffine_std=parser.min_equiaffine_std,
        smoothing_iterations=parser.smoothing_iterations,
        smoothing_window_length=parser.smoothing_window_length,
        smoothing_poly_order=parser.smoothing_poly_order,
        max_tasks=parser.max_tasks)

    image_level_curves_generator.start()
    image_level_curves_generator.join()
