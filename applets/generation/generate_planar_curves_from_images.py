# python peripherals
from argparse import ArgumentParser
from typing import List, Optional
import warnings
from pathlib import Path
from datetime import datetime

# deep-signature
from deep_signature.manifolds.planar_curves.generation import ImageLevelCurvesGenerator
from deep_signature.core import utils
from deep_signature.core.base import SeedableObject

# tap
from tap import Tap

SeedableObject.set_seed(seed=42)


class GeneratePlanarCurvesFromImagesArgumentParser(Tap):
    name: str = 'GeneratePlanarCurvesFromImages'
    seed: int
    num_workers: int
    images_base_dir_path: Path
    curves_base_dir_path: Path
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
    curves_file_name: str = 'curves.npy'
    max_image_files: Optional[int] = None


if __name__ == '__main__':
    image_level_curves_generator_parser = GeneratePlanarCurvesFromImagesArgumentParser().parse_args()

    datetime_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    curves_base_dir_path = image_level_curves_generator_parser.curves_base_dir_path / Path(datetime_string)

    utils.save_tap(
        dir_path=curves_base_dir_path,
        typed_argument_parser=image_level_curves_generator_parser)

    SeedableObject.seed = image_level_curves_generator_parser.seed

    image_level_curves_generator = ImageLevelCurvesGenerator(
        name=image_level_curves_generator_parser.name,
        log_dir_path=curves_base_dir_path,
        num_workers=image_level_curves_generator_parser.num_workers,
        images_base_dir_path=image_level_curves_generator_parser.images_base_dir_path,
        curves_base_dir_path=curves_base_dir_path,
        min_points_count=image_level_curves_generator_parser.min_points_count,
        max_points_count=image_level_curves_generator_parser.max_points_count,
        kernel_sizes=image_level_curves_generator_parser.kernel_sizes,
        contour_levels=image_level_curves_generator_parser.contour_levels,
        flat_point_threshold=image_level_curves_generator_parser.flat_point_threshold,
        max_flat_points_ratio=image_level_curves_generator_parser.max_flat_points_ratio,
        min_equiaffine_std=image_level_curves_generator_parser.min_equiaffine_std,
        smoothing_iterations=image_level_curves_generator_parser.smoothing_iterations,
        smoothing_window_length=image_level_curves_generator_parser.smoothing_window_length,
        smoothing_poly_order=image_level_curves_generator_parser.smoothing_poly_order)

    image_level_curves_generator.start()
    image_level_curves_generator.join()
