# python peripherals
from typing import List, Optional
from pathlib import Path

# deep-signature
from applets.core.utils import AppArgumentParser, init_app
from deep_signature.manifolds.planar_curves.generation import CirclesGenerator
from deep_signature.core.base import SeedableObject


class GeneratePlanarCurvesFromCirclesArgumentParser(AppArgumentParser):
    num_workers: int
    circles_count: int
    min_radius: float
    max_radius: float
    min_sampling_density: float
    max_sampling_density: float
    max_tasks: Optional[int] = None


if __name__ == '__main__':
    parser = GeneratePlanarCurvesFromCirclesArgumentParser().parse_args()
    results_dir_path = init_app(parser=parser)

    circles_generator = CirclesGenerator(
        log_dir_path=results_dir_path,
        num_workers=parser.num_workers,
        circles_count=parser.circles_count,
        curves_base_dir_path=results_dir_path,
        min_radius=parser.min_radius,
        max_radius=parser.max_radius,
        min_sampling_density=parser.min_sampling_density,
        max_sampling_density=parser.max_sampling_density)

    circles_generator.start()
    circles_generator.join()
