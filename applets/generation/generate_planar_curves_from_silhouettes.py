# python peripherals
from pathlib import Path

# applets
from applets.core.utils import AppArgumentParser, init_app_tap

# deep-signature
from deep_signature.manifolds.planar_curves.generation import SilhouetteLevelCurvesGenerator


class GeneratePlanarCurvesFromSilhouettesArgumentParser(AppArgumentParser):
    num_workers: int
    images_base_dir_path: Path
    min_points_count: int
    max_points_count: int
    contour_level: float


if __name__ == '__main__':
    parser = GeneratePlanarCurvesFromSilhouettesArgumentParser().parse_args()
    results_dir_path = init_app_tap(parser=parser)

    silhouette_level_curves_generator = SilhouetteLevelCurvesGenerator(
        log_dir_path=results_dir_path,
        num_workers=parser.num_workers,
        images_base_dir_path=parser.images_base_dir_path,
        curves_base_dir_path=results_dir_path,
        min_points_count=parser.min_points_count,
        max_points_count=parser.max_points_count,
        contour_level=parser.contour_level)

    silhouette_level_curves_generator.start()
