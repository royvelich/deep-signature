# python peripherals
from pathlib import Path
from typing import List, Tuple

# applets
from applets.core.utils import AppArgumentParser, init_app

# deep-signature
from deep_signature.manifolds.planar_curves.groups import EuclideanGroup, EquiaffineGroup, SimilarityGroup, AffineGroup
from deep_signature.manifolds.planar_curves.generation import ShapeMatchingBenchmarkCurvesGenerator


class GenerateShapeMatchingBenchmarkCurvesArgumentParser(AppArgumentParser):
    num_workers: int
    curves_base_dir_path: Path
    sampling_ratios: List[float]
    multimodalities: List[int]
    group_names: List[str]
    min_det: float
    max_det: float
    min_cond: float
    max_cond: float
    fig_size: Tuple[int, int]
    point_size: float


if __name__ == '__main__':
    parser = GenerateShapeMatchingBenchmarkCurvesArgumentParser().parse_args()
    results_dir_path = init_app(parser=parser)

    shape_matching_benchmark_curves_generator = ShapeMatchingBenchmarkCurvesGenerator(
        log_dir_path=results_dir_path,
        num_workers=parser.num_workers,
        curves_base_dir_path=parser.curves_base_dir_path,
        benchmark_base_dir_path=results_dir_path,
        sampling_ratios=parser.sampling_ratios,
        multimodalities=parser.multimodalities,
        group_name=parser.group_names,
        min_cond=parser.min_cond,
        max_cond=parser.max_cond,
        min_det=parser.min_det,
        max_det=parser.max_det,
        fig_size=tuple(parser.fig_size),
        point_size=parser.point_size)

    shape_matching_benchmark_curves_generator.start()
