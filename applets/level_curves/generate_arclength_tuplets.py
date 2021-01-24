from deep_signature.data_generation.dataset_generation import ArcLengthTupletsDatasetGenerator
from common import settings

if __name__ == '__main__':
    ArcLengthTupletsDatasetGenerator.generate_tuples(
        dir_path=settings.level_curves_arclength_tuplets_dir_path,
        curves_dir_path=settings.level_curves_dir_path_train,
        sections_density=0.1,
        negative_examples_count=2,
        supporting_points_count=20,
        min_perturbation=1,
        max_perturbation=5,
        min_offset=25,
        max_offset=25,
        chunksize=10000)
