from deep_signature.data_generation.dataset_generation import ArcLengthTupletsDatasetGenerator
from common import settings

if __name__ == '__main__':
    ArcLengthTupletsDatasetGenerator.generate_tuples(
        dir_path=settings.level_curves_arclength_tuplets_dir_path,
        curves_dir_path=settings.level_curves_dir_path_train,
        sections_density=0.03,
        negative_examples_count=15,
        supporting_points_count=30,
        max_perturbation=0.25,
        min_offset=50,
        max_offset=80,
        chunksize=1000)
