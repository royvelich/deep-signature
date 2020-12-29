from deep_signature.data_generation.dataset_generation import TupletsDatasetGenerator
from common import settings

if __name__ == '__main__':
    TupletsDatasetGenerator.generate_tuples(
        dir_path=settings.level_curves_section_tuplets_dir_path,
        curves_dir_path=settings.level_curves_dir_path_train,
        sections_density=0.1,
        negative_examples_count=10,
        supporting_points_count=4,
        max_offset=10,
        chunksize=1000)
