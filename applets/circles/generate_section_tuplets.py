from deep_signature.data_generation.dataset_generation import CurvatureTupletsDatasetGenerator
from common import settings

if __name__ == '__main__':
    CurvatureTupletsDatasetGenerator.generate_tuples(
        dir_path=settings.circles_section_tuplets_dir_path,
        curves_dir_path=settings.circles_dir_path_train,
        sections_density=0.005,
        negative_examples_count=10,
        supporting_points_count=6,
        max_offset=30,
        chunksize=1000)
