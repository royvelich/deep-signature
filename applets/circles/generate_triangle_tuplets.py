from deep_signature.data_generation.dataset_generation import TupletsDatasetGenerator
from common import settings

if __name__ == '__main__':
    TupletsDatasetGenerator.generate_tuples(
        dir_path=settings.circles_triangle_tuplets_dir_path,
        curves_dir_path=settings.circles_dir_path_train,
        sections_density=0.01,
        negative_examples_count=15,
        supporting_points_count=1,
        chunksize=1000)
