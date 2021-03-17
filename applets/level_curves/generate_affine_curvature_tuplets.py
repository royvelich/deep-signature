from deep_signature.data_generation.dataset_generation import AffineCurvatureTupletsDatasetGenerator
from common import settings

if __name__ == '__main__':
    AffineCurvatureTupletsDatasetGenerator.generate_tuples(
        dir_path=settings.level_curves_affine_curvature_tuplets_dir_path,
        curves_dir_path=settings.level_curves_dir_path_train,
        sections_density=0.35,
        negative_examples_count=2,
        supporting_points_count=6,
        max_offset=15,
        chunksize=6000)
