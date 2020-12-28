from deep_signature.data_generation.curve_generation import CirclesGenerator
from common import settings

if __name__ == '__main__':
    CirclesGenerator.generate_curves(
        dir_path=settings.circles_dir_path_train,
        curves_count=5000,
        min_radius=20,
        max_radius=800,
        sampling_density=2,
        chunksize=2000)

    CirclesGenerator.generate_curves(
        dir_path=settings.circles_dir_path_test,
        curves_count=5000,
        min_radius=20,
        max_radius=800,
        sampling_density=2,
        chunksize=2000)