from deep_signature.data_generation import curve_generation
from common import settings

if __name__ == '__main__':
    # curve_generation.LevelCurvesGenerator.generate_curves(
    #     dir_path=settings.level_curves_dir_path_train,
    #     curves_count=25000,
    #     images_base_dir_path=settings.images_dir_path_train,
    #     sigmas=[2, 4, 8, 16, 32],
    #     contour_levels=[0.2, 0.5, 0.8],
    #     min_points=500,
    #     max_points=6000,
    #     flat_point_threshold=1e-3,
    #     max_flat_points_ratio=0.03,
    #     max_abs_kappa=50,
    #     chunksize=1000
    # )

    curve_generation.LevelCurvesGenerator.generate_curves(
        dir_path=settings.level_curves_dir_path_validation,
        curves_count=10000,
        images_base_dir_path=settings.images_dir_path_validation,
        sigmas=[2, 4, 8, 16, 32],
        contour_levels=[0.2, 0.5, 0.8],
        min_points=500,
        max_points=6000,
        flat_point_threshold=1e-3,
        max_flat_points_ratio=0.03,
        max_abs_kappa=50,
        chunksize=1000
    )