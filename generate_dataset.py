from deep_signature.data_generation import CurveDatasetGenerator
from deep_signature.data_generation import SimpleCurveDatasetGenerator

if __name__ == '__main__':
    # count = 3000
    # pairs_per_curve = 50

    SimpleCurveDatasetGenerator.generate_circles(
        dir_path="C:/deep-signature-data/circles/curves/test",
        min_radius=20,
        max_radius=1000,
        circles_count=1000,
        sampling_density=0.3)

    # SimpleCurveDatasetGenerator.generate_tuplets(
    #     curves_dir_path="C:/deep-signature-data/circles/curves/train",
    #     tuplets_dir_path="C:/deep-signature-data/circles/tuplets/train",
    #     tuplets_per_curve=300,
    #     tuplet_length=50,
    #     chunk_size=150)

    # SimpleCurveDatasetGenerator.generate_negative_pairs(
    #     pairs_dir_path="C:/deep-signature-data/circles/negative-pairs",
    #     curves_dir_path="C:/deep-signature-data/circles/curves",
    #     count=count*pairs_per_curve,
    #     chunk_size=1000)

    # SimpleCurveDatasetGenerator.generate_positive_pairs(
    #     pairs_dir_path="C:/deep-signature-data/circles/positive-pairs",
    #     curves_dir_path="C:/deep-signature-data/circles/curves",
    #     count=count,
    #     chunk_size=1000,
    #     pairs_per_curve=pairs_per_curve)
    #
    # SimpleCurveDatasetGenerator.generate_negative_pairs_from_positive_pairs(
    #     pairs_dir_path="C:/deep-signature-data/circles/negative-pairs",
    #     packed_positive_pairs_dir_path="C:/deep-signature-data/circles/positive-pairs",
    #     chunk_size=1000)

    # curve_dataset_generator = CurveDatasetGenerator()
    # curve_dataset_generator.generate_curve_managers(
    #     rotation_factor=1,
    #     sampling_factor=15,
    #     multimodality_factor=15,
    #     sampling_points_ratio=0.15,
    #     sampling_points_count=30,
    #     supporting_points_count=3,
    #     sectioning_points_count=None,
    #     sectioning_points_ratio=0.1,
    #     section_points_count=100,
    #     evolution_iterations=3,
    #     evolution_dt=1e-12,
    #     chunk_size=10,
    #     curves_dir_path="C:/deep-signature-data/curves/version1",
    #     curve_managers_dir_path="C:/deep-signature-data/curve-managers/version1/",
    #     curve_managers_per_file=20,
    #     limit=1)

    # curve_dataset_generator.generate_positive_pairs(
    #     rotation_factor=5,
    #     sampling_factor=10,
    #     multimodality_factor=15,
    #     sampling_points_ratio=0.15,
    #     sampling_points_count=30,
    #     supporting_points_count=3,
    #     sectioning_points_count=None,
    #     sectioning_points_ratio=0.02,
    #     section_points_count=100,
    #     evolution_iterations=3,
    #     evolution_dt=1e-12,
    #     chunk_size=10,
    #     curves_dir_path="C:/deep-signature-data/curves/version1",
    #     curve_managers_dir_path="C:/deep-signature-data/curve-managers/version1/",
    #     curve_managers_per_file=20,
    #     min_curvature=0.08,
    #     max_curvature_diff=0.3,
    #     min_curvature_diff=0.1,
    #     min_norm_diff=3,
    #     limit=40)
    #
    # # curve_dataset_generator.save_negative_pairs(dir_path="C:/deep-signature-data/negative-pairs/version1/")
    # curve_dataset_generator.save_positive_pairs(dir_path="C:/deep-signature-data/positive-pairs/version1/")
