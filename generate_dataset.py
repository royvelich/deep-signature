from deep_signature.data_generation import CurveDatasetGenerator

if __name__ == '__main__':
    curve_dataset_generator = CurveDatasetGenerator()
    curve_dataset_generator.generate_curve_managers(
        rotation_factor=1,
        sampling_factor=15,
        multimodality_factor=15,
        sampling_points_ratio=0.15,
        sampling_points_count=30,
        supporting_points_count=3,
        sectioning_points_count=None,
        sectioning_points_ratio=0.1,
        section_points_count=100,
        evolution_iterations=3,
        evolution_dt=1e-12,
        chunk_size=10,
        curves_dir_path="C:/deep-signature-data/curves/version1",
        curve_managers_dir_path="C:/deep-signature-data/curve-managers/version1/",
        curve_managers_per_file=20,
        limit=1)
