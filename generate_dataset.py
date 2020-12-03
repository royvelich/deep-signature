from deep_signature.data_generation import CurveDatasetGenerator

if __name__ == '__main__':
    curve_dataset_generator = CurveDatasetGenerator()
    curves = curve_dataset_generator.load_curves(dir_path="C:/deep-signature-data/curves/version1")
    negative_pairs, positive_pairs = curve_dataset_generator.generate_dataset(
        rotation_factor=1,
        sampling_factor=15,
        multimodality_factor=15,
        sampling_points_ratio=0.15,
        sampling_points_count=None,
        supporting_points_count=3,
        sectioning_points_count=None,
        sectioning_points_ratio=0.15,
        evolution_iterations=3,
        evolution_dt=1e-12,
        limit=1000,
        chunk_size=2)

    curve_dataset_generator.save_dataset(dir_path="C:/deep-signature-data/datasets/dataset3/")
