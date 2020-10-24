from deep_signature.data_generation import CurveDatasetGenerator

if __name__ == '__main__':
    curve_dataset_generator = CurveDatasetGenerator()
    curves = curve_dataset_generator.load_curves(file_path="C:/deep-signature-data/curves/curves.npy")
    negative_pairs, positive_pairs = curve_dataset_generator.generate_dataset(
        rotation_factor=10,
        sampling_factor=10,
        multimodality_factor=15,
        sampling_points_ratio=0.15,
        sampling_points_count=None,
        supporting_points_count=10,
        sectioning_points_count=None,
        sectioning_points_ratio=0.2,
        limit=3500,
        chunk_size=2)

    curve_dataset_generator.save_dataset(dir_path="C:/deep-signature-data/datasets/dataset1/")

