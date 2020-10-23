from deep_signature.data_generation import CurveDatasetGenerator
from deep_signature.data_generation import CurveDataGenerator

if __name__ == '__main__':
    curve_dataset_generator = CurveDatasetGenerator()
    curves = curve_dataset_generator.load_curves(file_path="C:/deep-signature-data/curves/curves.npy")
    negative_pairs, positive_pairs = curve_dataset_generator.generate_dataset(
        rotation_factor=10,
        sectioning_factor=20,
        sampling_factor=10,
        multimodality_factor=15,
        sampling_points_ratio=0.1,
        sampling_points_count=None,
        supporting_points_count=10,
        limit=8000)
