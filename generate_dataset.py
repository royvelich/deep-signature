from deep_signature.data_generation import CurveDatasetGenerator
from deep_signature.data_generation import CurveDataGenerator

if __name__ == '__main__':
    curve_dataset_generator = CurveDatasetGenerator()
    curves = curve_dataset_generator.load_curves(file_path="C:/deep-signature-data/curves/curves.npy")
    curve_data_generator = CurveDataGenerator(
        curve=curves[0],
        rotation_factor=10,
        sectioning_factor=20,
        sampling_factor=4,
        multimodality_factor=20,
        sampling_points_count=50,
        supporting_points_count=300)

    negative_pairs = curve_data_generator.generate_negative_pairs()