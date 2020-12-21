from deep_signature.data_generation import CurveDatasetGenerator
from deep_signature.data_generation import SimpleCurveDatasetGenerator

if __name__ == '__main__':
    curves_dir_path_train = "C:/deep-signature-data/circles/curves/tuplets/train"
    curves_dir_path_test = "C:/deep-signature-data/circles/curves/tuplets/test"
    tuplets_dir_path = "C:/deep-signature-data/circles/datasets/tuplets"

    SimpleCurveDatasetGenerator.generate_circles(
        dir_path=curves_dir_path_test,
        min_radius=20,
        max_radius=800,
        circles_count=1000,
        sampling_density=0.3)

    # SimpleCurveDatasetGenerator.generate_tuplets(
    #     curves_dir_path=curves_dir_path_train,
    #     tuplets_dir_path=tuplets_dir_path,
    #     tuplets_per_curve=300,
    #     tuplet_length=50,
    #     chunk_size=150)
