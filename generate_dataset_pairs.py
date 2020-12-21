from deep_signature.data_generation import SimpleCurveDatasetGenerator

if __name__ == '__main__':
    count = 100000

    curves_dir_path_train = "C:/deep-signature-data/circles/curves/pairs/train"
    curves_dir_path_test = "C:/deep-signature-data/circles/curves/pairs/test"
    negative_pairs_dir_path = "C:/deep-signature-data/circles/datasets/pairs/negative-pairs"
    positive_pairs_dir_path = "C:/deep-signature-data/circles/datasets/pairs/positive-pairs"

    SimpleCurveDatasetGenerator.generate_circles(
        dir_path=curves_dir_path_train,
        min_radius=20,
        max_radius=200,
        circles_count=2000,
        sampling_density=1)

    SimpleCurveDatasetGenerator.generate_circles(
        dir_path=curves_dir_path_test,
        min_radius=20,
        max_radius=200,
        circles_count=2000,
        sampling_density=1)

    SimpleCurveDatasetGenerator.generate_negative_pairs(
        pairs_dir_path=negative_pairs_dir_path,
        curves_dir_path=curves_dir_path_train,
        count=count,
        chunk_size=1000)

    SimpleCurveDatasetGenerator.generate_positive_pairs(
        pairs_dir_path=positive_pairs_dir_path,
        curves_dir_path=curves_dir_path_train,
        count=count,
        chunk_size=1000)
