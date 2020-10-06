from deep_signature.data_generation import DatasetGenerator

if __name__ == '__main__':
    dataset_generator = DatasetGenerator()
    dataset_generator.load_raw_curves(dir_path='C:/raw-data')
    dataset_generator.save(
        dir_path='C:/datasets/dataset1',
        pairs_per_curve=30,
        rotation_factor=10,
        sampling_factor=10,
        sample_points=500,
        metadata_only=False)
