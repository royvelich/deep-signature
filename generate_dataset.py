from deep_signature.data_generation import DatasetGenerator

if __name__ == '__main__':
    dataset_generator = DatasetGenerator()
    dataset_generator.load_raw_curves(dir_path='C:\\Users\\Roy\\OneDrive - Technion\\deep-signature-raw-data\\raw-data-new')
    dataset_generator.save(
        dir_path='./dataset2',
        pairs_per_curve=10,
        rotation_factor=12,
        sampling_factor=15,
        sample_points=600,
        metadata_only=True)
