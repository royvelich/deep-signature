from deep_signature.data_generation.curve_generation import CurveDatasetGenerator

if __name__ == '__main__':
    curve_dataset_generator = CurveDatasetGenerator()

    generated_curves = curve_dataset_generator.generate_curves(
        dir_path="C:/deep-signature-data/images",
        chunk_size=10,
        plot_curves=False)

    curve_dataset_generator.save_curves(
        dir_path="C:/deep-signature-data/curves/version1",
        curve_per_file=100)
