from deep_signature.data_generation import CurveDatasetGenerator

if __name__ == '__main__':
    curve_dataset_generator = CurveDatasetGenerator()
    generated_curves = curve_dataset_generator.generate_curves(dir_path="C:/deep-signature-data/images", plot_curves=False)
    curve_dataset_generator.save_curves(dir_path="C:/deep-signature-data/curves")
