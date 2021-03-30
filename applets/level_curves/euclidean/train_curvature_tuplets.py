import torch
import os
import numpy
from deep_signature.nn.datasets import DeepSignatureTupletsDataset
from deep_signature.nn.networks import DeepSignatureCurvatureNet
from deep_signature.nn.losses import TupletLoss
from deep_signature.nn.trainers import ModelTrainer
from common import settings


def all_subdirs_of(b='.'):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
  return result


if __name__ == '__main__':
    epochs = 1000
    batch_size = 6000
    learning_rate = 5e-6
    validation_split = .05

    torch.set_default_dtype(torch.float64)
    dataset = DeepSignatureTupletsDataset()
    dataset.load_dataset(dir_path=settings.level_curves_euclidean_curvature_tuplets_dir_path)
    model = DeepSignatureCurvatureNet(sample_points=13).cuda()
    print(model)

    device = torch.device('cuda')
    all_subdirs = all_subdirs_of(settings.level_curves_euclidean_curvature_tuplets_results_dir_path)
    latest_subdir = os.path.normpath(max(all_subdirs, key=os.path.getmtime))
    results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    model.load_state_dict(torch.load(results['model_file_path'], map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    tuplet_loss_fn = TupletLoss()
    model_trainer = ModelTrainer(model=model, loss_functions=[tuplet_loss_fn], optimizer=optimizer)
    model_trainer.fit(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        results_base_dir_path=settings.level_curves_euclidean_curvature_tuplets_results_dir_path)
