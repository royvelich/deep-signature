import torch
import numpy
import os
from deep_signature.nn import DeepSignatureTupletsDataset
from deep_signature.nn import DeepSignatureNet
from deep_signature.nn import TupletLoss
from deep_signature.nn import ModelTrainer


def all_subdirs_of(b='.'):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
  return result


if __name__ == '__main__':
    epochs = 350
    batch_size = 2000
    learning_rate = 1e-5
    validation_split = .1

    torch.set_default_dtype(torch.float64)
    dataset = DeepSignatureTupletsDataset()
    dataset.load_dataset(dir_path='C:/deep-signature-data/circles/tuplets/train')
    model = DeepSignatureNet(layers=6, sample_points=3).cuda()
    print(model)

    device = torch.device('cuda')
    all_subdirs = all_subdirs_of('C:/deep-signature-data/circles/results/tuplets')
    latest_subdir = os.path.normpath(max(all_subdirs, key=os.path.getmtime))
    results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    model.load_state_dict(torch.load(results['model_file_path'], map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = TupletLoss()
    model_trainer = ModelTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    model_trainer.fit(dataset=dataset, epochs=epochs, batch_size=batch_size, validation_split=validation_split, results_base_dir_path='C:/deep-signature-data/circles/results/tuplets')
