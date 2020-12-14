import torch
import numpy
import os
from deep_signature.training import SimpleDeepSignatureDataset
from deep_signature.training import SimpleDeepSignatureNet
from deep_signature.training import ContrastiveLoss
from deep_signature.training import ModelTrainer


def all_subdirs_of(b='.'):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
  return result


if __name__ == '__main__':
    epochs = 100
    batch_size = 50
    learning_rate = 1e-5
    validation_split = .1
    mu = 1

    torch.set_default_dtype(torch.float64)
    dataset = SimpleDeepSignatureDataset()
    dataset.load_dataset(
        negative_pairs_dir_path='C:/deep-signature-data/circles/negative-pairs',
        positive_pairs_dir_path='C:/deep-signature-data/circles/positive-pairs')
    model = SimpleDeepSignatureNet(layers=6, sample_points=3).cuda()
    print(model)

    device = torch.device('cuda')
    all_subdirs = all_subdirs_of('C:/deep-signature-data/circles/results')
    latest_subdir = os.path.normpath(max(all_subdirs, key=os.path.getmtime))
    results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    model.load_state_dict(torch.load(results['model_file_path'], map_location=device))

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loss_fn = ContrastiveLoss(mu)
    model_trainer = ModelTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    model_trainer.fit(dataset=dataset, epochs=epochs, batch_size=batch_size, validation_split=validation_split, results_base_dir_path='C:/deep-signature-data/circles/results')
