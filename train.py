import torch
from deep_signature.training import SimpleDeepSignatureDataset
from deep_signature.training import SimpleDeepSignatureNet
from deep_signature.training import ContrastiveLoss
from deep_signature.training import ModelTrainer

if __name__ == '__main__':
    epochs = 50
    batch_size = 1024
    learning_rate = 1e-4

    torch.set_default_dtype(torch.float64)
    dataset = SimpleDeepSignatureDataset()
    dataset.load_dataset(
        negative_pairs_dir_path='C:/deep-signature-data/circles/negative-pairs',
        positive_pairs_dir_path='C:/deep-signature-data/circles/positive-pairs')
    model = SimpleDeepSignatureNet(layers=6, sample_points=3).cuda()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = ContrastiveLoss(1)
    model_trainer = ModelTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    model_trainer.fit(dataset=dataset, epochs=epochs, batch_size=batch_size, results_base_dir_path='C:/deep-signature-data/circles/results')
