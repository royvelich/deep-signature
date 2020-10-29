import torch
from deep_signature.training import DeepSignatureDataset
from deep_signature.training import DeepSignatureNet
from deep_signature.training import ContrastiveLoss
from deep_signature.training import ModelTrainer

if __name__ == '__main__':
    epochs = 30
    batch_size = 64
    learning_rate = 1e-4

    torch.set_default_dtype(torch.float64)
    dataset = DeepSignatureDataset()
    dataset.load_dataset(dir_path='C:/deep-signature-data/datasets/dataset2')
    model = DeepSignatureNet(sample_points=13).cuda()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = ContrastiveLoss(1)
    model_trainer = ModelTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    model_trainer.fit(dataset=dataset, epochs=epochs, batch_size=batch_size, results_base_dir_path='C:/deep-signature-data/results')
