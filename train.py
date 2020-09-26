import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy
from deep_signature.training.deep_signature_dataset import DeepSignatureDataset
from deep_signature.training.deep_signature_net import DeepSignatureNet

if __name__ == '__main__':
    epochs = 10
    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    dataset = DeepSignatureDataset(dir_path='./dataset')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(numpy.floor(validation_split * dataset_size))

    if shuffle_dataset is True:
        numpy.random.seed(random_seed)
        numpy.random.shuffle(indices)

    train_indices, validation_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)

    net = DeepSignatureNet(sample_points=500, padding=2).cuda()

    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            pair = data['pairs'][0]
            x1 = pair[0,:]
            y = 5
