import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy
from deep_signature.training import DeepSignatureDataset
from deep_signature.training import DeepSignatureNet
from deep_signature.training import ContrastiveLoss


if __name__ == '__main__':
    epochs = 3
    batch_size = 32
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    learning_rate = 5e-5

    dataset = DeepSignatureDataset(dir_path='./dataset2')
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

    net = DeepSignatureNet(sample_points=600, padding=2).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = ContrastiveLoss(3)

    best_accumulated_avg_loss = 0
    for epoch in range(epochs):
        print(f' - Training Epoch #{epoch}:')
        running_loss = 0
        for batch, data in enumerate(train_loader, 0):
            # print(f'    - Batch #{batch}:')

            x1 = data['curves'][0]
            x2 = data['curves'][1]
            labels = data['labels']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out1 = net(x1)
            out2 = net(x2)
            loss = loss_fn(out1, out2, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if batch % 10 == 0:
            fill = ' '
            align = '<'
            width = 7
            print(f'    - [Epoch {epoch:{fill}{align}{width}} | Batch {batch:{fill}{align}{width}}] - Running Loss = {running_loss}')
            running_loss = 0

        print('\n')
        print(f' - Validating Epoch #{epoch}:')
        accumulated_avg_loss = 0
        for batch, data in enumerate(validation_loader, 0):
            x1 = data['curves'][0]
            x2 = data['curves'][1]
            labels = data['labels']

            out1 = net(x1)
            out2 = net(x2)
            loss = loss_fn(out1, out2, labels)

            # print statistics
            accumulated_avg_loss += (loss.item() / batch_size)
            if batch % 10 == 0:
                fill = ' '
                align = '<'
                width = 7
                print(f'    - [Batch {batch:{fill}{align}{width}}] - Accumulated Avg. Loss = {accumulated_avg_loss}')

        if epoch == 0:
            torch.save(net.state_dict(), './models/best_model.pt')
            best_accumulated_avg_loss = accumulated_avg_loss
        else:
            if accumulated_avg_loss < best_accumulated_avg_loss:
                torch.save(net.state_dict(), './models/best_model.pt')
                best_accumulated_avg_loss = accumulated_avg_loss
