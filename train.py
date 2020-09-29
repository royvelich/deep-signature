import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy
from deep_signature.training import DeepSignatureDataset
from deep_signature.training import DeepSignatureNet
from deep_signature.training import ContrastiveLoss
from datetime import datetime
import os
from pathlib import Path

if __name__ == '__main__':
    epochs = 100
    batch_size = 256
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    learning_rate = 5e-4

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

    net = DeepSignatureNet(sample_points=dataset.sample_points).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = ContrastiveLoss(1)

    fill = ' '
    align = '>'
    width = 7
    loss_width = 25

    print(f'Epochs                                    {epochs:{fill}{align}{width}}')
    print(f'Batch size                                {batch_size:{fill}{align}{width}}')
    print(f'Training dataset length                   {len(train_indices):{fill}{align}{width}}')
    print(f'Training dataset batches per epoch        {int(numpy.ceil(len(train_indices) / batch_size)):{fill}{align}{width}}')
    print(f'Validation dataset length                 {len(validation_indices):{fill}{align}{width}}')
    print(f'Validation dataset batches per epoch      {int(numpy.ceil(len(validation_indices) / batch_size)):{fill}{align}{width}}')

    stats = {
        'epoch': epochs,
        'batch_size': batch_size,
        'training_loss_list': [],
        'validation_loss_list': []
    }

    dir_name = f"./train_results/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    Path(dir_name).mkdir(parents=True, exist_ok=True)

    best_average_validation_batch_loss = 0
    for epoch in range(epochs):
        print(f' - Training Epoch #{epoch}:')
        accumulated_training_epoch_loss = 0
        average_training_batch_loss = 0
        for batch, data in enumerate(train_loader, 0):

            # if batch == 5:
            #     break

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
            batch_loss = loss.item()
            accumulated_training_epoch_loss = accumulated_training_epoch_loss + batch_loss
            average_training_batch_loss = accumulated_training_epoch_loss / (batch + 1)
            print(f'    - [Epoch {epoch:{fill}{align}{width}} | Batch {batch:{fill}{align}{width}}]: Batch Loss = {batch_loss:{fill}{align}{loss_width}}, Avg. Batch Loss = {average_training_batch_loss:{fill}{align}{loss_width}}')

        stats['training_loss_list'].append(average_training_batch_loss)

        print(f' - Validating Epoch #{epoch}:')
        accumulated_validation_epoch_loss = 0
        average_validation_batch_loss = 0
        for batch, data in enumerate(validation_loader, 0):

            # if batch == 5:
            #     break

            x1 = data['curves'][0]
            x2 = data['curves'][1]
            labels = data['labels']

            out1 = net(x1)
            out2 = net(x2)
            loss = loss_fn(out1, out2, labels)

            # print statistics
            batch_loss = loss.item()
            accumulated_validation_epoch_loss = accumulated_validation_epoch_loss + batch_loss
            average_validation_batch_loss = accumulated_validation_epoch_loss / (batch + 1)
            print(f'    - [Epoch {epoch:{fill}{align}{width}} | Batch {batch:{fill}{align}{width}}]: Batch Loss = {batch_loss:{fill}{align}{loss_width}}, Avg. Batch Loss = {average_validation_batch_loss:{fill}{align}{loss_width}}')

        stats['validation_loss_list'].append(average_validation_batch_loss)

        model_file_name = os.path.join(dir_name, 'best_model.pt')
        if epoch == 0:
            torch.save(net.state_dict(), model_file_name)
            best_average_validation_batch_loss = average_validation_batch_loss
        else:
            if average_validation_batch_loss < best_average_validation_batch_loss:
                torch.save(net.state_dict(), model_file_name)
                best_average_validation_batch_loss = average_validation_batch_loss

    loss_file_name = os.path.join(dir_name, 'loss.npy')
    numpy.save(loss_file_name, stats, allow_pickle=True)
