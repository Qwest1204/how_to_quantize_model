from model import Net
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import urllib.request
from urllib.request import urlopen
import ssl
import json

ssl._create_default_https_context = ssl._create_unverified_context

# Only define global constants outside main guard
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
    ])

    # Set hyperparameters
    batch_size = 4  # Number of samples per batch

    # Initialize CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data',  # Dataset storage directory
        train=True,  # Use training split
        download=True,  # Download if missing
        transform=transform
    )

    # Create training data loader with multiprocessing
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle samples each epoch
        num_workers=2  # Subprocesses for data loading
    )

    # Initialize CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,  # Use test split
        download=True,
        transform=transform
    )

    # Create test data loader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=2
    )

    # Create iterator and get first batch of training data
    dataiter = iter(trainloader)
    images, labels = next(dataiter)  # Used for debugging/exploration

    # Initialize neural network
    net = Net()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Classification loss
    optimizer = torch.optim.SGD(
        net.parameters(),  # Network parameters to optimize
        lr=0.001,  # Learning rate
        momentum=0.9  # Momentum factor
    )

    # Training loop
    for epoch in range(2):  # Number of full dataset passes
        running_loss = 0.0  # Accumulated loss per epoch
        for i, data in enumerate(trainloader, 0):
            # Unpack batch data
            inputs, labels = data

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track training statistics
            running_loss += loss.item()

            # Print periodic progress
            if i % 2000 == 1999:  # Every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Save trained model
    PATH = 'cifar_net.pth'
    torch.save(net.state_dict(), PATH)