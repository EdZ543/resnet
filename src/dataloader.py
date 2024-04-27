"""Data loader for CIFAR-10 dataset."""

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # Paper uses per-pixel mean subtraction
        # I use PyTorch's normalization
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomCrop(32, padding=4),
    ]
)


def get_dataloader(train, batch_size):
    """Get train and test data loaders for CIFAR-10 dataset."""

    if train:
        data = datasets.CIFAR10(
            root="data", train=True, download=True, transform=train_transform
        )

        # Split data into train and validation sets
        generator = torch.Generator().manual_seed(42)
        train_data, val_data = random_split(data, [45000, 5000], generator=generator)

        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        return train_dataloader, val_dataloader
    else:
        # If testing, no transformations
        data = datasets.CIFAR10(
            root="data", train=False, download=True, transform=transforms.ToTensor()
        )

        dataloader = DataLoader(data, batch_size=batch_size)

        return dataloader
