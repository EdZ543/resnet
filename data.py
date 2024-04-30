"""Data loader for CIFAR-10 dataset."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets


def get_dataloaders(
    root,
    train_transform,
    test_transform,
    batch_size,
    shuffle,
    pin_memory,
):
    """Get training, validation, and testing loaders for CIFAR-10 dataset."""

    train_data = datasets.CIFAR10(root=root, train=True, download=True)

    # Split training data into training and validation sets
    generator = torch.Generator().manual_seed(0)
    train_data, val_data = random_split(train_data, [45000, 5000], generator=generator)

    train_data.dataset.transform = train_transform
    val_data.dataset.transform = test_transform

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )

    test_data = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )

    return train_dataloader, val_dataloader, test_dataloader
