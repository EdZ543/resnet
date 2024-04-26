import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
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

    # Get data
    data = datasets.CIFAR10(
        root="data", train=train, download=True, transform=transform
    )

    # Create data loaders
    if train:
        generator = torch.Generator().manual_seed(42)
        train_data, val_data = random_split(data, [45000, 5000], generator=generator)

        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        return train_dataloader, val_dataloader
    else:
        dataloader = DataLoader(data, batch_size=batch_size)
        return dataloader
