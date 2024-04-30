"""Data loader for CIFAR-10 dataset."""

from torch.utils.data import DataLoader
from torchvision import datasets


def get_dataloaders(train_transform, test_transform, batch_size, shuffle, pin_memory):
    """Get training and testing loaders for CIFAR-10 dataset."""

    train_data = datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, pin_memory=pin_memory, shuffle=shuffle
    )

    test_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, pin_memory=pin_memory, shuffle=shuffle
    )

    return train_dataloader, test_dataloader
