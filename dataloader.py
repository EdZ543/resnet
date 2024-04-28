"""Data loader for CIFAR-10 dataset."""

from torch.utils.data import DataLoader
from torchvision import datasets


def get_dataloader(train_transform, test_transform, batch_size):
    """Get training and testing loaders for CIFAR-10 dataset."""

    train_data = datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    test_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform
    )
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader
