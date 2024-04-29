"""Data loader for CIFAR-10 dataset."""

from torch.utils.data import DataLoader
from torchvision import datasets


def get_dataloader(transform, batch_size):
    """Get training and testing loaders for CIFAR-10 dataset."""

    data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size)

    return dataloader
