"""Data loader for CIFAR-10 dataset."""

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split


def get_dataloaders(
    root,
    train_transform,
    test_transform,
    batch_size,
    shuffle,
    pin_memory,
):
    """Get training, validation, and testing loaders for CIFAR-10 dataset."""

    # Load training and validation data
    train_data = datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    val_data = datasets.CIFAR10(
        root=root, train=True, download=False, transform=test_transform
    )

    # Generate indices
    indices = list(range(len(train_data)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.1, random_state=0
    )

    # Split into training and validation sets
    train_data = Subset(train_data, train_indices)
    val_data = Subset(val_data, val_indices)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    # Load test data
    test_data = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    return train_dataloader, val_dataloader, test_dataloader
