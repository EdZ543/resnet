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


def get_data_loaders(batch_size):
    """Get train and test data loaders for CIFAR-10 dataset."""

    # Get data
    training_data = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )

    test_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader
