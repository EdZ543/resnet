from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
BATCH_SIZE = 128

# Define transformations
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # Per-pixel mean subtraction
        transforms.Normalize(mean=[], std=[]),
    ]
)

# Get data
training_data = datasets.CIFAR10(
    root="data", train=True, download=True, transform=transform
)

test_data = datasets.CIFAR10(
    root="data", train=False, download=True, transform=transform
)

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
