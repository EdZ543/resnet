from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutions and a skip connection,
    as described in Figure 2 of the paper
    """

    def __init__(self, feature_map_size, num_filters, filter_size, subsample=False):
        super().__init__()
        self.subsample = subsample

        in_channels = feature_map_size * 2 if subsample else feature_map_size
        stride = 2 if subsample else 1
        padding = filter_size // 2

        self.conv1 = nn.Conv2d(in_channels, num_filters, filter_size, stride, padding)
        self.conv2 = nn.Conv2d(num_filters, num_filters, filter_size, stride, padding)
        # The paper uses option A for the shortcut connection, I use option B
        self.projection_shortcut = nn.Conv2d(in_channels, num_filters, 1, 2, 0)

    def forward(self, x):
        original_x = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        # Apply projection shortcut if input and output dimensions don't match
        if self.subsample:
            original_x = self.projection_shortcut(original_x)

        # Residual connection
        x += original_x

        x = F.relu(x)


class ResNet(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)

        # Stack of residual blocks with map size 32 and 16 filters
        self.stack1 = nn.Sequential(*[ResidualBlock(32, 16, 3) for _ in range(n)])

        # Stack of residual blocks with map size 16 and 32 filters
        self.stack2 = nn.Sequential(
            ResidualBlock(16, 32, 3, subsample=True),
            *[ResidualBlock(16, 32, 3) for _ in range(n - 1)]
        )

        # Stack of residual blocks with map size 64 and 8 filters
        self.stack3 = nn.Sequential(
            ResidualBlock(8, 64, 3, subsample=True),
            *[ResidualBlock(8, 64, 3) for _ in range(n - 1)]
        )

        self.global_avg_pool = nn.AvgPool2d(8)
        self.fully_connected = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.global_avg_pool(x)  # Size is now (batch_size, 64, 1, 1)
        x = x.view(-1, 64)  # Size is now (batch_size, 64)
        x = self.fully_connected(
            x
        )  # For each sample, takes in 64 inputs (average of each feature map)
        return x
