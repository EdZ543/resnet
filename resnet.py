"""Main ResNet model implementation."""

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutions and a skip connection,
    as described in Figure 2 of the paper

    Parameters:
        num_filters: number of filters outputted
        subsample: whether to halve the feature map size and double the number of filters of the input
    """

    def __init__(self, num_filters, subsample=False):
        super().__init__()
        self.subsample = subsample

        if subsample:
            self.conv1 = nn.Conv2d(num_filters // 2, num_filters, 3, 2, 1)
        else:
            self.conv1 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.batch_norm1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.batch_norm2 = nn.BatchNorm2d(num_filters)

        self.mp = nn.MaxPool2d(1, 2)

    def increase_dim(self, x):
        """
        Transforms input for shortcut connections across dimensions.
        As described in the paper, it halves the feature map size with a parameterless
        stride 2 convolution and uses zero-padding to increase dimensions.
        """

        # A max pool with kernel size 1 and stride 2 will halve the input's dimensions without introducing new parameters!
        out = self.mp(x)

        # (batch_size, channels, h, w) -> (batch_size, 2 * channels, h, w)
        out = torch.cat([out, out.mul(0)], 1)

        return out

    def forward(self, x):
        """Feed forward step"""

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)

        # Residual connection
        if self.subsample:
            out += self.increase_dim(x)
        else:
            out += x

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model, as described in CIFAR-10 section of the paper."""

    def __init__(self, n):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)

        # Stack of residual blocks with map size 32 and 16 filters
        self.stack1 = nn.Sequential(*[ResidualBlock(16) for _ in range(n)])

        # Stack of residual blocks with map size 16 and 32 filters
        self.stack2 = nn.Sequential(
            ResidualBlock(32, subsample=True),
            *[ResidualBlock(32) for _ in range(n - 1)]
        )

        # Stack of residual blocks with map size 64 and 8 filters
        self.stack3 = nn.Sequential(
            ResidualBlock(64, subsample=True),
            *[ResidualBlock(64) for _ in range(n - 1)]
        )

        self.global_avg_pool = nn.AvgPool2d(8)
        self.fully_connected = nn.Linear(64, 10)

    def forward(self, x):
        """Feed forward step"""

        out = self.conv1(x)
        out = F.relu(out)
        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 64)
        out = self.fully_connected(out)
        return out
