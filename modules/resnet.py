"""Module containing the ResNet class"""

from torch import nn
import torch.nn.functional as F

from .residual_block import ResidualBlock


class ResNet(nn.Module):
    """ResNet model, as described in CIFAR-10 section of the paper."""

    def __init__(self, n):
        super().__init__()

        self.init_conv = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(16)

        # Stack of residual blocks with map size 32 and 16 filters
        self.stack1 = nn.Sequential(
            *[ResidualBlock(16, subsample=False) for _ in range(n)]
        )

        # Stack of residual blocks with map size 16 and 32 filters
        self.stack2 = nn.Sequential(
            ResidualBlock(32, subsample=True),
            *[ResidualBlock(32, subsample=False) for _ in range(n - 1)]
        )

        # Stack of residual blocks with map size 8 and 64 filters
        self.stack3 = nn.Sequential(
            ResidualBlock(64, subsample=True),
            *[ResidualBlock(64, subsample=False) for _ in range(n - 1)]
        )

        self.global_avg_pool = nn.AvgPool2d(8)
        self.fully_connected = nn.Linear(64, 10)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """Feed forward step"""

        out = self.init_conv(x)
        out = self.batch_norm(out)
        out = F.relu(out)

        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)

        out = self.global_avg_pool(out)
        out = out.view(-1, 64)
        out = self.fully_connected(out)
        out = F.log_softmax(out, dim=-1)

        return out
