"""Module containing the ResidualBlock class"""

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutions and a skip connection,
    as described in Figure 2 of the paper

    Parameters:
        num_filters: number of filters outputted

        subsample: whether to halve the feature map size and
            double the number of filters of the input
    """

    def __init__(self, num_filters, subsample):
        super().__init__()
        self.subsample = subsample

        if subsample:
            num_filters_in = num_filters // 2
            stride_in = 2
        else:
            num_filters_in = num_filters
            stride_in = 1

        self.conv1 = nn.Conv2d(
            num_filters_in,
            num_filters,
            kernel_size=3,
            stride=stride_in,
            padding=1,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(num_filters)

        self.mp = nn.MaxPool2d(1, 2)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def increase_dim(self, x):
        """
        Transforms input for shortcut connections across dimensions.
        As described in the paper, it halves the feature map size with a parameterless
        stride 2 convolution and uses zero-padding to increase dimensions.
        """

        # A max pool with kernel size 1 and stride 2 will
        # halve the input's dimensions without introducing new parameters!
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
