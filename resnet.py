import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, feature_map_size, num_filters, filter_size, subsample=False):
        super().__init__()
        in_channels = feature_map_size * 2 if subsample else feature_map_size
        stride = 2 if subsample else 1
        padding = filter_size // 2

        self.conv1 = nn.Conv2d(in_channels, num_filters, filter_size, stride, padding)
        self.conv2 = nn.Conv2d(num_filters, num_filters, filter_size, stride, padding)

    def forward(self, x):
        original_x = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        # Residual connection
        x += original_x

        x = F.relu(x)


class ResNet(nn.Module):
    def __init__(self, n):
        super().__init__()
