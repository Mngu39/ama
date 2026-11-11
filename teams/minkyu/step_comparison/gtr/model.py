"""Unmodified original definitions moved from puyo_train.py; imports are wiring only."""

import torch
from torch import nn


COLOR_CHANNEL = {"R": 0, "Y": 1, "G": 2, "B": 3, "#": 4}


CHANNELS = 5


HEIGHT = 14


WIDTH = 6


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class AfterstateValueNet(nn.Module):
    """[N,5,14,6] -> [N] future value in units of score_scale points."""

    def __init__(self, input_channels: int = 5, channels: int = 256,
                 blocks: int = 6):
        super().__init__()
        if blocks < 1 or channels < 1:
            raise ValueError("blocks and channels must be >= 1")
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.residuals = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head_conv = nn.Conv2d(channels, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14 * 6, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.stem[0].in_channels, 14, 6):
            raise ValueError(f"Expected [N,{self.stem[0].in_channels},14,6], got {tuple(x.shape)}")
        x = self.stem(x)
        x = self.residuals(x)
        x = self.head_conv(x)
        return self.head(x).squeeze(-1)
