from typing import Union
import torch
import torch.nn as nn


def create_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: Union[int, str] = 0,
) -> list[nn.Module]:
    """Building block for Deep CNNs based on BatchNorm paper"""
    return [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    ]


class ResidualBlock(nn.Module):
    """Residual learning block (ResNet-34) from the ResNet paper"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[int, str] = 0,
    ):
        super(ResidualBlock, self).__init__()
        self.shortcut = (
            nn.Sequential(
                *create_conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                )
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv = nn.Sequential(
            *create_conv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride if in_channels == out_channels else 2,
                padding=padding,
            ),
            *create_conv_block(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return out + self.shortcut(x)


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduce_dim: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[int, str] = 0,
    ):
        """Residual learning block (ResNet-50/101/152) from the ResNet paper
        - 1×1 convolution that reduces the channel dimension (serving as a "bottleneck")
        - 3×3 convolution that operates on this reduced channel space
        - 1×1 convolution that expands the channels back to a higher dimension
        """
        super(BottleneckBlock, self).__init__()
        self.shortcut = (
            nn.Sequential(
                *create_conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                )
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv = nn.Sequential(
            # reduce dimension from in_channels to desired_in_channels
            *create_conv_block(
                in_channels=in_channels,
                out_channels=reduce_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            *create_conv_block(
                in_channels=reduce_dim,
                out_channels=reduce_dim,
                kernel_size=kernel_size,
                stride=stride if in_channels == out_channels else 2,
                padding=padding,
            ),
            *create_conv_block(
                in_channels=reduce_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return out + self.shortcut(x)
