from typing import Union
import torch
import torch.nn as nn


def create_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    groups: int = 1,
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
            groups=groups,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    ]


# region ResNet


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


# endregion

# region BiSeNetV2


class StemBlock(nn.Module):
    def __init__(self, out_channels):
        """Stem block from BiSeNetV2 paper"""
        super(StemBlock, self).__init__()
        self.conv_1 = nn.Sequential(
            *create_conv_block(
                in_channels=3,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Sequential(
            *create_conv_block(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            *create_conv_block(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.conv_3 = nn.Sequential(
            *create_conv_block(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, image: torch.Tensor):
        x = self.conv_1(image)
        x1 = self.pool(x)
        x2 = self.conv_2(x)
        print(x1.shape, x2.shape)
        x3 = torch.concat([x1, x2], dim=1)
        return self.conv_3(x3)


class GatherExpansion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion_size: int = 6,
    ):
        """Gather-and-Expansion Layer from the BiSeNet V2 paper"""
        super(GatherExpansion, self).__init__()
        self.shortcut = (
            nn.Sequential(
                *create_conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                *create_conv_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv = nn.Identity()
        if in_channels == out_channels:
            # GE block from Fig.5 (b)
            self.conv = nn.Sequential(
                *create_conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                *create_conv_block(
                    in_channels=out_channels,
                    out_channels=expansion_size * out_channels,
                    kernel_size=3,
                    groups=out_channels,
                    stride=1,
                    padding=1,
                ),
                *create_conv_block(
                    in_channels=expansion_size * out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
        else:
            # GE block from Fig.5 (c)
            self.conv = nn.Sequential(
                *create_conv_block(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                *create_conv_block(
                    in_channels=in_channels,
                    out_channels=expansion_size * in_channels,
                    kernel_size=3,
                    groups=in_channels,
                    stride=stride,
                    padding=1,
                ),
                *create_conv_block(
                    in_channels=expansion_size * in_channels,
                    out_channels=expansion_size * in_channels,
                    kernel_size=3,
                    groups=in_channels,
                    stride=1,
                    padding=1,
                ),
                *create_conv_block(
                    in_channels=expansion_size * in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return out + self.shortcut(x)


# endregion
