from typing import Union
import torch
import torch.functional as F
import torch.nn as nn


def create_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    groups: int = 1,
    padding: Union[int, str] = 0,
    relu: bool = True,
    batch_norm: bool = True,
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
        nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
        nn.ReLU() if relu else nn.Identity(),
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


class BilateralGuidedAggregation(nn.Module):
    def __init__(self, in_channels: int = 3):
        """Bilateral Guided Aggregation from the BiSeNetV2 paper"""
        super(BilateralGuidedAggregation, self).__init__()
        self.pool_1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # going left to right on the Fig.6 from the paper
        self.conv_1 = nn.Sequential(
            *create_conv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                relu=False,
            ),
            *create_conv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                relu=False,
                batch_norm=False,
            ),
        )
        self.conv_2 = nn.Sequential(
            *create_conv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                relu=False,
            ),
            self.pool_1,
        )
        self.conv_3 = nn.Sequential(
            *create_conv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                relu=False,
            ),
            nn.Upsample(scale_factor=4, align_corners=True),
            nn.Sigmoid(),
        )
        self.conv_4 = nn.Sequential(
            *create_conv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                relu=False,
            ),
            *create_conv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                relu=False,
                batch_norm=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, detail_branch_x: torch.Tensor, semantic_branch_x: torch.Tensor):
        x1 = self.conv_1(detail_branch_x)
        x2 = self.conv_2(detail_branch_x)
        x3 = self.conv_3(semantic_branch_x)
        x4 = self.conv_4(semantic_branch_x)
        x5 = x1 * x3  # 𝐻 × 𝑊 × 𝐶)
        x6 = x2 * x4  # 𝐻 / 4 × 𝑊 / 4 × 𝐶

        return x5 + nn.Upsample(scale_factor=4, align_corners=True)(x6)


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
        x3 = torch.concat([x1, x2], dim=1)
        return self.conv_3(x3)


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(ContextEmbeddingBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv_1 = nn.Sequential(
            *create_conv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.conv_2 = nn.Sequential(
            *create_conv_block(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(self, x: torch.Tensor):
        x1 = self.bn(self.pool(x))
        x2 = self.conv_1(x1)
        x3 = self.conv_2(x + x2)
        return x3


class GatherExpansionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion_size: int = 6,
    ):
        """Gather-and-Expansion Layer from the BiSeNet V2 paper"""
        super(GatherExpansionBlock, self).__init__()
        self.shortcut = (
            nn.Sequential(
                *create_conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    relu=False,
                ),
                *create_conv_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    relu=False,
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
                    relu=False,
                ),
                *create_conv_block(
                    in_channels=expansion_size * out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    relu=False,
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
                    relu=False,
                ),
                *create_conv_block(
                    in_channels=expansion_size * in_channels,
                    out_channels=expansion_size * in_channels,
                    kernel_size=3,
                    groups=in_channels,
                    stride=1,
                    padding=1,
                    relu=False,
                ),
                *create_conv_block(
                    in_channels=expansion_size * in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    relu=False,
                ),
            )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return nn.ReLU()(out + self.shortcut(x))


# endregion
