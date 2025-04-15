from typing import Union
import torch
import torch.nn as nn


def create_conv_block(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: Union[int, str] = 0
) -> list[nn.Module]:
    return [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    ]
