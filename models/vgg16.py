import torch
import torch.nn as nn
from utils import create_conv_block


class VGG16(nn.Module):
    """VGG-16 architecture"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # Conv-1
            *create_conv_block(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv-2
            *create_conv_block(
                in_channels=64, out_channels=128, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=128, out_channels=256, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv-3
            *create_conv_block(
                in_channels=256, out_channels=256, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=256, out_channels=256, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=256, out_channels=512, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv-4
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv-5
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=4096),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=1000),
            nn.Linear(in_features=1000, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_output = self.conv(x)
        return self.dense(conv_output)

    @staticmethod
    def get_encoder_layer():
        return nn.Sequential(
            # Conv-1
            *create_conv_block(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv-2
            *create_conv_block(
                in_channels=64, out_channels=128, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=128, out_channels=256, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv-3
            *create_conv_block(
                in_channels=256, out_channels=256, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=256, out_channels=256, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=256, out_channels=512, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv-4
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv-5
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=512, out_channels=512, kernel_size=3, stride=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
