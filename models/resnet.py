import torch
import torch.nn as nn
from utils import create_conv_block, ResidualBlock


class ResNet34(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes=10) -> None:
        super(ResNet34, self).__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            # region conv1_x
            *create_conv_block(
                in_channels=in_channels,
                out_channels=64,
                stride=2,
                kernel_size=7,
                padding=3,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # endregion
            # region conv2_x
            ResidualBlock(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            # endregion
            # region conv3_x
            ResidualBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            # endregion
            # region conv4_x
            ResidualBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            # endregion
            # region conv5_x
            ResidualBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            # endregion
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classfier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=1000),
            nn.Identity() if num_classes == 1000 else nn.Linear(in_features=1000, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor):
        enc = self.encoder(x)
        pool = self.pool(enc)
        logits = self.classfier(pool)
        return logits

    @staticmethod
    def get_encoder_layer(in_channels: int = 3) -> nn.Sequential:
        return nn.Sequential(
            # region conv1_x
            *create_conv_block(
                in_channels=in_channels,
                out_channels=64,
                stride=2,
                kernel_size=7,
                padding=3,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # endregion
            # region conv2_x
            ResidualBlock(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            # endregion
            # region conv3_x
            ResidualBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            # endregion
            # region conv4_x
            ResidualBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            # endregion
            # region conv5_x
            ResidualBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            # endregion
        )
