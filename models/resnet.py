import torch
import torch.nn as nn
from utils import create_conv_block, ResidualBlock


class ResNet34(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes=10) -> None:
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            # region conv1_x
            *create_conv_block(
            in_channels=in_channels,
            out_channels=64,
            stride=2,
            kernel_size=7,
            padding=3,),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # endregion

        )
        # region conv2_x
        self.conv2_1 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # endregion
        # region conv3_x
        self.conv3_1 = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # endregion
        # region conv4_x
        self.conv4_1 = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_4 = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_5 = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_6 = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # endregion
        # region cpnv5_x
        self.conv5_1 = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(512)
        self.classfier = nn.Linear(in_features=512, out_features=num_classes)



    def forward(self, x: torch.Tensor):
        return x

    @staticmethod
    def get_encoder_layer() -> nn.Sequential:
        return nn.Sequential()
