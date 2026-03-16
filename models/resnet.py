import torch
import torch.nn as nn

from utils import BottleneckBlock, ResidualBlock, create_conv_block

from .model import ModelImplem


class ResNet50(
    ModelImplem,
    pipeline_tag="image-classification",
    tags=["arxiv:1512.03385"],
):
    def __init__(self, in_channels: int = 3, num_classes=10) -> None:
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        self.encoder = ResNet50._build_encoder(in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1000),
            nn.Identity()
            if num_classes == 1000
            else nn.Linear(in_features=1000, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.encoder(x)))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(x))

    @staticmethod
    def _build_encoder(in_channels: int = 3) -> nn.Sequential:
        return nn.Sequential(
            # region conv1_x (112x112)
            *create_conv_block(
                in_channels=in_channels,
                out_channels=64,
                stride=2,
                kernel_size=7,
                padding=3,
            ),
            # endregion
            # region conv2_x (56x56)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # endregion
            BottleneckBlock(
                in_channels=64,
                reduce_dim=64,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=256,
                reduce_dim=64,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=256,
                reduce_dim=64,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # endregion
            # region conv3_x (28x28)
            BottleneckBlock(
                in_channels=256,
                reduce_dim=128,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=512,
                reduce_dim=128,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=512,
                reduce_dim=128,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=512,
                reduce_dim=128,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # endregion
            # region conv4_x (14x14)
            BottleneckBlock(
                in_channels=512,
                reduce_dim=256,
                out_channels=1024,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=1024,
                reduce_dim=256,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=1024,
                reduce_dim=256,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=1024,
                reduce_dim=256,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=1024,
                reduce_dim=256,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=1024,
                reduce_dim=256,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # endregion
            # region conv5_x (7x7)
            BottleneckBlock(
                in_channels=1024,
                reduce_dim=512,
                out_channels=2048,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=2048,
                reduce_dim=512,
                out_channels=2048,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BottleneckBlock(
                in_channels=2048,
                reduce_dim=512,
                out_channels=2048,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # endregion
        )


class ResNet34(
    ModelImplem,
    pipeline_tag="image-classification",
    tags=["arxiv:1512.03385"],
):
    def __init__(self, in_channels: int = 3, num_classes=10) -> None:
        super(ResNet34, self).__init__()
        self.num_classes = num_classes
        self.encoder = ResNet34._build_encoder(in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=1000),
            nn.Identity()
            if num_classes == 1000
            else nn.Linear(in_features=1000, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.encoder(x)))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(x))

    @staticmethod
    def _build_encoder(in_channels: int = 3) -> nn.Sequential:
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
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
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
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
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
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            ResidualBlock(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ResidualBlock(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            # endregion
        )
