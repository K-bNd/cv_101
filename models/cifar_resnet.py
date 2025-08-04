import torch.nn as nn
from utils import create_conv_block, ResidualBlock
from huggingface_hub import PyTorchModelHubMixin


class CIFAR_ResNet(nn.Module, PyTorchModelHubMixin, pipeline_tag="image-classification", license="mit", tags=["arxiv:1512.03385"], repo_url="https://github.com/K-bNd/cv_101"):
    def __init__(self, num_blocks: list[int], in_channels: int = 3, num_classes: int = 10):
        super(CIFAR_ResNet, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Sequential(
            *create_conv_block(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        )
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classfier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=num_classes),
        )

    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                ResidualBlock(
                    in_channels=self.in_channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    identity_shortcut=True,
                )
            )
            self.in_channels = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        pool = self.pool(out)
        logits = self.classfier(pool)
        return logits


def cifar_resnet20(in_channels: int = 3, num_classes: int = 10):
    return CIFAR_ResNet([3, 3, 3], in_channels=in_channels, num_classes=num_classes)


def cifar_resnet32(in_channels: int = 3, num_classes: int = 10):
    return CIFAR_ResNet([5, 5, 5], in_channels=in_channels, num_classes=num_classes)


def cifar_resnet44(in_channels: int = 3, num_classes: int = 10):
    return CIFAR_ResNet([7, 7, 7], in_channels=in_channels, num_classes=num_classes)


def cifar_resnet56(in_channels: int = 3, num_classes: int = 10):
    return CIFAR_ResNet([9, 9, 9], in_channels=in_channels, num_classes=num_classes)


def cifar_resnet110(in_channels: int = 3, num_classes: int = 10):
    return CIFAR_ResNet([18, 18, 18], in_channels=in_channels, num_classes=num_classes)


def cifar_resnet1202(in_channels: int = 3, num_classes: int = 10):
    return CIFAR_ResNet([200, 200, 200], in_channels=in_channels, num_classes=num_classes)
