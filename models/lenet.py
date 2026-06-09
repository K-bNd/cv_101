import torch
import torch.nn as nn

from .model import ModelImplem


class LeNet(ModelImplem, pipeline_tag="image-classification"):
    """LeNet-5 Architecture"""

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, kernel_size=5, padding=2, out_channels=6),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, kernel_size=5, out_channels=16),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=num_classes),
            nn.Softmax(dim=1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Xavier/Glorot init — matched to LeNet's Sigmoid activations."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.classifier(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
