import torch.nn as nn
import torch

class BasicCNN(nn.Module):
    """LeNet-5 Architecture"""
    def __init__(self, in_channels:int, num_classes:int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, kernel_size=5, padding=2, out_channels=6),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, kernel_size=5, out_channels=16),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return self.classifier(x)
        