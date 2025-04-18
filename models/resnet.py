import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor):
        return x

    @staticmethod
    def get_encoder_layer() -> nn.Sequential:
        return nn.Sequential(

        )
    