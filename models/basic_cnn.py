import torch.nn as nn
import torch

class BasicCNN(nn.Module):
    def __init__(self, in_channels:int, ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d()
        )