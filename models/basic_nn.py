import torch
import torch.nn as nn

from .model import ModelImplem


class BasicNN(ModelImplem, pipeline_tag="image-classification"):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("BasicNN has no separable encoder layer")
