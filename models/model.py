from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from pytorch_lightning import LightningModule


class ModelImplem(
    ABC,
    LightningModule,
    PyTorchModelHubMixin,
    license="mit",
    repo_url="https://github.com/K-bNd/cv_101",
):
    """Base interface for all model implementations.

    Subclasses must:
    - Pass `pipeline_tag` and `tags` in their class definition for HuggingFace Hub metadata
    - Implement `forward`
    - Call `super().__init__()` in `__init__`

    Example:
        class MyModel(
            ModelImplem,
            pipeline_tag="image-classification",
            tags=["arxiv:1234.56789"],
        ):
            def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
                super().__init__()
                ...

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                ...
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @staticmethod
    @abstractmethod
    def get_encoder_layer() -> nn.Sequential: ...
