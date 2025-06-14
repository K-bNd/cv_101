import torch
import torch.nn as nn
from utils import (
    create_conv_block,
    GatherExpansionBlock,
    StemBlock,
    BilateralGuidedAggregation,
)
from huggingface_hub import PyTorchModelHubMixin

from utils.conv_utils import ContextEmbeddingBlock


class BiSeNetV2(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="image-classification",
    license="mit",
    tags=["arxiv:2004.02147"],
    repo_url="https://github.com/K-bNd/cv_101",
):
    """BiSeNetV2 architecture"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.bga = BilateralGuidedAggregation()
        self.detail_branch = nn.Sequential(
            # S-1
            *create_conv_block(
                in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            *create_conv_block(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            # S-2
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=2
            ),
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            # S-3
            *create_conv_block(
                in_channels=64, out_channels=128, kernel_size=3, stride=2
            ),
            *create_conv_block(
                in_channels=128, out_channels=128, kernel_size=3, stride=1
            ),
            *create_conv_block(
                in_channels=128, out_channels=128, kernel_size=3, stride=1
            ),
        )
        self.semantic_branch = nn.Sequential(
            # S-1 + S-2
            StemBlock(out_channels=16),
            # S-3
            GatherExpansionBlock(
                in_channels=3, out_channels=32, stride=2, expansion_size=6
            ),
            GatherExpansionBlock(
                in_channels=32, out_channels=32, stride=1, expansion_size=6
            ),
            # S-4
            GatherExpansionBlock(
                in_channels=32, out_channels=64, stride=2, expansion_size=6
            ),
            GatherExpansionBlock(
                in_channels=64, out_channels=64, stride=1, expansion_size=6
            ),
            # S-5
            GatherExpansionBlock(
                in_channels=64, out_channels=128, stride=2, expansion_size=6
            ),
            GatherExpansionBlock(
                in_channels=128, out_channels=128, stride=1, expansion_size=6
            ),
            GatherExpansionBlock(
                in_channels=128, out_channels=128, stride=1, expansion_size=6
            ),
            GatherExpansionBlock(
                in_channels=128, out_channels=128, stride=1, expansion_size=6
            ),
            ContextEmbeddingBlock(in_channels=128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail_output = self.detail_branch(x)
        semantic_output = self.semantic_branch(x)
        return self.bga(detail_output, semantic_output)
