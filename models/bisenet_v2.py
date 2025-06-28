from typing import Union
import torch
import torch.nn as nn
from utils import (
    create_conv_block,
    GatherExpansionBlock,
    StemBlock,
    BilateralGuidedAggregation,
    ContextEmbeddingBlock,
    SegmentationHead,
)
from huggingface_hub import PyTorchModelHubMixin


class SemanticBranch(nn.Module):
    """Semantic Branch from the BiSeNetV2 paper"""
    def __init__(self, num_classes: int = 3, expansion_size: int = 6) -> None:
        super(SemanticBranch, self).__init__()
        self.booster = nn.ModuleList([
            SegmentationHead(in_channels=16, scale_factor=4, num_classes=num_classes),  # after stem block
            SegmentationHead(in_channels=32, scale_factor=8, num_classes=num_classes),  # after S3
            SegmentationHead(in_channels=64, scale_factor=16, num_classes=num_classes),  # after S4
            SegmentationHead(in_channels=128, scale_factor=32, num_classes=num_classes),  # after S5-4
        ])
        self.s1_s2 = StemBlock(out_channels=16)
        self.s3 = nn.Sequential(
            GatherExpansionBlock(
                in_channels=16, out_channels=32, stride=2, expansion_size=expansion_size
            ),
            GatherExpansionBlock(
                in_channels=32, out_channels=32, stride=1, expansion_size=expansion_size
            ),
        )
        self.s4 = nn.Sequential(
            GatherExpansionBlock(
                in_channels=32, out_channels=64, stride=2, expansion_size=expansion_size
            ),
            GatherExpansionBlock(
                in_channels=64, out_channels=64, stride=1, expansion_size=expansion_size
            ),
        )
        self.s5_4 = nn.Sequential(
            GatherExpansionBlock(
                in_channels=64,
                out_channels=128,
                stride=2,
                expansion_size=expansion_size,
            ),
            GatherExpansionBlock(
                in_channels=128,
                out_channels=128,
                stride=1,
                expansion_size=expansion_size,
            ),
            GatherExpansionBlock(
                in_channels=128,
                out_channels=128,
                stride=1,
                expansion_size=expansion_size,
            ),
            GatherExpansionBlock(
                in_channels=128,
                out_channels=128,
                stride=1,
                expansion_size=expansion_size,
            ),
        )
        self.s5_5 = ContextEmbeddingBlock(in_channels=128)

    def forward(self, x: torch.Tensor, inference: bool = True) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x1 = self.s1_s2(x)
        x2 = self.s3(x1)
        x3 = self.s4(x2)
        x4 = self.s5_4(x3)
        x5 = self.s5_5(x4)
        if inference:
            return x5, []
        else:
            return x5, [
                self.booster[0](x1),
                self.booster[1](x2),
                self.booster[2](x3),
                self.booster[3](x4),
            ]


class BiSeNetV2(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="image-segmentation",
    license="mit",
    tags=["arxiv:2004.02147"],
    repo_url="https://github.com/K-bNd/cv_101",
):
    """BiSeNetV2 architecture"""

    def __init__(self, in_channels: int = 3, num_classes: int = 3) -> None:
        super().__init__()
        self.semantic_branch = SemanticBranch(num_classes=num_classes)
        self.final_segmentation_head = SegmentationHead(
            in_channels=128,
            hidden_channels=128,
            scale_factor=8,
            num_classes=num_classes,
        )
        self.bga = BilateralGuidedAggregation()
        self.detail_branch = nn.Sequential(
            # S-1
            *create_conv_block(
                in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            *create_conv_block(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            # S-2
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            # S-3
            *create_conv_block(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            *create_conv_block(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            *create_conv_block(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x: torch.Tensor, inference: bool = True) -> tuple[torch.Tensor, list[torch.Tensor]]:
        detail_output = self.detail_branch(x) # H/8 × W/8 x C
        if inference:
            semantic_output, _ = self.semantic_branch(x, inference) # H/32 x W/32 x C
            x1 = self.bga(detail_output, semantic_output) # H/8 x W/8 x C

            return self.final_segmentation_head(x1), []
        else:
            semantic_output, seg_heads = self.semantic_branch.forward(x, inference) # H/32 x W/32 x C
            x1 = self.bga(detail_output, semantic_output) # H/8 x W/8 x C

            return self.final_segmentation_head(x1), seg_heads
