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
    def __init__(self, expansion_size: int = 6) -> None:
        super(SemanticBranch, self).__init__()
        self.segmentation_heads: list[nn.Module] = [
            SegmentationHead(in_channels=16, scale_factor=4),  # after stem block
            SegmentationHead(in_channels=32, scale_factor=8),  # after S3
            SegmentationHead(in_channels=64, scale_factor=16),  # after S4
            SegmentationHead(in_channels=128, scale_factor=32),  # after S5-4
        ]
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

    def forward(self, x: torch.Tensor, inference: bool = True):
        x1 = self.s1_s2(x)
        x2 = self.s3(x1)
        x3 = self.s4(x2)
        x4 = self.s5_4(x3)
        x5 = self.s5_5(x4)
        if inference:
            return x5
        else:
            return x5, [
                self.segmentation_heads[0](x1),
                self.segmentation_heads[1](x2),
                self.segmentation_heads[2](x3),
                self.segmentation_heads[3](x4),
            ]


class BiSeNetV2(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="image-classification",
    license="mit",
    tags=["arxiv:2004.02147"],
    repo_url="https://github.com/K-bNd/cv_101",
):
    """BiSeNetV2 architecture"""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.final_segmentation_head = SegmentationHead(
            in_channels=128,
            hidden_channels=128,
            scale_factor=4,
            num_classes=num_classes,
        )
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
        self.semantic_branch = SemanticBranch()

    def forward(self, x: torch.Tensor, inference: bool = True):
        detail_output = self.detail_branch(x)
        if inference:
            semantic_output = self.semantic_branch(x, inference)
            x1 = self.bga(detail_output, semantic_output)

            return self.final_segmentation_head(x1)
        else:
            semantic_output, seg_heads = self.semantic_branch(x, inference)
            x1 = self.bga(detail_output, semantic_output)

            return self.final_segmentation_head(x1), seg_heads
