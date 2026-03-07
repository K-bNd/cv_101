import torch
import torch.nn as nn

from utils.conv_utils import ConvNeXtBlock, ConvNeXtDownsample

from .model import ModelImplem

# ConvNeXt-Tiny: depths and dims per stage
_DEPTHS = [3, 3, 9, 3]
_DIMS = [96, 192, 384, 768]
# Stochastic depth rate linearly scaled across all blocks (max 0.1 for Tiny)
_DROP_PATH_MAX = 0.1


class ConvNeXt(
    ModelImplem,
    pipeline_tag="image-classification",
    tags=["arxiv:2201.03545"],
):
    """ConvNeXt-Tiny (arXiv:2201.03545).

    depths=[3,3,9,3], dims=[96,192,384,768]
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super(ConvNeXt, self).__init__()
        self.num_classes = num_classes

        # Linearly spaced drop path rates across all 18 blocks
        total_blocks = sum(_DEPTHS)
        dp_rates = [r.item() for r in torch.linspace(0, _DROP_PATH_MAX, total_blocks)]
        block_idx = 0

        # Stem: patchify Conv + channel-last LayerNorm
        self.stem = nn.Conv2d(in_channels, _DIMS[0], kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm(_DIMS[0])

        # region stage 1
        self.stage1 = nn.Sequential(
            *[ConvNeXtBlock(_DIMS[0], dp_rates[block_idx + i]) for i in range(_DEPTHS[0])]
        )
        block_idx += _DEPTHS[0]
        # endregion

        self.down1 = ConvNeXtDownsample(_DIMS[0], _DIMS[1])

        # region stage 2
        self.stage2 = nn.Sequential(
            *[ConvNeXtBlock(_DIMS[1], dp_rates[block_idx + i]) for i in range(_DEPTHS[1])]
        )
        block_idx += _DEPTHS[1]
        # endregion

        self.down2 = ConvNeXtDownsample(_DIMS[1], _DIMS[2])

        # region stage 3
        self.stage3 = nn.Sequential(
            *[ConvNeXtBlock(_DIMS[2], dp_rates[block_idx + i]) for i in range(_DEPTHS[2])]
        )
        block_idx += _DEPTHS[2]
        # endregion

        self.down3 = ConvNeXtDownsample(_DIMS[2], _DIMS[3])

        # region stage 4
        self.stage4 = nn.Sequential(
            *[ConvNeXtBlock(_DIMS[3], dp_rates[block_idx + i]) for i in range(_DEPTHS[3])]
        )
        # endregion

        # Head: global avg pool → LayerNorm → Linear
        self.head_norm = nn.LayerNorm(_DIMS[3])
        self.head_fc = nn.Linear(_DIMS[3], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, encoding_only: bool = False) -> torch.Tensor:
        # Stem
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)   # (N,C,H,W) → (N,H,W,C)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)   # (N,H,W,C) → (N,C,H,W)

        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)

        if encoding_only:
            return x

        x = x.mean([-2, -1])        # global average pool → (N, C)
        x = self.head_norm(x)
        return self.head_fc(x)

    @staticmethod
    def get_encoder_layer() -> nn.Sequential:
        total_blocks = sum(_DEPTHS)
        dp_rates = [r.item() for r in torch.linspace(0, _DROP_PATH_MAX, total_blocks)]
        return nn.Sequential(
            # region stage 1
            *[ConvNeXtBlock(_DIMS[0], dp_rates[i]) for i in range(_DEPTHS[0])],
            # endregion
            ConvNeXtDownsample(_DIMS[0], _DIMS[1]),
            # region stage 2
            *[ConvNeXtBlock(_DIMS[1], dp_rates[_DEPTHS[0] + i]) for i in range(_DEPTHS[1])],
            # endregion
            ConvNeXtDownsample(_DIMS[1], _DIMS[2]),
            # region stage 3
            *[ConvNeXtBlock(_DIMS[2], dp_rates[_DEPTHS[0] + _DEPTHS[1] + i]) for i in range(_DEPTHS[2])],
            # endregion
            ConvNeXtDownsample(_DIMS[2], _DIMS[3]),
            # region stage 4
            *[ConvNeXtBlock(_DIMS[3], dp_rates[_DEPTHS[0] + _DEPTHS[1] + _DEPTHS[2] + i]) for i in range(_DEPTHS[3])],
            # endregion
        )
