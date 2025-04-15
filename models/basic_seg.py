import torch.nn as nn
import torch
from utils import create_conv_block


class SegNet(nn.Module):
    """SegNet Architecture"""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            *create_conv_block(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"
            ),
        )
        self.conv2 = nn.Sequential(
            *create_conv_block(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )
        self.conv3 = nn.Sequential(
            *create_conv_block(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )
        self.conv4 = nn.Sequential(
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )
        self.conv5 = nn.Sequential(
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax2d()

        self.deconv1 = nn.Sequential(
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        self.deconv2 = nn.Sequential(
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        self.deconv3 = nn.Sequential(
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        self.deconv4 = nn.Sequential(
            *create_conv_block(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            *create_conv_block(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        self.deconv5 = nn.Sequential(
            *create_conv_block(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"
            ),
            *create_conv_block(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # region encoder
        x1 = self.conv1(x)
        x1_pool, x1_indices = self.pool(x1)
        x2 = self.conv2(x1_pool)
        x2_pool, x2_indices = self.pool(x2)
        x3 = self.conv3(x2_pool)
        x3_pool, x3_indices = self.pool(x3)

        x4 = self.conv4(x3_pool)
        x4_pool, x4_indices = self.pool(x4)

        x5 = self.conv5(x4_pool)
        x5_pool, x5_indices = self.pool(x5)
        # endregion

        # region decoder
        print(
            "Encoding done",
            x5_indices.size(),
            x4_indices.size(),
            x3_indices.size(),
            x2_indices.size(),
            x1_indices.size(),
        )
        print("Shapes ", x5.size(), x4.size(), x3.size(), x2.size(), x1.size())
        x6_unpool = self.unpool(x5_pool, x5_indices, output_size=x5.size())
        x6 = self.deconv1(x6_unpool)
        x7_unpool = self.unpool(x6, x4_indices, output_size=x4.size())
        x7 = self.deconv2(x7_unpool)
        x8_unpool = self.unpool(x7, x3_indices, output_size=x3.size())
        x8 = self.deconv3(x8_unpool)
        x9_unpool = self.unpool(x8, x2_indices, output_size=x2.size())
        x9 = self.deconv4(x9_unpool)
        x10_unpool = self.unpool(x9, x1_indices, x1.size())
        x10 = self.deconv5(x10_unpool)
        print("This should be an image ?", x10.size())
        preds = self.softmax(x10)
        return preds


if __name__ == "__main__":
    x = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    model = SegNet()
    print(model(x).shape)
