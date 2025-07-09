from typing import Callable
import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import v2

from configs.config_models import TrainConfig


def handle_ambiguous_label(num_classes=21) -> Callable:
    """In VOCSegmentation, bordering regions are marked with a `void' label (index 255)
    We shall remap it to background (index 0) for now"""
    def func(label: torch.Tensor):
        label[label > num_classes] = 0.0
        return torch.squeeze(label)
    return func


class VOCSegmentationDataModule(L.LightningDataModule):
    def __init__(
        self, config: TrainConfig, data_dir: str = "datasets/voc_segmentation"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((config.image_size, config.image_size)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.target_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((config.image_size, config.image_size)),
                v2.ToDtype(torch.long),
                # v2.Lambda(handle_ambiguous_label),
            ]
        )

    def prepare_data(self):
        # download
        VOCSegmentation(self.data_dir, image_set="train", download=True)
        VOCSegmentation(self.data_dir, image_set="val", download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            voc_train = VOCSegmentation(
                self.data_dir,
                image_set="train",
                download=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.train_dataset, self.val_dataset = random_split(
                voc_train, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = VOCSegmentation(
                self.data_dir,
                image_set="val",
                download=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )
