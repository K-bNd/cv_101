import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import Imagenette
from torchvision.transforms import v2

from configs.config_models import TrainConfig


class ImagenetteDataModule(L.LightningDataModule):
    def __init__(self, config: TrainConfig, data_dir: str = "datasets/imagenette"):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((256, 256)),
                v2.RandomCrop((config.image_size, config.image_size)),
                (
                    v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)
                    if config.auto_augment
                    else v2.Identity()
                ),
                v2.RandAugment() if config.rand_augment else v2.Identity(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.test_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((256, 256)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def prepare_data(self):
        # download
        Imagenette(self.data_dir, split="train", download=True)
        Imagenette(self.data_dir, split="val", download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            imagenette_full = Imagenette(
                self.data_dir, split="train", download=False, transform=self.train_transform
            )
            self.train_dataset, self.val_dataset = random_split(
                imagenette_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = Imagenette(
                self.data_dir, split="val", download=False, transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,  # Usually good for GPU training
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
