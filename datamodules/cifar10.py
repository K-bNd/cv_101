import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2, InterpolationMode

from configs.config_models import TrainConfig


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, config: TrainConfig, data_dir: str = "datasets/cifar10"):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(config.image_size),
                v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10, interpolation=InterpolationMode.BILINEAR) if config.auto_augment else v2.Identity(),
                v2.RandAugment(interpolation=InterpolationMode.BILINEAR) if config.rand_augment else v2.Identity(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar10_full = CIFAR10(
                self.data_dir, train=True, download=False, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(
                cifar10_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = CIFAR10(
                self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,  # Usually good for GPU training
            persistent_workers=self.config.num_workers > 0,  # Avoid worker restart overhead
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
