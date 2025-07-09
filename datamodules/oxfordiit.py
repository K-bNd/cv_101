import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2

from configs.config_models import TrainConfig


class OxfordIITDataModule(L.LightningDataModule):
    def __init__(
        self, config: TrainConfig, data_dir: str = "datasets/oxford_iit_pets"
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
                # indexes start at 1 according to readme, squeeze for CE loss
                v2.Lambda(lambda x: torch.squeeze(x) - 1),
            ]
        )

    def prepare_data(self):
        # download
        OxfordIIITPet(
            self.data_dir, split="trainval", target_types="segmentation", download=True
        )
        OxfordIIITPet(
            self.data_dir, split="test", target_types="segmentation", download=True
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            oxford_full = OxfordIIITPet(
                self.data_dir,
                split="trainval",
                target_types="segmentation",
                download=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.train_dataset, self.val_dataset = random_split(
                oxford_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = OxfordIIITPet(
                self.data_dir,
                split="test",
                target_types="segmentation",
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
