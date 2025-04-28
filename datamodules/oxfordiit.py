import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2


class OxfordIITDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "../datasets/oxford_iit_pets", batch_size: int = 32, image_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])
        self.target_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size),
                v2.ToDtype(torch.long),
                # indexes start at 1 according to readme, squeeze for CE loss
                v2.Lambda(lambda x: torch.squeeze(x) - 1)
            ]
        )
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        OxfordIIITPet(self.data_dir, split="trainval",
                      target_types="segmentation", download=True)
        OxfordIIITPet(self.data_dir, split="test",
                      target_types="segmentation", download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            oxford_full = OxfordIIITPet(self.data_dir, split="trainval", target_types="segmentation",
                                        download=False, transform=self.transform, target_transform=self.target_transform)
            self.oxford_train, self.oxford_val = random_split(
                oxford_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.oxford_test = OxfordIIITPet(self.data_dir, split="test", target_types="segmentation",
                                             download=False, transform=self.transform, target_transform=self.target_transform)

        # Assign predict dataset for use in dataloader(s)
        if stage == "predict":
            self.oxford_predict = OxfordIIITPet(self.data_dir, split="test", target_types="segmentation",
                                             download=False, transform=self.transform, target_transform=self.target_transform)

    def train_dataloader(self):
        return DataLoader(self.oxford_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.oxford_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.oxford_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.oxford_predict, batch_size=self.batch_size)
