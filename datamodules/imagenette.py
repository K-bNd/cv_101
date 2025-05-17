import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import Imagenette
from torchvision.transforms import v2


class ImagenetteDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "../datasets/imagenette", batch_size: int = 32, image_size: int = 224):
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
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        Imagenette(self.data_dir, split="train", download=True)
        Imagenette(self.data_dir, split="val", download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            imagenette_full = Imagenette(
                self.data_dir, split="train", download=False, transform=self.transform)
            self.imagenette_train, self.imagenette_val = random_split(
                imagenette_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.imagenette_test = Imagenette(
                self.data_dir, split="val", download=False, transform=self.transform)

        if stage == "predict":
            self.imagenette_predict = Imagenette(
                self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.imagenette_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.imagenette_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.imagenette_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.imagenette_predict, batch_size=self.batch_size)
