import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2, InterpolationMode
from urllib.request import urlretrieve
from configs.config_models import TrainConfig
import zipfile
import os

class ADE20kDataModule(L.LightningDataModule):
    def __init__(self, config: TrainConfig, data_dir: str = "datasets/ade20k"):
        super().__init__()
        self.data_dir = data_dir
        self.config = config

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(config.image_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def prepare_data(self):
        # download
        urlretrieve("http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip", "datasets/ade20k/ade20k.zip")
        with zipfile.ZipFile("datasets/ade20k/ade20k.zip", 'r') as zip_ref:
            zip_ref.extractall("datasets/ade20k")        
        os.remove("datasets/ade20k/ade20k.zip")

    def setup(self, stage: str):
        raise NotImplementedError("Setup phase not implemented yet")

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
