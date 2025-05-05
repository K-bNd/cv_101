import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2
from utils import IMAGENET_MEAN, IMAGENET_STD


class VOCDetectionDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for ImageNet-1k using Hugging Face datasets.

    Args:
        data_dir (str): Directory to download the dataset to.
        batch_size (int): Batch size for DataLoader. Defaults to 32.
        num_workers (int): Number of workers for DataLoader. Defaults to 4.
        image_size (int): Target image size (height and width). Defaults to 224.
    """

    def __init__(
        self,
        data_dir: str = "../datasets/vocdetection",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 256,
    ):
        super().__init__()
        # Saves args to self.hparams
        self.save_hyperparameters(
            'data_dir', 'batch_size', 'num_workers', 'image_size')
        # Define transforms
        self.train_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            ),
        ])
        self.test_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            ),
        ])

    def prepare_data(self):
        # download
        VOCDetection(self.hparams.data_dir, year="2012",
                     split="train", download=True)
        VOCDetection(self.hparams.data_dir, year="2012",
                     split="val", download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            vocdetection_full = VOCDetection(
                self.hparams.data_dir, split="train", download=False, transform=self.train_transform)
            self.vocdetection_train, self.vocdetection_val = random_split(
                vocdetection_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.vocdetection_test = VOCDetection(
                self.hparams.data_dir, split="val", download=False, transform=self.test_transform)

        if stage == "predict":
            self.vocdetection_predict = VOCDetection(
                self.hparams.data_dir, train=False, download=False)

    def train_dataloader(self):
        return DataLoader(
            self.vocdetection_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,  # Usually good for GPU training
            persistent_workers=self.hparams.num_workers > 0,  # Avoid worker restart overhead
        )

    def val_dataloader(self):
        return DataLoader(
            self.vocdetection_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,  # Usually good for GPU training
            persistent_workers=self.hparams.num_workers > 0,  # Avoid worker restart overhead
        )

    def test_dataloader(self):
        return DataLoader(
            self.vocdetection_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,  # Usually good for GPU training
            persistent_workers=self.hparams.num_workers > 0,  # Avoid worker restart overhead
        )

    def predict_dataloader(self):
        return DataLoader(
            self.vocdetection_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,  # Usually good for GPU training
            persistent_workers=self.hparams.num_workers > 0,  # Avoid worker restart overhead
        )
