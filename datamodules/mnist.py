import lightning as L
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "../datasets/mnist", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.batch_size = batch_size


    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, download=False, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, download=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)
