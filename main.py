import torch
from torch import utils
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from task import BasicClassification
from models import BasicCNN, BasicNN
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, Callback


def main_mnist():
    classifier = BasicClassification(num_classes=10)
    # ffn = BasicNN(in_features=28*28, hidden_features=800, out_features=10)
    cnn = BasicCNN(in_channels=1)
    classifier.select_model(cnn)
    train_dataset = MNIST(
        "./mnist",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    test_dataset = MNIST(
        "./mnist",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    train_dataloader = utils.data.DataLoader(
        train_dataset, num_workers=7, batch_size=512
    )
    test_dataset, val_dataset = utils.data.random_split(
        test_dataset, [0.8, 0.2], torch.Generator().manual_seed(1)
    )
    test_dataloader = utils.data.DataLoader(test_dataset, num_workers=7, batch_size=64)
    val_dataloader = utils.data.DataLoader(val_dataset, num_workers=7, batch_size=64)
    wandb_logger = WandbLogger(project="MNIST")
    wandb_logger.watch(classifier)
    callbacks: list[Callback] = [EarlyStopping("val/loss", patience=10)]
    trainer = L.Trainer(max_epochs=500, logger=wandb_logger, callbacks=callbacks)
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.test(classifier, dataloaders=test_dataloader)
    return


if __name__ == "__main__":
    main_mnist()
