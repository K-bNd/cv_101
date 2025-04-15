import torch
from torch import utils
from torchvision.datasets import MNIST, Imagenette, CIFAR10
from torchvision.transforms import v2
from task import BasicClassification
from models import LeNet, BasicNN, VGG16
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, Callback, LearningRateMonitor
from argparse import ArgumentParser


def main_cifar10(batch_size: int = 128, early_stopping_patience: int = 10):
    classifier = BasicClassification(
        num_classes=10, early_stopping_patience=early_stopping_patience
    )
    vgg16 = VGG16(num_classes=10)
    classifier.select_model(vgg16)
    train_dataset = CIFAR10(
        "./datasets/cifar10",
        train=True,
        download=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # normalize to ImageNet values
            ]
        ),
    )
    test_dataset = CIFAR10(
        "./datasets/cifar10",
        train=False,
        download=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # normalize to ImageNet values
            ]
        ),
    )
    train_dataloader = utils.data.DataLoader(
        train_dataset, num_workers=7, batch_size=batch_size
    )
    test_dataset, val_dataset = utils.data.random_split(
        test_dataset, [0.8, 0.2], torch.Generator().manual_seed(1)
    )
    test_dataloader = utils.data.DataLoader(
        test_dataset, num_workers=7, batch_size=batch_size
    )
    val_dataloader = utils.data.DataLoader(
        val_dataset, num_workers=7, batch_size=batch_size
    )
    wandb_logger = WandbLogger(project="CIFAR10")
    wandb_logger.watch(classifier)
    callbacks: list[Callback] = [
        EarlyStopping("val/loss", patience=early_stopping_patience),
        LearningRateMonitor("epoch"),
    ]
    trainer = L.Trainer(
        max_epochs=500,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=len(train_dataloader),
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.test(classifier, dataloaders=test_dataloader)
    print(f"Complete accuracy over training run = {classifier.accuracy.compute()}")
    return


def main_imagenette(batch_size: int = 128, early_stopping_patience: int = 10):
    classifier = BasicClassification(
        num_classes=10, early_stopping_patience=early_stopping_patience
    )
    vgg16 = VGG16(num_classes=10)
    classifier.select_model(vgg16)
    train_dataset = Imagenette(
        "./datasets/imagenette",
        split="train",
        download=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # normalize to ImageNet values
            ]
        ),
    )
    test_dataset = Imagenette(
        "./datasets/imagenette",
        split="val",
        download=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # normalize to ImageNet values
            ]
        ),
    )
    train_dataloader = utils.data.DataLoader(
        train_dataset, num_workers=7, batch_size=batch_size
    )
    test_dataset, val_dataset = utils.data.random_split(
        test_dataset, [0.8, 0.2], torch.Generator().manual_seed(1)
    )
    test_dataloader = utils.data.DataLoader(
        test_dataset, num_workers=7, batch_size=batch_size
    )
    val_dataloader = utils.data.DataLoader(
        val_dataset, num_workers=7, batch_size=batch_size
    )
    wandb_logger = WandbLogger(project="Imagenette")
    wandb_logger.watch(classifier)
    callbacks: list[Callback] = [
        EarlyStopping("train/loss", patience=early_stopping_patience),
        LearningRateMonitor("epoch"),
    ]
    trainer = L.Trainer(
        max_epochs=500,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=len(train_dataloader),
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.test(classifier, dataloaders=test_dataloader)
    print(f"Complete accuracy over training run = {classifier.accuracy.compute()}")
    return


def main_mnist(batch_size: int = 512, early_stopping_patience: int = 10):
    classifier = BasicClassification(num_classes=10)
    # ffn = BasicNN(in_features=28*28, hidden_features=800, out_features=10)
    # cnn = BasicCNN(in_channels=1)
    vgg16 = VGG16(num_classes=10)
    classifier.select_model(vgg16)
    train_dataset = MNIST(
        "./datasets/mnist",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    test_dataset = MNIST(
        "./datasets/mnist",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    train_dataloader = utils.data.DataLoader(
        train_dataset, num_workers=7, batch_size=batch_size
    )
    test_dataset, val_dataset = utils.data.random_split(
        test_dataset, [0.8, 0.2], torch.Generator().manual_seed(1)
    )
    test_dataloader = utils.data.DataLoader(test_dataset, num_workers=7, batch_size=64)
    val_dataloader = utils.data.DataLoader(val_dataset, num_workers=7, batch_size=64)
    wandb_logger = WandbLogger(project="MNIST")
    wandb_logger.watch(classifier)
    callbacks: list[Callback] = [
        EarlyStopping("train/loss", patience=early_stopping_patience),
        LearningRateMonitor("epoch"),
    ]
    trainer = L.Trainer(max_epochs=500, logger=wandb_logger, callbacks=callbacks)
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.test(classifier, dataloaders=test_dataloader)
    return


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    main_cifar10(
        batch_size=args.batch_size, early_stopping_patience=args.early_stopping_patience
    )
