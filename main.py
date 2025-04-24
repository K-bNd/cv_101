import torch
from torch import utils
from utils import get_cifar10, get_imagenette, get_mnist, get_oxford
from task import BasicClassification, BasicSegmentation
from models import LeNet, BasicNN, VGG16, SegNet, ResNet34, ResNet50
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, Callback, LearningRateMonitor
from argparse import ArgumentParser


def main_cifar10(batch_size: int = 128, early_stopping_patience: int = 10):
    train_dataloader, val_dataloader, test_dataloader = get_cifar10(batch_size)
    classifier = BasicClassification(
        num_classes=10, early_stopping_patience=early_stopping_patience
    )
    resnet50 = ResNet50(num_classes=10)
    classifier.select_model(resnet50)
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
    train_dataloader, val_dataloader, test_dataloader = get_imagenette(batch_size)
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
    train_dataloader, val_dataloader, test_dataloader = get_mnist(batch_size)
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


def main_oxford(batch_size: int = 128, early_stopping_patience: int = 10):
    segment = BasicSegmentation(num_classes=3)
    model = SegNet(num_classes=3)
    segment.select_model(model)
    train_dataloader, val_dataloader, test_dataloader = get_oxford(batch_size)
    wandb_logger = WandbLogger(project="Oxford IIT Pets")
    wandb_logger.watch(segment)
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
        model=segment,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.test(segment, dataloaders=test_dataloader)
    print(f"Complete accuracy over training run = {segment.accuracy.compute()}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="cifar10")

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    match args.dataset:
        case "cifar10":
            main_cifar10(
                batch_size=args.batch_size,
                early_stopping_patience=args.early_stopping_patience,
            )
        case "imagenette":
            main_imagenette(
                batch_size=args.batch_size,
                early_stopping_patience=args.early_stopping_patience,
            )
        case "mnist":
            main_mnist(
                batch_size=args.batch_size,
                early_stopping_patience=args.early_stopping_patience,
            )
        case "oxford":
            main_oxford(
                batch_size=args.batch_size,
                early_stopping_patience=args.early_stopping_patience,
            )
        case _:
            raise NotImplementedError()
