from typing import Literal
from task import BasicClassification, BasicSegmentation
from models import LeNet, BasicNN, VGG16, SegNet, ResNet34, ResNet50
from datamodules import CIFAR10DataModule, MNISTDataModule, OxfordIITDataModule, ImagenetteDataModule, ImageNetDataModule
import lightning as L
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, Callback, LearningRateMonitor
from argparse import ArgumentParser
from datasets import load_dataset, Dataset, IterableDataset


def pick_dataset(dataset: str) -> tuple[L.LightningDataModule, int, int, Literal["classification", "segmentation"]]:
    """Init datamodule based on the dataset name
        Args:
            dataset (str): The name of the dataset
        Returns:
            datamodule (L.LightningDataModule): The datamodule
            in_channels (int): The number of input channels
            num_classes (int): The number of classes
            task_type (Literal["classification", "segmentation"]): The type of task
    """
    task_type = "classification"
    num_classes = 10
    in_channels = 3
    match dataset:
        case "cifar10":
            datamodule = CIFAR10DataModule(batch_size=args.batch_size)
        case "imagenette":
            datamodule = ImagenetteDataModule(batch_size=args.batch_size)
        case "mnist":
            datamodule = MNISTDataModule(batch_size=args.batch_size)
            in_channels = 1
        case "oxford":
            datamodule = OxfordIITDataModule(batch_size=args.batch_size)
            task_type = "segmentation"
            num_classes = 3
        case "imagenet":
            datamodule = ImageNetDataModule(batch_size=args.batch_size)
            num_classes = 1000
        case _:
            raise NotImplementedError(
                "The chosen dataset is invalid, please choose from the following: cifar10, imagenette, mnist, oxford")

    return datamodule, in_channels, num_classes, task_type


def pick_model(model: str, in_channels: int, num_classes: int) -> nn.Module:
    """Init model based on the model name"""
    match model:
        case "lenet":
            return LeNet(in_channels, num_classes)
        case "vgg16":
            return VGG16(num_classes=num_classes)
        case "segnet":
            return SegNet(in_channels, num_classes)
        case "resnet34":
            return ResNet34(in_channels, num_classes)
        case "resnet50":
            return ResNet50(in_channels, num_classes)
        case _:
            raise NotImplementedError(
                "The chosen model is invalid, please choose from the following: lenet, basic_nn, vgg16, segnet, resnet34, resnet50")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--model", type=str, default="resnet50")

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    datamodule, in_channels, num_classes, task_type = pick_dataset(
        args.dataset)
    model = pick_model(args.model, in_channels, num_classes)
    task = None
    match task_type:
        case "classification":
            task = BasicClassification(num_classes=num_classes)
        case "segmentation":
            task = BasicSegmentation(num_classes=num_classes)
        case _:
            raise NotImplementedError()

    task.select_model(model)
    wandb_logger = WandbLogger(project=args.dataset.capitalize())
    # wandb_logger.watch(task)
    callbacks: list[Callback] = [
        EarlyStopping("val/loss", patience=args.early_stopping_patience),
        LearningRateMonitor("epoch"),
    ]
    # these steps are necessary to get the dataloader info for logging purposes
    datamodule.prepare_data()
    datamodule.setup("fit")
    trainer = L.Trainer(
        max_epochs=500,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=len(datamodule.train_dataloader()) if args.dataset != "imagenet" else 50,
    )
    trainer.fit(
        model=task,
        datamodule=datamodule
    )
    trainer.test(task, datamodule=datamodule)
