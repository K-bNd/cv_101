from typing import Literal
from task import BasicClassification, BasicSegmentation
from models import LeNet, BasicNN, VGG16, SegNet, ResNet34, ResNet50
from datamodules import CIFAR10DataModule, MNISTDataModule, OxfordIITDataModule, ImagenetteDataModule
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, Callback, LearningRateMonitor
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="cifar10")

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    task_type: Literal["classification", "segmentation"] = "classification"

    match args.dataset:
        case "cifar10":
            datamodule = CIFAR10DataModule(batch_size=args.batch_size)
        case "imagenette":
            datamodule = ImagenetteDataModule(batch_size=args.batch_size)
        case "mnist":
            datamodule = MNISTDataModule(batch_size=args.batch_size)
        case "oxford":
            datamodule = OxfordIITDataModule(batch_size=args.batch_size)
            task_type = "segmentation"
        case _:
            raise NotImplementedError(
                "The chosen dataset is invalid, please choose from the following: cifar10, imagenette, mnist, oxford")

    task = None
    match task_type:
        case "classification":
            task = BasicClassification(num_classes=10)
        case "segmentation":
            task = BasicSegmentation(num_classes=3)
        case _:
            raise NotImplementedError()

    resnet50 = ResNet50(num_classes=10)
    task.select_model(resnet50)
    wandb_logger = WandbLogger(project=args.dataset.capitalize())
    wandb_logger.watch(task)
    callbacks: list[Callback] = [
        EarlyStopping("val/loss", patience=args.early_stopping_patience),
        LearningRateMonitor("epoch"),
    ]
    trainer = L.Trainer(
        max_epochs=500,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=len(datamodule.train_dataloader()),
    )
    trainer.fit(
        model=task,
        datamodule=datamodule
    )
    trainer.test(task, datamodule=datamodule)
