from typing import Literal
from models.cifar_resnet import cifar_resnet1202
from task import BasicClassification, BasicSegmentation, BiSeNetV2Segmentation
from models import LeNet, BasicNN, VGG16, SegNet, ResNet34, ResNet50, BiSeNetV2, cifar_resnet20, cifar_resnet32, cifar_resnet44, cifar_resnet56, cifar_resnet110
from datamodules import (
    CIFAR10DataModule,
    MNISTDataModule,
    OxfordIITDataModule,
    ImagenetteDataModule,
    ImageNetDataModule,
    VOCSegmentationDataModule
)
import lightning as L
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    OnExceptionCheckpoint,
)
from argparse import ArgumentParser
from configs import TrainConfig, ImageNetTrainConfig, BiSeNetV2TrainConfig
from yaml import load, FullLoader


def pick_dataset(
    dataset: str, config: TrainConfig | ImageNetTrainConfig | BiSeNetV2TrainConfig
) -> tuple[L.LightningDataModule, Literal["classification", "segmentation"]]:
    """Init datamodule based on the dataset name
    Args:
        dataset (str): The name of the dataset
        config (TrainConfig): config info
    Returns:
        datamodule (L.LightningDataModule): The datamodule
        task_type (Literal["classification", "segmentation"]): The type of task
    """
    task_type = "classification"
    match dataset:
        case "cifar10":
            datamodule = CIFAR10DataModule(config)
        case "imagenette":
            datamodule = ImagenetteDataModule(config)
        case "mnist":
            datamodule = MNISTDataModule(config)
        case "oxford":
            datamodule = OxfordIITDataModule(config)
            task_type = "segmentation"
        case "imagenet":
            datamodule = ImageNetDataModule(config)
        case "voc_seg":
            datamodule = VOCSegmentationDataModule(config=config)
            task_type = "segmentation"
        case _:
            raise NotImplementedError(
                "The chosen dataset is invalid, please choose from the following: cifar10, imagenette, mnist, oxford"
            )

    return datamodule, task_type


def pick_model(model: str, in_channels: int, num_classes: int) -> nn.Module:
    """Init model based on the model name"""
    match model:
        case "lenet":
            return LeNet(in_channels, num_classes)
        case "vgg16":
            return VGG16(num_classes=num_classes)
        case "segnet":
            return SegNet(in_channels, num_classes)
        case 'bisenetv2':
            return BiSeNetV2(in_channels, num_classes)
        case "resnet34":
            return ResNet34(in_channels, num_classes)
        case "resnet50":
            return ResNet50(in_channels, num_classes)
        case "cifar_resnet_20":
            return cifar_resnet20(in_channels, num_classes)
        case "cifar_resnet_32":
            return cifar_resnet32(in_channels, num_classes)
        case "cifar_resnet_44":
            return cifar_resnet44(in_channels, num_classes)
        case "cifar_resnet_56":
            return cifar_resnet56(in_channels, num_classes)
        case "cifar_resnet_110":
            return cifar_resnet110(in_channels, num_classes)
        case "cifar_resnet_1202":
            return cifar_resnet1202(in_channels, num_classes)
        case _:
            raise NotImplementedError(
                "The chosen model is invalid, please choose from the following: lenet, basic_nn, vgg16, segnet, resnet34, resnet50, cifar_resnet_{20, 32, 44, 56, 110, 1202}"
            )


def get_config(model: str, dataset: str) -> ImageNetTrainConfig | TrainConfig | BiSeNetV2TrainConfig:
    config_class = None
    match [dataset, model]:
        case ["imagenet", _]:
            config_class = ImageNetTrainConfig
        case [_, 'bisenetv2']:
            config_class = BiSeNetV2TrainConfig
        case _:
            config_class = TrainConfig
    with open(f"configs/{dataset}.yaml", "r") as f:
        return config_class(**load(f, Loader=FullLoader))


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--upload_model", type=bool, default=False)
    parser.add_argument("--hf_username", type=str, default="kevin-nd")

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    config = get_config(args.model, args.dataset)
    datamodule, task_type = pick_dataset(args.dataset, config)
    model = pick_model(args.model, config.in_channels, config.num_classes)
    task = None
    match [task_type, args.model]:
        case ["classification", _]:
            task = BasicClassification(config)
        case ['segmentation', 'bisenetv2']:
            task = BiSeNetV2Segmentation(config)
        case ["segmentation", _]:
            task = BasicSegmentation(config)
        case _:
            raise NotImplementedError()

    task.select_model(model)
    config_dict = config.model_dump()
    config_dict["model_name"] = args.model
    wandb_logger = WandbLogger(project=args.dataset.capitalize())
    wandb_logger.experiment.config.update(config_dict)
    wandb_logger.watch(model)
    # wandb_logger.watch(task)
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-val_loss{val/loss:.2f}-val_top5{val/top5:.2f}",
        monitor="val/loss",
        mode="min",
        save_top_k=5,
    )
    callbacks: list[Callback] = [
        LearningRateMonitor("epoch"),
        checkpoint_callback,
    ]
    if config.early_stopping:
        callbacks.append(
            EarlyStopping("val/loss", patience=config.early_stopping_patience)
        )

    # these steps are necessary to get the dataloader info for logging purposes
    datamodule.prepare_data()
    datamodule.setup("fit")
    trainer = L.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=len(datamodule.train_dataloader()),
    )
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(task, datamodule=datamodule)

    if args.upload_model:
        task.model.push_to_hub(f"{args.hf_username}/{args.model}_{args.dataset}")
