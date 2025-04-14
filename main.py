import torch
from torch import utils
from torchvision.datasets import MNIST, Imagenette
from torchvision.transforms import v2
from task import BasicClassification
from models import BasicCNN, BasicNN, VGG16
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, Callback
from argparse import ArgumentParser


def main_imagenette(batch_size=128):
    classifier = BasicClassification(num_classes=10)
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
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize to ImageNet values
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
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize to ImageNet values
            ]
        ),
    )
    train_dataloader = utils.data.DataLoader(
        train_dataset, num_workers=7, batch_size=batch_size
    )
    test_dataset, val_dataset = utils.data.random_split(
        test_dataset, [0.8, 0.2], torch.Generator().manual_seed(1)
    )
    test_dataloader = utils.data.DataLoader(test_dataset, num_workers=7, batch_size=batch_size)
    val_dataloader = utils.data.DataLoader(val_dataset, num_workers=7, batch_size=batch_size)
    wandb_logger = WandbLogger(project="Imagenette")
    wandb_logger.watch(classifier)
    callbacks: list[Callback] = [EarlyStopping("train/loss", patience=10)]
    trainer = L.Trainer(max_epochs=500, logger=wandb_logger, callbacks=callbacks, log_every_n_steps=len(train_dataloader))
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.test(classifier, dataloaders=[test_dataloader, val_dataloader, train_dataloader])
    return


def main_mnist():
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
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--batch_size", type=int, default=128)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    torch.cuda.memory._record_memory_history()
    main_imagenette(batch_size=args.batch_size)
    torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
