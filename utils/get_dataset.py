import torch
from torch import utils
from torchvision.datasets import MNIST, Imagenette, CIFAR10, OxfordIIITPet
from torchvision.transforms import v2

def get_cifar10(batch_size: int = 128):
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
    return train_dataloader, val_dataloader, test_dataloader


def get_imagenette(batch_size: int = 128, early_stopping_patience: int = 10):
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
    return train_dataloader, val_dataloader, test_dataloader


def get_mnist(batch_size: int = 512, early_stopping_patience: int = 10):
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
    return train_dataloader, val_dataloader, test_dataloader


def get_oxford(batch_size: int = 128, early_stopping_patience: int = 10):
    image_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # normalize to ImageNet values
        ]
    )

    target_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.long),
            v2.Lambda(lambda x: torch.squeeze(x) - 1) # indexes start at 1 according to readme, squeese for CE loss
        ]
    )
    trainval_dataset = OxfordIIITPet(
        "./datasets/oxford-iit-pets",
        split="trainval",
        target_types="segmentation",
        download=True,
        transform=image_transform,
        target_transform=target_transform
    )
    test_dataset = OxfordIIITPet(
        "./datasets/oxford-iit-pets",
        split="test",
        target_types="segmentation",        
        download=True,
        transform=image_transform,
        target_transform=target_transform
    )

    train_dataset, val_dataset = utils.data.random_split(
        trainval_dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
    )
    test_dataloader = utils.data.DataLoader(
        test_dataset, num_workers=7, batch_size=batch_size
    )
    train_dataloader = utils.data.DataLoader(
        train_dataset, num_workers=7, batch_size=batch_size
    )
    val_dataloader = utils.data.DataLoader(
        val_dataset, num_workers=7, batch_size=batch_size
    )
    return train_dataloader, val_dataloader, test_dataloader
