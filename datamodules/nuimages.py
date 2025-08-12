import os.path as osp
from typing import Literal, Optional, Callable
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import lightning as L
import torchvision
from torchvision.transforms import v2
from torchvision.datasets.utils import verify_str_arg
from torchvision import tv_tensors
from PIL import Image
from nuimages import NuImages
from configs.config_models import TrainConfig


class NuImagesDataset(VisionDataset):
    def __init__(
        self,
        data_dir: str,
        version: str,
        task: Literal[
            "semantic_segmentation", "instance_segmentation", "object_detection"
        ],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            data_dir, transform=transform, target_transform=target_transform
        )
        self.task_type = verify_str_arg(
            task,
            "task",
            ["semantic_segmentation", "instance_segmentation", "object_detection"],
        )
        self.version = verify_str_arg(
            version, "version", ["v1.0-train", "v1.0-val", "v1.0-test"]
        )
        self.nuim = NuImages(dataroot=data_dir, version=version, lazy=True)

    def __len__(self):
        return len(self.nuim.sample)

    def __getitem__(self, index):
        sample = self.nuim.sample[index]
        key_camera_token = sample["key_camera_token"]
        sample_data = self.nuim.get("sample_data", key_camera_token)
        # making sure we load a keyframe
        self.nuim.check_sweeps(sample_data["filename"])
        im_path = osp.join(self.nuim.dataroot, sample_data["filename"])
        image = tv_tensors.Image(Image.open(im_path).convert("RGB"))
        semantic_mask, instance_mask = self.nuim.get_segmentation(key_camera_token)
        object_anns = [
            o
            for o in self.nuim.object_ann
            if o["sample_data_token"] == key_camera_token
        ]
        bboxes = []
        h, w = image.shape[-2:]
        for obj in object_anns:
            bboxes.append(obj["bbox"])

        if self.transform is not None:
            image = self.transform(image)
        label = None
        match self.task_type:
            case "semantic_segmentation":
                label = tv_tensors.Mask(semantic_mask)
            case "instance_segmentation":
                label = tv_tensors.Mask(instance_mask)
            case "object_detection":
                label = tv_tensors.BoundingBoxes.__new__(
                    cls=tv_tensors.BoundingBoxes,
                    data=bboxes,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(h, w),
                )
            case _:
                label = tv_tensors.Mask(semantic_mask)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class NuImagesDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: TrainConfig,
        task: Literal[
            "semantic_segmentation", "instance_segmentation", "object_detection"
        ],
        data_dir: str = "datasets/nuimages",
    ):
        super().__init__()
        # Define transforms
        self.config = config
        self.data_dir = data_dir
        self.task: Literal[
            "semantic_segmentation", "instance_segmentation", "object_detection"
        ] = task

        self.train_transform = None
        self.test_transform = None
        self.val_transform = None

    def prepare_data(self) -> None:
        train_dir = osp.join(self.data_dir, "v1.0-train")
        val_dir = osp.join(self.data_dir, "v1.0-val")
        test_dir = osp.join(self.data_dir, "v1.0-test")
        self.train_dataset = NuImagesDataset(
            train_dir, version="v1.0-train", task=self.task
        )
        self.val_dataset = NuImagesDataset(val_dir, version="v1.0-val", task=self.task)
        self.test_dataset = NuImagesDataset(
            test_dir, version="v1.0-test", task=self.task
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,  # Usually good for GPU training
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )
