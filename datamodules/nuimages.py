import os.path as osp
from typing import Literal
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import lightning as L
import torchvision
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image
from nuimages import NuImages
from configs.config_models import TrainConfig
# Define standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class NuImagesDataset(VisionDataset):
    def __init__(self, data_dir: str, version: str, task: Literal['semantic_segmentation', 'instance_segmentation', 'object_detection']) -> None:
        super().__init__()
        self.nuim = NuImages(dataroot=data_dir, version=version, lazy=True)
        self.task_type = task

    def __len__(self):
        return len(self.nuim.sample)

    def __getitem__(self, index):
        sample = self.nuim.sample[index]
        key_camera_token = sample['key_camera_token']
        sample_data = self.nuim.get('sample_data', key_camera_token)
        # making sure we load a keyframe
        self.nuim.check_sweeps(sample_data['filename'])
        im_path = osp.join(self.nuim.dataroot, sample_data['filename'])
        image = tv_tensors.Image(Image.open(im_path))
        semantic_mask, instance_mask = self.nuim.get_segmentation(key_camera_token)
        object_anns = [o for o in self.nuim.object_ann if o['sample_data_token'] == key_camera_token]
        bboxes = []
        for obj in object_anns:
            bboxes.append(obj['bbox'])
        match self.task_type:
            case "semantic_segmentation":
                return image, tv_tensors.Mask(semantic_mask)
            case "instance_segmentation":
                return image, tv_tensors.Mask(instance_mask)
            case "object_detection":
                return image, tv_tensors.BoundingBoxes(data=bboxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=image.shape[-2:])
            case _:
                return image, tv_tensors.Mask(semantic_mask)


class NuImagesDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: TrainConfig,
        task: Literal['semantic_segmentation', 'instance_segmentation', 'object_detection'],
        data_dir: str = "datasets/nuimages",
    ):
        super().__init__()
        # Define transforms
        self.config = config
        self.data_dir = data_dir
        self.task = task
    
    def prepare_data(self) -> None:
        train_dir = osp.join(self.data_dir, "v1.0-train")
        val_dir = osp.join(self.data_dir, "v1.0-val")
        test_dir = osp.join(self.data_dir, "v1.0-test")
        self.train_dataset = NuImagesDataset(train_dir, version="v1.0-train", task=self.task)
        self.val_dataset = NuImagesDataset(val_dir, version="v1.0-val", task=self.task)
        self.test_dataset = NuImagesDataset(test_dir, version="v1.0-test", task=self.task)
    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Run setup() first.")
        # For IterableDataset, set shuffle=False in DataLoader; shuffling is handled by .shuffle() on the dataset itself.
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

    def test_dataloader(self):  # Implement if you have a test split/stage
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )