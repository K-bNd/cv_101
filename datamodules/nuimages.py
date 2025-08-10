import os.path as osp
from typing import Literal
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import lightning as L
import torchvision
from torchvision.transforms import v2
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

    def __getitem__(self, index):
        sample = self.nuim.sample[index]
        key_camera_token = sample['key_camera_token']
        sample_data = self.nuim.get('sample_data', key_camera_token)
        self.nuim.check_sweeps(sample_data['filename'])
        im_path = osp.join(self.nuim.dataroot, sample_data['filename'])
        im = Image.open(im_path)
        semantic_mask, instance_mask = self.nuim.get_segmentation(key_camera_token)
        object_anns = [o for o in self.nuim.object_ann if o['sample_data_token'] == key_camera_token]
        match self.task_type:
            case "semantic_segmentation":
                return im, semantic_mask
            case "instance_segmentation":
                return im, instance_mask
            case "object_detection":
                return im, object_anns # TODO: figure out proper format for bboxes
            case _:
                return im, semantic_mask


class NuImagesDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: TrainConfig,
        data_dir: str = "datasets/nuimages",
    ):
        super().__init__()
        # Define transforms
        self.config = config
        self.data_dir = data_dir
    
    def prepare_data(self) -> None:
        self.nuim = NuImages(dataroot=self.data_dir)
    