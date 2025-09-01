from typing import Literal, Union
from pydantic import BaseModel


class TrainConfig(BaseModel):
    image_size: Union[int, list[int]] = 224
    num_classes: int = 10
    in_channels: int = 3
    batch_size: int = 512
    start_lr: float = 1e-3
    base_lr: float = 0.1
    base_batch_size: int = 256
    epochs: int = 300
    early_stopping: bool = False
    early_stopping_patience: int = 5
    num_workers: int = 4
    trivial_augment: bool = True
    optimizer: Literal["Adam", "AdamW", "SGD"] = "Adam"
    optimizer_params: dict = {}
    lr_scheduler: Literal["step", "cosine"] = 'cosine'
    lr_scheduler_params: dict = {}
    ignore_index: int = 255

class ImageNetTrainConfig(TrainConfig):
    train_res: int = 224
    val_res: int = 224

class BiSeNetV2TrainConfig(TrainConfig):
    seg_heads_loss_weight: float = 0.4