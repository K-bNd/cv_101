from typing import Literal
from pydantic import BaseModel


class TrainConfig(BaseModel):
    image_size: int = 224
    num_classes: int = 10
    in_channels: int = 3
    batch_size: int = 512
    start_lr: float = 1e-3
    epochs: int = 300
    early_stopping: bool = False
    early_stopping_patience: int = 5
    num_workers: int = 4
    auto_augment: bool = False
    rand_augment: bool = False
    optimizer: Literal["Adam", "AdamW", "SGD"] = "Adam"
    optimizer_params: dict = {}
    lr_scheduler: Literal["step", "cosine", "plateau"] = 'plateau'
    lr_scheduler_params: dict = {}

class ImageNetTrainConfig(TrainConfig):
    train_res: int = 224
    val_res: int = 224

class BiSeNetV2TrainConfig(TrainConfig):
    seg_heads_loss_weight: float = 0.4