from typing import Literal, Union

from pydantic import BaseModel


class TrainConfig(BaseModel):
    image_size: Union[int, list[int]] = 224
    num_classes: int = 10
    in_channels: int = 3
    batch_size: int = 512
    start_lr: float = 0.1
    base_lr: float = 0.1
    base_batch_size: int = 256
    linear_scaling_lr: bool = False
    epochs: int = 300
    early_stopping: bool = False
    early_stopping_patience: int = 5
    num_workers: int = 4
    trivial_augment: bool = True
    random_erasing: float = 0.1
    optimizer: Literal["Adam", "AdamW", "SGD"] = "SGD"
    optimizer_params: dict = {}
    lr_scheduler: Literal["step", "cosine"] = "cosine"
    lr_scheduler_params: dict = {}
    ignore_index: int = 255
    label_smoothing: float = 0.0
    gradient_clip_val: float = 0.0


class ImageNetTrainConfig(TrainConfig):
    train_res: int = 224
    val_res: int = 224
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0


class BiSeNetV2TrainConfig(TrainConfig):
    seg_heads_loss_weight: float = 0.4


class ViTTrainConfig(TrainConfig):
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    qkv_bias: bool = False
    qk_norm: bool = False
    num_registers: int = 0
    rope: bool = False
    reg_theta: int = 10000
    layer_scale: bool = True
    init_scale: float = 1e-4
    ape: bool = True
