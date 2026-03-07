from typing import Callable, Optional

import lightning as L
from torch import nn, optim
from torchmetrics import Accuracy
from torchvision.transforms import v2

from configs.config_models import ImageNetTrainConfig, TrainConfig
from models import ModelImplem


class BasicClassification(L.LightningModule):
    """Basic Multiclass Classification framework\n
    We assume labels of shape (C)
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.top5 = Accuracy(task="multiclass", num_classes=config.num_classes, top_k=5)
        self.preprocessing = None
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        batch_transforms = []
        if isinstance(config, ImageNetTrainConfig) and config.mixup_alpha > 0:
            batch_transforms.append(
                v2.MixUp(alpha=config.mixup_alpha, num_classes=config.num_classes)
            )
        if isinstance(config, ImageNetTrainConfig) and config.cutmix_alpha > 0:
            batch_transforms.append(
                v2.CutMix(alpha=config.cutmix_alpha, num_classes=config.num_classes)
            )
        self.mixup_cutmix = (
            v2.RandomChoice(batch_transforms) if batch_transforms else None
        )

    def select_model(
        self, model: ModelImplem, preprocessing: Optional[Callable] = None
    ):
        self.model = model
        self.model.train(True)
        self.preprocessing = preprocessing

    def forward(self, x):
        x = self.preprocessing(x) if self.preprocessing else x
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
        else:
            x, y = batch
        x = self.preprocessing(x) if self.preprocessing else x
        if self.mixup_cutmix is not None:
            x, y = self.mixup_cutmix(x, y)
        logits = self.model(x)
        loss = self.loss_fn(input=logits, target=y)
        preds = self.softmax(logits)
        hard_y = y.argmax(dim=1) if y.is_floating_point() else y
        acc = self.accuracy(preds, hard_y)
        top5 = self.top5(preds, hard_y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        self.log("train/top5", top5, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
        else:
            x, y = batch
        x = self.preprocessing(x) if self.preprocessing else x
        logits = self.model(x)
        loss = self.loss_fn(input=logits, target=y)
        preds = self.softmax(logits)
        acc = self.accuracy(preds, y)
        top5 = self.top5(preds, y)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        self.log("test/top5", top5, prog_bar=True)
        return acc

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
        else:
            x, y = batch
        x = self.preprocessing(x) if self.preprocessing else x
        logits = self.model(x)
        loss = self.loss_fn(input=logits, target=y)
        preds = self.softmax(logits)
        acc = self.accuracy(preds, y)
        top5 = self.top5(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/top5", top5, prog_bar=True)
        return acc

    def configure_optimizers(self):
        start_lr = (
            self.config.base_lr * self.config.batch_size / self.config.base_batch_size
            if self.config.linear_scaling_lr
            else self.config.start_lr
        )
        optimizer = getattr(optim, self.config.optimizer)(
            self.parameters(), lr=start_lr, **self.config.optimizer_params
        )
        warmup_epochs = min(int(self.config.epochs * 0.1), 5)
        main_epochs = self.config.epochs - warmup_epochs
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1 / warmup_epochs,
            total_iters=warmup_epochs,
        )

        match self.config.lr_scheduler:
            case "cosine":
                main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=main_epochs
                )
            case "step":
                main_scheduler = optim.lr_scheduler.StepLR(
                    optimizer=optimizer, step_size=main_epochs // 3
                )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
