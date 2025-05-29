from torch import optim, nn
from torchmetrics import Accuracy
import lightning as L
import torch
from typing import Callable, Optional
from huggingface_hub import PyTorchModelHubMixin

from configs.config_models import TrainConfig


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
        self.loss_fn = nn.CrossEntropyLoss()

    def select_model(self, model: nn.Module, preprocessing: Optional[Callable] = None):
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
        logits = self.model(x)
        loss = self.loss_fn(input=logits, target=y)
        preds = self.softmax(logits)
        acc = self.accuracy(preds, y)
        top5 = self.top5(preds, y)
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
        optimizer = getattr(optim, self.config.optimizer)(
            self.parameters(), lr=self.config.start_lr, **self.config.optimizer_params
        )
        match self.config.lr_scheduler:
            case "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, **self.config.lr_scheduler_params
                )
            case "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, **self.config.lr_scheduler_params
                )
            case "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, **self.config.lr_scheduler_params
                )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
