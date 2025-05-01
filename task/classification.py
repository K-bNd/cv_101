from torch import optim, nn
from torchmetrics import Accuracy
import lightning as L
import torch
from typing import Callable, Optional


class BasicClassification(L.LightningModule):
    """Basic Multiclass Classification framework\n
    We assume labels of shape (C)
    """

    def __init__(
        self,
        num_classes: int,
        early_stopping_patience: int = 10,
        optimizer: str = "Adam",
        start_learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.early_stopping_patience = early_stopping_patience
        self.optimizer = optimizer
        self.start_learning_rate = start_learning_rate
        self.weight_decay = weight_decay
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.preprocessing = None
        self.loss_fn = nn.CrossEntropyLoss()

    def select_model(self, model: nn.Module, preprocessing: Optional[Callable] = None):
        self.model = model
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
        optimizer = getattr(optim, self.optimizer)(self.parameters(), lr=self.start_learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, patience=self.early_stopping_patience // 3
                ),
                "monitor": "val/loss",
            },
        }
