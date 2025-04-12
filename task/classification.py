from torch import optim, nn
from torchmetrics import Accuracy
import lightning as L
import torch
from typing import Callable, Optional


class BasicClassification(L.LightningModule):
    """Basic Multiclass Classification framework\n
    We assume labels of shape (C)
    """

    def __init__(self, num_classes: int, optimizer: str = "Adam"):
        super().__init__()
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.preprocessing = None
        self.loss_fn = nn.CrossEntropyLoss()

    def select_model(self, model: nn.Module, preprocessing: Optional[Callable] = None):
        self.model = model
        self.preprocessing = preprocessing

    def forward(self, x):
        x = self.preprocessing(x) if self.preprocessing else x
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.preprocessing(x) if self.preprocessing else x
        logits = self.model(x)
        loss = self.loss_fn(input=logits, target=y)
        acc = self.accuracy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.preprocessing(x) if self.preprocessing else x
        logits = self.model(x)
        loss = self.loss_fn(input=logits, target=y)
        acc = self.accuracy(logits, y)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        return acc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.preprocessing(x) if self.preprocessing else x
        logits = self.model(x)
        loss = self.loss_fn(input=logits, target=y)
        acc = self.accuracy(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer)(self.parameters())
        return optimizer
