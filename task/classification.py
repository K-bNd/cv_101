from torch import optim, nn
from torchmetrics import Accuracy
import lightning as L
import torch
from typing import Callable, Optional
from huggingface_hub import PyTorchModelHubMixin

class BasicClassification(L.LightningModule):
    """Basic Multiclass Classification framework\n
    We assume labels of shape (C)
    """

    def __init__(
        self,
        num_classes: int,
        optimizer: str = "SGD",
        start_learning_rate: float = 1e-3,
        optimizer_specific_args: dict = {"weight_decay": 1e-5, "momentum": 0.9},
        warmup_epochs: int = 5,
        warmup_decay: float = 0.01,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.start_learning_rate = start_learning_rate
        self.optimizer_specific_args = optimizer_specific_args
        self.warmup_epochs = warmup_epochs
        self.warmup_decay = warmup_decay
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.preprocessing = None
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
        self.log("train/top1", acc, prog_bar=True)
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
        self.log("test/top1", acc, prog_bar=True)
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
        self.log("val/top1", acc, prog_bar=True)
        self.log("val/top5", top5, prog_bar=True)
        return acc

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer)(self.parameters(), lr=self.start_learning_rate, **self.optimizer_specific_args)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[
                    optim.lr_scheduler.LinearLR(start_factor=self.warmup_decay),
                    optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.trainer.max_epochs - self.warmup_epochs)
                ], milestones=[self.warmup_epochs]),
            },
        }
