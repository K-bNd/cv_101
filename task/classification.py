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
        warmup_epochs = int(self.config.epochs * 0.05)
        main_epochs = int(self.config.epochs * 0.7)
        final_epochs = int(self.config.epochs * 0.25)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1e-4, total_iters=warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.ConstantLR(
            optimizer=optimizer, factor=1.0, total_iters=main_epochs
        )
        
        match self.config.last_lr_scheduler:
            case "cosine":
                last_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=final_epochs
                )
            case "step":
                last_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10)
            case "plateau":
                last_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.SequentialLR(
                    optimizer=optimizer,
                    schedulers=[warmup_scheduler, main_scheduler, last_scheduler],
                    milestones=[warmup_epochs, main_epochs + warmup_epochs],
                ),
                "monitor": "val/loss",
            },
        }
