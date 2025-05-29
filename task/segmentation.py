from torch import optim, nn
from torchmetrics import Accuracy
import lightning as L
import torch
from typing import Callable, Literal, Optional
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

from configs.config_models import TrainConfig


class BasicSegmentation(L.LightningModule):
    """Basic Semantic Semantic Segmentation framework\n
    Expected shapes:
        labels: (N, H, W)
        preds: (N, C, H, W) with class probabilities
    with:
        N: batch_size
        C: number of classes
        H: height
        W: width
    """

    def __init__(
        self,
        config: TrainConfig,
        input_format: Literal["one-hot", "index"] = "index",
    ):
        super().__init__()
        self.postprocessing = None
        self.config = config
        self.accuracy = MeanIoU(
            num_classes=config.num_classes, input_format=input_format
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_2 = GeneralizedDiceScore(
            num_classes=config.num_classes, input_format=input_format
        )

    def select_model(self, model: nn.Module, postprocessing: Optional[Callable] = None):
        self.model = model
        self.postprocessing = postprocessing

    def forward(self, x):
        logits = self.model(x)
        preds = self.postprocessing(logits) if self.postprocessing else logits
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        preds = self.postprocessing(logits) if self.postprocessing else logits
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        preds = self.postprocessing(logits) if self.postprocessing else logits
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        return acc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        preds = self.postprocessing(logits) if self.postprocessing else logits
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
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
