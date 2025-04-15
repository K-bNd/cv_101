from torch import optim, nn
from torchmetrics import Accuracy
import lightning as L
import torch
from typing import Callable, Literal, Optional
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore


class BasicSegmentation(L.LightningModule):
    """Basic Semantic Segmentation framework\n
    We assume labels of shape (H, W)
    """

    def __init__(
        self,
        num_classes: int = 3,
        optimizer: str = "Adam",
        input_format: Literal["one-hot", "index"] = "index",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.postprocessing = None
        self.accuracy = MeanIoU(num_classes=num_classes, input_format=input_format)
        self.loss_fn = GeneralizedDiceScore(
            num_classes=num_classes, input_format=input_format
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
        loss = 1 - self.loss_fn(preds, y)
        loss.requires_grad = True
        acc = self.accuracy(preds, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        preds = self.postprocessing(logits) if self.postprocessing else logits
        loss = 1 - self.loss_fn(preds, y)
        acc = self.accuracy(preds, y)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        return acc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        preds = self.postprocessing(logits) if self.postprocessing else logits
        print(preds.shape, preds[0, :, 0, 0])
        loss = 1 - self.loss_fn(preds, y)
        acc = self.accuracy(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer)(self.parameters())
        return optimizer
