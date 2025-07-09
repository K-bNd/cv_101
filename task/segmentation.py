from torch import optim, nn
import lightning as L
import torch
from typing import Callable, Literal, Optional
from torchmetrics import Accuracy
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

from configs.config_models import TrainConfig, BiSeNetV2TrainConfig


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
        self.iou = MeanIoU(num_classes=config.num_classes, input_format=input_format)
        self.accuracy = Accuracy(task='multiclass', num_classes=config.num_classes, ignore_index=255)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255) # 255 is for ambiguous labels in VOC
        self.loss_fn_2 = GeneralizedDiceScore(
            num_classes=config.num_classes, input_format=input_format
        )

    def select_model(self, model: nn.Module, postprocessing: Optional[Callable] = None):
        self.model = model
        self.postprocessing = postprocessing

    def forward(self, x, inference=True):
        logits = self.model(x, inference)
        preds = self.postprocessing(logits) if self.postprocessing else logits
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x, inference=False)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        iou = self.iou(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        self.log("train/iou", iou, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x, inference=False)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        iou = self.iou(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        self.log("test/iou", iou, prog_bar=True)
        return acc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x, inference=False)
        print(x.shape, y.shape)
        print(y.min(), y.max())
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        iou = self.iou(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/iou", iou, prog_bar=True)
        return acc

    def predict_step(self, batch):
        return self(batch)

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


class BiSeNetV2Segmentation(BasicSegmentation):
    def __init__(
        self,
        config: BiSeNetV2TrainConfig,
        input_format: Literal["one-hot", "index"] = "index",
    ):
        super(BiSeNetV2Segmentation, self).__init__(config, input_format)
        self.config = config

    def forward(self, x, inference=True):
        logits, seg_heads = self.model(x, inference)
        preds = self.postprocessing(logits) if self.postprocessing else logits
        return preds, seg_heads

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds, seg_heads = self(x, inference=False)
        loss = self.loss_fn(preds, y)
        aux_loss = 0.0
        for seg_head in seg_heads:
            seg_head_preds = self.postprocessing(seg_head) if self.postprocessing else seg_head
            aux_loss += self.loss_fn(seg_head_preds, y)
        loss += self.config.seg_heads_loss_weight * aux_loss
        acc = self.accuracy(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds, seg_heads = self(x, inference=False)
        loss = self.loss_fn(preds, y)
        aux_loss = 0.
        for seg_head in seg_heads:
            seg_head_preds = self.postprocessing(seg_head) if self.postprocessing else seg_head
            aux_loss += self.loss_fn(seg_head_preds, y)
        loss += self.config.seg_heads_loss_weight * aux_loss
        acc = self.accuracy(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds, seg_heads = self(x, inference=False)
        loss = self.loss_fn(preds, y)
        aux_loss = 0.
        for seg_head in seg_heads:
            seg_head_preds = self.postprocessing(seg_head) if self.postprocessing else seg_head
            aux_loss += self.loss_fn(seg_head_preds, y)
        loss += self.config.seg_heads_loss_weight * aux_loss
        acc = self.accuracy(torch.argmax(preds, dim=1, keepdim=True), y[:, None, :, :])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return acc
