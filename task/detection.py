from torch import optim, nn
from torchmetrics import Accuracy
import lightning as L
import torch
import torch.functional as F
from models import BasicNN

class BasicDetection(L.LightningModule):
    """Basic Multiclass Classification framework\n
    We assume labels of shape (C)
    """
    def __init__(self, input_height: int, input_width:int, num_classes: int, hidden_dim:int):
        super().__init__()
        self.num_classes = num_classes
        self.model = BasicNN(input_height * input_width, hidden_dim, num_classes).model
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = nn.functional.cross_entropy(input=logits, target=y)
        acc = self.accuracy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = nn.functional.cross_entropy(input=logits, target=y)
        acc = self.accuracy(logits, y)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        return acc
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = nn.functional.cross_entropy(input=logits, target=y)
        acc = self.accuracy(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return acc
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer