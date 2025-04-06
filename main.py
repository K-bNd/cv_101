import torch
from torch import utils
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from task.detection import BasicDetection
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

def main():
    model = BasicDetection(input_width=28, input_height=28, num_classes=10, hidden_dim=800)
    train_dataset = MNIST("./mnist", train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    test_dataset = MNIST("./mnist", train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    train_dataloader = utils.data.DataLoader(train_dataset, num_workers=7, batch_size=512)
    test_dataset, val_dataset = utils.data.random_split(test_dataset, [0.8, 0.2], torch.Generator().manual_seed(1))
    test_dataloader = utils.data.DataLoader(test_dataset, num_workers=7, batch_size=64)
    val_dataloader = utils.data.DataLoader(val_dataset, num_workers=7, batch_size=64)
    wandb_logger = WandbLogger(project="MNIST")
    wandb_logger.watch(model)
    callbacks = [EarlyStopping('val/loss', patience=10)]
    trainer = L.Trainer(max_epochs=500, logger=wandb_logger, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
