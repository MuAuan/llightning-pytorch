import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST,CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torchsummary import summary


class LitCIFAR10(pl.LightningModule):
    
    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 32, 32)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Conv2d(3, 256, 5),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 5),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1924, 2),
            nn.BatchNorm2d(1924),
            nn.Linear(1924 * 2 * 2, 160),
            nn.Linear(160, 10)
        )
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
       

    def forward(self, x):
        x = self.model(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full =CIFAR10(self.data_dir, train=True, transform=self.transform)
            n_train = int(len(cifar_full)*0.8)
            n_val = len(cifar_full)-n_train
            self.cifar_train, self.cifar_val = torch.utils.data.random_split(cifar_full, [n_train, n_val])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=32)
        
model = LitCIFAR10()
#summary(model,(3,32,32))
trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)
trainer.fit(model)

trainer.test()