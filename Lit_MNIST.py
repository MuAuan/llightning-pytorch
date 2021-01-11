import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms

from torch.optim import Adam

class LitMNIST(LightningModule):

    def __init__(self, data_dir='./'):
        super().__init__()
        self.data_dir=data_dir
        self.transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.nll_loss(y, t)
        preds = torch.argmax(y, dim=1)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(y,t), prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)    
        
    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        mnist_full =MNIST(self.data_dir, train=True, transform=self.transform)
        n_train = int(len(mnist_full)*0.8)
        n_val = len(mnist_full)-n_train
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [n_train, n_val])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        # prepare transforms standard to MNIST
        # data
        mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        return DataLoader(mnist_train, batch_size=64)

    def val_dataloader(self):
        mnist_val = MNIST(self.data_dir, train=False, transform=self.transform)
        return DataLoader(mnist_val, batch_size=64)

    def test_dataloader(self):
        mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        return DataLoader(mnist_test, batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pl.seed_everything(0)
net = LitMNIST()
model = net.to(device)
x = torch.randn(1, 1, 28, 28).to(device)
out = model(x)
print(out)

trainer = Trainer(gpus=1, max_epochs=10)
trainer.fit(model)
results = trainer.test()
print(results)

x = torch.randn(1, 1, 28, 28).to(device)
out = model(x)
print(out)