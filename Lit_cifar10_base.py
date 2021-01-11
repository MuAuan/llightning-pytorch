import argparse
import time
import numpy as np

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
import torch.optim as optim
from torchsummary import summary

import matplotlib.pyplot as plt
import tensorboard

#from sam import SAM

from net_cifar10 import Net_cifar10
#from net_vgg16 import VGG16

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.pause(1)
    plt.close()

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar,LearningRateMonitor

class MyPrintingCallback(Callback):

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('Trainer is init now')
        
    def on_epoch_end(self, trainer, pl_module):
        #print("pl_module = ",pl_module)
        #print("trainer = ",trainer)
        print('')
        
    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')    

class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar
    
class Net(pl.LightningModule):
    def __init__(self, data_dir='./'):
        super().__init__()
        self.data_dir = data_dir

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 32, 32)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.net = Net_cifar10()
        #self.net = VGG16()
        
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, t =batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        #print('Ir =',self.optimizers().param_groups[0]['lr'])
        self.log('test_loss', loss, on_step = False, on_epoch = True)
        self.log('test_acc', self.val_acc(y, t), on_step = False, on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        preds = torch.argmax(y, dim=1)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(y,t), prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(self.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)}
        scheduler = {'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)}

        """
        optimizer = SAM(self.net.parameters(), base_optimizer, lr=0.1, momentum=0.9)
                # first forward-backward pass
        loss = loss_function(output, model(input))  # use this loss for any training statistics
        loss.backward()
        optimizer.first_step(zero_grad=True)
        # second forward-backward pass
        loss_function(output, model(input)).backward()
        optimizer.second_step(zero_grad=True)
        """
        print("CF;Ir = ", optimizer.param_groups[0]['lr'])
        return [optimizer], [scheduler]
    
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
        #classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        trainloader = DataLoader(self.cifar_train, shuffle=True, drop_last = True, batch_size=32, num_workers=0)
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
        return trainloader
    
    def val_dataloader(self):
        return DataLoader(self.cifar_val, shuffle=False, batch_size=32, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.cifar_test, shuffle=False, batch_size=32, num_workers=0)

   
def main():
    '''
    main
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #for gpu
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    pl.seed_everything(0)
    # model
    net = Net()
    model = net.to(device)  #for gpu
    summary(model,(3,32,32))
    early_stopping = EarlyStopping('val_acc')  #('val_loss'
    bar = LitProgressBar(refresh_rate = 10, process_position = 1)
    #lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(gpus=1, max_epochs=60,callbacks=[MyPrintingCallback(), bar]) # progress_bar_refresh_rate=10
    trainer.fit(model)
    results = trainer.test()
    print(results)
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))    
