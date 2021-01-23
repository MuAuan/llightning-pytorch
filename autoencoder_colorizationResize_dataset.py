import os
import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision.datasets import CIFAR10 #MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchsummary import summary

#from net_encoder_decoder2D import Encoder, Decoder
from net_encoder_decoder1D2DResize import Encoder, Decoder
#from net_simple_cifar10_2d import Encoder, Decoder

def imshow(img,file='', text_=''):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy() #img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(x = 3, y = 2, s = text_, c = "red")
    plt.pause(3)
    if file != '':
        plt.savefig(file+'.png')
    plt.close()

from pytorch_lightning.callbacks import Callback    
class MyPrintingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        print('')

class MyAddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)  
    

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_num,train_=True, transform1 = None, transform2 = None,train = True):
                
        self.transform1 = transform1
        self.transform2 = transform2
        self.ts = torchvision.transforms.ToPILImage()
        self.ts2 = transform=transforms.ToTensor()
        self.train = train_
        
        self.data_dir = './'
        self.data_num = data_num
        self.data = []
        self.label = []

        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
        self.data =CIFAR10(self.data_dir, train=self.train, transform=self.ts2)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx][0]
        out_label =  np.array(self.data[idx][1])
        if self.transform1:
            out_data1 = self.transform1(out_data)
        if self.transform2:
            out_data2 = self.transform2(out_data)
        return out_data, out_data1, out_data2, out_label
    
class LitAutoEncoder(pl.LightningModule):

    def __init__(self, data_dir='./'):
        super().__init__()
        self.data_dir = data_dir
        self.data_num =50000
        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.dims = (32*2, 32*2) 
        self.mean, self.std =[0.5,0.5,0.5], [0.25,0.25,0.25]
        self.trans2 = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.Resize(self.dims)
        ])
        self.trans1 =  torchvision.transforms.Compose([
            torchvision.transforms.Normalize(self.mean, self.std),
            MyAddGaussianNoise(0., 0.5),
            torchvision.transforms.Grayscale()
        ])
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.encoder(x)
        #enbedding = self.latent(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        _,x,x_ , y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x_)
        self.log('train_loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        _,x, x_, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x_)
        self.log('test_loss', loss, prog_bar = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) 
        return optimizer
    
    def setup(self, stage=None): #train, val, testデータ分割
        # Assign train/val datasets for use in dataloaders
        cifar10_full =ImageDataset(self.data_num, train=True, transform1=self.trans1, transform2=self.trans2)
        n_train = int(len(cifar10_full)*0.8)
        n_val = int(len(cifar10_full)*0.1)
        n_test = len(cifar10_full)-n_train -n_val
        
        self.cifar10_train, self.cifar10_val, self.cifar10_test = torch.utils.data.random_split(cifar10_full, [n_train, n_val, n_test])
        
    
    def train_dataloader(self):
        self.trainloader = DataLoader(self.cifar10_train, shuffle=True, drop_last = True, batch_size=32, num_workers=0)
        return self.trainloader
    
    def val_dataloader(self):
        self.valloader = DataLoader(self.cifar10_val, shuffle=False, batch_size=32, num_workers=0)
        return self.valloader
    
    def test_dataloader(self):
        self.testloader = DataLoader(self.cifar10_test, shuffle=False, batch_size=32, num_workers=0)
        return self.testloader
    
    
def main():    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #for gpu
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    pl.seed_everything(0)

    # model
    autoencoder = LitAutoEncoder()
    autoencoder = autoencoder.to(device) #for gpu
    print(autoencoder)
    summary(autoencoder.encoder,(1,32,32))
    summary(autoencoder,(1,32,32))
    
    trainer = pl.Trainer(max_epochs=1, gpus=1, callbacks=[MyPrintingCallback()]) ####epoch
    
    trainer.fit(autoencoder)    
    print('training_finished')
    
    results = trainer.test(autoencoder)
    print(results)

    dataiter = iter(autoencoder.valloader)
    _,images, _, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images.reshape(32,1,32,32)), 'cifar10_results',text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
    # print labels
    print(' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))

    # torchscript
    #torch.jit.save(autoencoder.to_torchscript(), "model_cifar10.pt")
    trainer.save_checkpoint("example_cifar10.ckpt")

    PATH = 'example_cifar10.ckpt'
    pretrained_model = autoencoder.load_from_checkpoint(PATH)
    pretrained_model.freeze()
    pretrained_model.eval()

    latent_dim,ver = "Gray2ClolarizationResize", "1"  #####save condition
    dataiter = iter(autoencoder.valloader)
    images0,images, images1, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images.reshape(32,1,32,32)),'original_images_cifar10_{}_{}'.format(latent_dim,ver),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
    # show images0
    imshow(torchvision.utils.make_grid(images0.reshape(32,3,32,32)),'original_images0_cifar10_{}_{}'.format(latent_dim,ver),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
    # show images1
    imshow(torchvision.utils.make_grid(images1.reshape(32,3,32*2,32*2)),'normalized_images1_cifar10_{}_{}'.format(latent_dim,ver),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))     

    encode_img = pretrained_model.encoder(images[0:32].to('cpu').reshape(32,1,32,32))
    decode_img = pretrained_model.decoder(encode_img)
    imshow(torchvision.utils.make_grid(decode_img.cpu().reshape(32,3,32*2,32*2)), 'autoencode_preds_cifar10_{}_{}'.format(latent_dim,ver),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))    
