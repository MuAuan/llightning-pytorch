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

#from net_encoder_decoder_vgg16_1d import Encoder, Decoder #, Latent
#from net_encoder_decoder_latent_vgg16 import Encoder, Decoder #, Latent
from net_simple_cifar10_1d import Encoder, Decoder

def imshow(img,file='', text_=''):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy() #img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(x = 3, y = 2, s = text_, c = "red")
    #plt.imshow(npimg)
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
        out_data = self.ts(self.data[idx][0])
        out_label =  np.array(self.data[idx][1])
        if self.transform1:
            out_data1 = self.transform1(out_data).reshape(3*32*32,)
        if self.transform2:
            out_data2 = self.transform2(out_data)
        return out_data1, out_data2, out_label
    
class LitAutoEncoder(pl.LightningModule):

    def __init__(self, data_dir='./'):
        super().__init__()
        self.data_dir = data_dir
        self.data_num =60000
        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        #self.dims = (3, 32, 32) #(1, 28, 28)
        
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
        x,_ , y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss, prog_bar = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) #list(self.encoder.parameters())+list(self.latent.parameters())+list(self.decoder.parameters()
        return optimizer
    """ 
    def prepare_data(self):
        #self.dataset= ImageDataset(self.data_num, train_ = True, transform1 = self.trans1, transform2 = self.trans2)
        pass

    
    def setup(self, stage=None): #train, val, testデータ分割
        # Assign train/val datasets for use in dataloaders
        #self.dataset= ImageDataset(self.data_num, train_ = True, transform1 = self.trans1, transform2 = self.trans2)
        cifar10_full =ImageDataset(self.data_num, train=True, transform1=self.trans1, transform2=self.trans2)
        n_train = int(len(cifar10_full[0])*0.8)
        print(n_train)
        n_val = len(cifar10_full[0])-n_train
        self.cifar10_train, self.cifar10_val = torch.utils.data.random_split(cifar10_full, [n_train, n_val])
        self.cifar10_test = ImageDataset(self.data_num, train=False, transform1=self.trans1, transform2=self.trans2)
    
    def train_dataloader(self):
        self.trainloader = DataLoader(self.cifar10_train, shuffle=True, drop_last = True, batch_size=32, num_workers=0)
        # get some random training images
        return self.trainloader
    
    def val_dataloader(self):
        return DataLoader(self.cifar10_val, shuffle=False, batch_size=32, num_workers=0)
    
    def test_dataloader(self):
        self.testloader = DataLoader(self.cifar10_test, shuffle=False, batch_size=32, num_workers=0)
        return self.testloader
    """
    
def main():    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #for gpu
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    pl.seed_everything(0)
    # model
    trans1 = torchvision.transforms.ToTensor()
    trans2 =  torchvision.transforms.ToTensor() #torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])
    ts = torchvision.transforms.ToPILImage()

    data_num=50000   
    dataset = ImageDataset(data_num, transform1=trans1, transform2 = trans2)

    n_train = int(len(dataset)*0.8)
    n_val = int(len(dataset)*0.1)
    n_test = len(dataset)-n_train -n_val

    train, val, test = random_split(dataset,[n_train, n_val, n_test])
    print('type(train)',type(train))
    train_dataloader = DataLoader(train, batch_size = 32, shuffle= True)
    val_dataloader = DataLoader(val, batch_size = 32, shuffle= False)
    test_dataloader = DataLoader(test, batch_size = 32, shuffle= False)
    print(type(train_dataloader))
    """
    for out_data, out_label in train_loader:
        print(len(out_label),out_label)
        for i in range(1):
            image =  out_data[i].reshape(3,32,32)
            #image_gray = out_data2[i]
            im = ts(image)
            #im_gray = ts(image_gray)
            print(out_label[i])
            #plt.imshow(np.array(im_gray),  cmap='gray')
            #plt.title('{}'.format(out_label[i]))
            #plt.pause(1)
            #plt.clf()
            plt.imshow(np.array(im))
            plt.title('{}'.format(out_label[i]))
            plt.pause(1)
            plt.clf()
    plt.close()
    """
    autoencoder = LitAutoEncoder()
    autoencoder = autoencoder.to(device) #for gpu
    print(autoencoder)
    summary(autoencoder.encoder,(3*32*32,))
    #summary(autoencoder.decoder,(256,8,8)) #(256,4,4))
    #summary(autoencoder.latent,(256,4,4)) #(256,4,4))
    summary(autoencoder,(3*32*32,))
    
    trainer = pl.Trainer(max_epochs=1, gpus=1, callbacks=[MyPrintingCallback()])
    
    trainer.fit(autoencoder, train_dataloader, val_dataloader)    
    print('training_finished')
    trainer.test(autoencoder, test_dataloader) 

    dataiter = iter(train_dataloader)
    images, _, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images.reshape(32,3,32,32)),'cifar10_initial',text_='original')
    # print labels
    print(' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
    results = trainer.test(autoencoder)
    print(results)

    dataiter = iter(val_dataloader)
    images, _, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images.reshape(32,3,32,32)), 'cifar10_results',text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
    # print labels
    print(' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))

    # torchscript
    #torch.jit.save(autoencoder.to_torchscript(), "model_cifar10.pt")
    trainer.save_checkpoint("example_cifar10.ckpt")

    PATH = 'example_cifar10.ckpt'
    pretrained_model = autoencoder.load_from_checkpoint(PATH)
    pretrained_model.freeze()
    pretrained_model.eval()

    latent_dim,ver = 16384*2, 12803
    dataiter = iter(val_dataloader)
    images, _, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images.reshape(32,3,32,32)),'original_images_cifar10_{}_{}'.format(latent_dim,ver),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))

    encode_img = pretrained_model.encoder(images[0:32].to('cpu').reshape(32,3*32*32,))
    decode_img = pretrained_model.decoder(encode_img)
    imshow(torchvision.utils.make_grid(decode_img.cpu().reshape(32,3,32,32)), 'original_autoencode_preds_cifar10_{}_{}'.format(latent_dim,ver),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))    
