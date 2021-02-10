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
import cv2

#from net_encoder_decoder_vgg16 import Encoder, Decoder
#from net_encoder_decoder_vgg_resnet import Encoder, Decoder
#from net_encoder_decoder_vgg_resnet2 import Encoder, Decoder
from net_colarization_resnet import ColorizationNet

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

class rgb2YCrCb(object):
    def __init__(self):
        self.ts = transforms.ToPILImage()
        self.ts2 = transforms.ToTensor()
        mean, std =[0.5,0.5,0.5], [0.25,0.25,0.25]
        self.ts3 = transforms.Normalize(mean, std)
        pass
    
    def __call__(self, tensor):
        tensor = tensor  / 4 + 0.5     # unnormalize
        orgYCrCb = cv2.cvtColor(np.float32(self.ts(tensor)), cv2.COLOR_BGR2YCR_CB)
        Y, Cr,Cb = cv2.split(orgYCrCb)
        CC = cv2.merge((Cr,Cb))
        CC = np.array(CC).reshape(2,32*8,32*8) #(2,32*2,32*2)
        #print(CC.shape)
        return np.array(CC)
    
    def __repr__(self):
        return self.__class__.__name__
    
class rgb2YCrCb_(object):
    def __init__(self):
        self.ts = transforms.ToPILImage()
        self.ts2 = transforms.ToTensor()
        mean, std =[0.5,0.5,0.5], [0.25,0.25,0.25]
        self.ts3 = transforms.Normalize(mean, std)
        pass
    
    def __call__(self, tensor):
        #tensor = self.ts3(self.ts2(self.ts(tensor)))  / 4 + 0.5     # unnormalize        
        tensor = tensor  / 4 + 0.5     # unnormalize
        orgYCrCb = cv2.cvtColor(np.float32(self.ts(tensor)), cv2.COLOR_BGR2YCR_CB)
        Y, Cr,Cb = cv2.split(orgYCrCb)
        CC = cv2.merge((Cr,Cb))
        Y = np.array(Y).reshape(1,32*8,32*8) #(1,32*2,32*2)
        #print(Y.shape)
        return Y

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_num,train_=True, transform1 = None, transform2 = None,train = True):
                
        self.transform1 = transform1
        self.transform2 = transform2
        self.ts = transforms.ToPILImage()
        self.ts2 = transforms.ToTensor()
        mean, std =[0.5,0.5,0.5], [0.25,0.25,0.25]
        self.ts3 =  transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize((64,64)),
            transforms.Normalize(mean, std),
        ])
        self.train = train_
        
        self.data_dir = './'
        self.data_num = data_num
        self.data = []
        self.label = []

        # download
        CIFAR10(self.data_dir, train=True, download=True)
        self.data =CIFAR10(self.data_dir, train=self.train, transform=self.ts3)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx][0]
        out_label_ =  self.data[idx][1]
        out_label = torch.from_numpy(np.array(out_label_)).long()
        
        if self.transform1:
            out_data1 = self.transform1(out_data)
        if self.transform2:
            out_data2 = self.transform2(out_data)
            
        return out_data, out_data1, out_data2, out_label
    
class LitAutoEncoder(pl.LightningModule):

    def __init__(self, data_dir='./'):
        super().__init__()
        self.ts2 = transforms.ToTensor()
        self.ts = transforms.ToPILImage()
        self.data_dir = data_dir
        self.data_num =50000 #50000
        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.dims = (32*8, 32*8)
        
        self.encoder_decoder = ColorizationNet()
        #self.encoder = Encoder()
        #self.decoder = Decoder()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        
        x = self.encoder_decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        _,x,x_ , y = batch
        x_hat = self.encoder_decoder(x)  ##resnet
        loss = F.mse_loss(x_hat, x_)
        self.log('train_loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        _,x, x_, y = batch
        x_hat = self.encoder_decoder(x)
        loss = F.mse_loss(x_hat, x_)
        self.log('test_loss', loss, prog_bar = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) 
        return optimizer
    
def main():
    ts = transforms.ToPILImage()
    ts2 = transforms.ToTensor()
    mean, std =[0.5,0.5,0.5], [0.25,0.25,0.25]
    ts3 =  transforms.Normalize(mean, std)
    ts4 =  transforms.Resize((256,256))
    meang, stdg =[0.5], [0.25]
    ts5 =  transforms.Normalize(meang, stdg)
    trans2 =  transforms.Compose([
        transforms.Resize((256,256)),
        #transforms.Normalize(mean, std),
        rgb2YCrCb(), #CrCb
    ])
    trans1 =  transforms.Compose([
        transforms.Resize((256,256)),
        #transforms.Normalize(mean, std),
        rgb2YCrCb_(),  #Y
    ])
    dim1 =(256,256)
    dim2 = (1,256,256)
    dim3 = (256,256,2)
    data_num = 50000
    cifar10_full =ImageDataset(data_num, train=True, transform1=trans1, transform2=trans2)
    n_train = int(len(cifar10_full)*0.95)
    n_val = int(len(cifar10_full)*0.04)
    n_test = len(cifar10_full)-n_train -n_val
    cifar10_train, cifar10_val, cifar10_test = torch.utils.data.random_split(cifar10_full, [n_train, n_val, n_test])
    
    trainloader = DataLoader(cifar10_train, shuffle=True, drop_last = True, batch_size=32, num_workers=0)
    valloader = DataLoader(cifar10_val, shuffle=False, batch_size=32, num_workers=0)
    testloader = DataLoader(cifar10_test, shuffle=False, batch_size=32, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #for gpu
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    pl.seed_everything(0)

    # model
    autoencoder = LitAutoEncoder()
    #path_ = './simple_coloring/'
    #PATH = path_+'example_cifar4Ln100_9.ckpt'
    #autoencoder = autoencoder.load_from_checkpoint(PATH)
    
    #autoencoder = LitAutoEncoder()
    autoencoder = autoencoder.to(device) #for gpu
    print(autoencoder)
    summary(autoencoder,dim2)
    
    trainer = pl.Trainer(max_epochs=1, gpus=1, callbacks=[MyPrintingCallback()]) ####epoch
    sk = 0
    for i in range(0,10,1):
        trainer.fit(autoencoder, trainloader, valloader)    
        print('training_finished')
    
        results = trainer.test(autoencoder, testloader)
        print(results)
        if sk%1==0:
            dataiter = iter(trainloader)
            _,images, images_, labels = dataiter.next()
            print(images.shape, images_.shape)

            images0 = []
            for i in range(32):
                print(i, images[i].shape, images_[i].shape)
                YCC_ = cv2.merge((np.array(images[i]).reshape(dim1),np.array(images_[i]).reshape(dim3)))
                images0_ = cv2.cvtColor(YCC_, cv2.COLOR_YCR_CB2BGR)
                images0.append(ts2(images0_/255.))
            # show images 
            imshow(torchvision.utils.make_grid(images0), 'cifar10_results',text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4))) #3
            # print labels
            print(' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))

            path_ = './simple_coloring/'
            PATH = path_+'example_cifar4Ln100_{}.ckpt'.format(sk)
            trainer.save_checkpoint(PATH)

            pretrained_model = autoencoder.load_from_checkpoint(PATH)
            pretrained_model.freeze()
            pretrained_model.eval()

            latent_dim,ver = "Gray2Clolor_resnet", "1_{}".format(sk)  #####save condition
            dataiter = iter(testloader)
            images0,images, images1, labels = dataiter.next() #original, Y, CrCb, label
            # show images
            imshow(torchvision.utils.make_grid(images.reshape(32,1,32*8,32*8)/255.),path_+'1_Y_cifar10_{}_{}'.format(latent_dim,0),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
            # show images0
            imshow(torchvision.utils.make_grid(images0.reshape(32,3,32,32)),path_+'2_original_cifar10_{}_{}'.format(latent_dim,0),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
            # show images0
            imshow(torchvision.utils.make_grid(ts4(images0).reshape(32,3,32*8,32*8)),path_+'3_original_normx2_cifar10_{}_{}'.format(latent_dim,0),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))    
            # show images1
            #imshow(torchvision.utils.make_grid(images1.reshape(32,3,32*2,32*2)),'normalized_images1_cifar10_{}_{}'.format(latent_dim,ver),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))     

            decode_img = pretrained_model.encoder_decoder(images[0:32].to('cpu').reshape(32,1,32*8,32*8)) #3
            #decode_img = pretrained_model.decoder(encode_img)
            decode_img_cpu = decode_img.cpu()
            images2 = []
            for i in range(32):
                print(i, images[i].shape, decode_img_cpu[i].shape)
                YCC_ = cv2.merge((np.array(images[i].reshape(dim1)),np.array(decode_img_cpu[i].reshape(dim3))))
                images2_ = cv2.cvtColor(YCC_, cv2.COLOR_YCR_CB2BGR)
                images2.append(ts3(ts2(images2_/255.)))
                #images2.append(ts2(images2_/255.))
            imshow(torchvision.utils.make_grid(images2), path_+'4_preds_cifar10_{}_{}'.format(latent_dim,ver),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
        sk += 1

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))    
