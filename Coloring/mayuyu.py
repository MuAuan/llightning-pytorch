import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image


ts = transforms.ToPILImage()
ts2 = transforms.ToTensor()
ts3 = transforms.Grayscale()
mean, std = [0.5,0.5,0.5],[0.25,0.25,0.25]
ts4 = transforms.Normalize(mean, std)

#image0 = cv2.imread('YCC.jpg')
#image0 = cv2.imread('Lenna_(test_image).png')
#image0 = cv2.imread('mayuyu.jpg')
#autoencode_preds_cifar10_Gray2ClolarizationNormalizeResize3LYCC_100.png
#image0 = cv2.imread('autoencode_preds_cifar10_Gray2ClolarizationNormalizeResize3LYCC_100.png')
#Lenna_(test_image).png
"""
image1=cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
npimg =ts(ts4(ts2(image1/255.)))
image_ = np.transpose(npimg, (0,1, 2))
plt.title('normalize')
plt.imshow(image_)
plt.pause(3)
plt.savefig('./YCC/normalize.png')
plt.clf()

image_=ts(image0).convert('L')
plt.title('gray')
plt.imshow(image_)
plt.pause(3)
plt.savefig('./YCC/image_gray.png')
plt.clf()

plt.title('gray_gray')
plt.imshow(image_, cmap='gray')
plt.pause(3)
plt.savefig('./YCC/image_gray_gray.png')
plt.clf()

image_g=ts3(ts(image0))
plt.title('gray_ts')
plt.imshow(image_g)
plt.pause(3)
plt.savefig('./YCC/image_g_gray.png')
plt.clf()

plt.title('gray_ts')
plt.imshow(image_g, cmap = 'gray')
plt.pause(3)
plt.savefig('./YCC/image_g_gray_gray.png')
plt.clf()


image1=cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
plt.title('image1')
plt.imshow(image1)
plt.pause(3)
plt.savefig('./YCC/original.png')
plt.clf()

orgYCrCb = cv2.cvtColor(image1, cv2.COLOR_BGR2YCR_CB)
plt.title('orgYCrCb')
plt.imshow(orgYCrCb)
plt.savefig('./YCC/orgYCrCb.png')
plt.pause(1)
plt.clf()

orgYCrCb_ = cv2.cvtColor(orgYCrCb, cv2.COLOR_YCR_CB2BGR)
plt.title('orgYCrCb_')
plt.imshow(orgYCrCb_)
plt.savefig('./YCC/orgYCrCb_.png')
plt.pause(3)
plt.clf()

Y, Cr,Cb = cv2.split(orgYCrCb)
plt.title('Y')
plt.imshow(Y) #, cmap = 'gray')
plt.savefig('./YCC/Y.png')
plt.pause(1)
plt.clf()

plt.title('Y_gray')
plt.imshow(Y, cmap = 'gray')
plt.savefig('./YCC/Y_gray.png')
plt.pause(1)
plt.clf()

plt.title('Cr')
plt.imshow(Cr) #, cmap = 'gray')
plt.savefig('./YCC/Cr.png') # _gray.png')
plt.pause(1)
plt.clf()

plt.title('Cr_gray')
plt.imshow(Cr, cmap = 'gray')
plt.savefig('./YCC/Cr_gray.png') # _gray.png')
plt.pause(1)
plt.clf()


plt.title('Cb')
plt.imshow(Cb) #, cmap = 'gray')
plt.savefig('./YCC/Cb.png') #_gray.png')
plt.pause(1)
plt.clf()

plt.title('Cb_gray')
plt.imshow(Cb, cmap = 'gray')
plt.savefig('./YCC/Cb_gray.png') #_gray.png')
plt.pause(1)
plt.clf()

Cr_=ts(Cr).convert("RGB")
Cb_ = ts(Cb).convert("RGB")
Y_ = ts(Y).convert('RGB')
#CC = cv2.merge((Y,Cr_,Cb_))

plt.title('Cr_RGB')
plt.imshow(Cr_)
plt.savefig('./YCC/Cr_RGB.png')
plt.pause(3)
plt.clf()

plt.title('Cb_RGB')
plt.imshow(Cb_)
plt.savefig('./YCC/Cb_RGB.png')
plt.pause(3)
plt.clf()

plt.title('Y_RGB')
plt.imshow(Y_)
plt.savefig('./YCC/Y_RGB.png')
plt.pause(3)
plt.clf()


YCC = cv2.merge((Y,Cr,Cb))
orgYCrCb_2 = cv2.cvtColor(YCC, cv2.COLOR_YCR_CB2BGR)

plt.title('YCrCb_merge')
plt.imshow(orgYCrCb_2)
plt.savefig('./YCC/YCC_RGB_merge.png')
plt.pause(3)
plt.clf()

orgLAB = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
plt.title('orgLAB')
plt.imshow(orgLAB)
plt.savefig('./YCC/orgLAB.png')
plt.pause(1)
plt.clf()

L, A,B = cv2.split(orgLAB)
plt.title('L')
plt.imshow(L) #, cmap = 'gray')
plt.savefig('./YCC/L.png')
plt.pause(1)
plt.clf()

plt.title('L_gray')
plt.imshow(L, cmap = 'gray')
plt.savefig('./YCC/L_gray.png')
plt.pause(1)
plt.clf()


print(L.shape,A.shape,B.shape)

plt.title('A')
plt.imshow(A)
plt.savefig('./YCC/A.png')
plt.pause(1)
plt.clf()

plt.title('A_gray')
plt.imshow(A, cmap ='gray')
plt.savefig('./YCC/A_gray.png')
plt.pause(1)
plt.clf()

plt.title('B')
plt.imshow(B)
plt.savefig('./YCC/B.png')
plt.pause(1)
plt.clf()

plt.title('B_gray')
plt.imshow(B, cmap = 'gray')
plt.savefig('./YCC/B_gray.png')
plt.pause(1)
plt.clf()

LAB = cv2.merge((L,A,B))
orgLAB_2 = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)

plt.title('LAB_merge')
plt.imshow(orgLAB_2)
plt.savefig('./YCC/LAB_merge.png')
plt.pause(3)
plt.clf()

X = np.zeros(L.shape,np.uint8)
print(X.shape)

plt.title('X')
plt.imshow(X)
plt.pause(1)
plt.clf()

XAB = cv2.merge((X,A,B))
orgXAB_2 = cv2.cvtColor(XAB, cv2.COLOR_LAB2BGR)

plt.title('orgXAB_2')
plt.imshow(orgXAB_2)
plt.pause(3)
plt.clf()

trans =  torchvision.transforms.Compose([
            #torchvision.transforms.Normalize(self.mean, self.std),
            #torchvision.transforms.Resize(self.dims),
            #MyAddGaussianNoise(0., 0.1),
            torchvision.transforms.Grayscale()
        ])
x = ts3(ts(image1))
plt.title('grayscale')
plt.imshow(x, cmap = 'gray')
plt.pause(3)
plt.clf()

orgYCrCb = cv2.cvtColor(image1, cv2.COLOR_BGR2YCR_CB)
Y, Cr,Cb = cv2.split(orgYCrCb)
plt.title('Y')
plt.imshow(Y, cmap = 'gray')
plt.pause(3)
plt.clf()

xCC = cv2.merge((np.uint8(x),Cr,Cb))
orgxCrCb_2 = cv2.cvtColor(xCC, cv2.COLOR_YCR_CB2BGR)

plt.title('orgxCrCb_2')
plt.imshow(orgxCrCb_2)
plt.savefig('./YCC/orgxCrCb_2.png')
plt.pause(3)
plt.clf()

orgLAB = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
L, A,B = cv2.split(orgLAB)

plt.title('L')
plt.imshow(L, cmap = 'gray')
plt.savefig('./YCC/L_gray.png')
plt.pause(3)
plt.clf()

plt.title('A')
plt.imshow(A, cmap = 'gray')
plt.savefig('./YCC/A_gray.png')
plt.pause(3)
plt.clf()

plt.title('B')
plt.imshow(B, cmap = 'gray')
plt.savefig('./YCC/B_gray.png')
plt.pause(3)
plt.clf()

CC = cv2.merge((Cr,Cb))
#xAB = cv2.merge((np.uint8(x),Cr,Cb))
xAB = cv2.merge((np.uint8(x),CC))
orgxAB_2 = cv2.cvtColor(xAB, cv2.COLOR_YCR_CB2BGR)

plt.title('xAB_2')
plt.imshow(orgxAB_2)
plt.savefig('./YCC/orgxAB_2.png')
plt.pause(3)
plt.clf()
"""

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

#from net_encoder_decoder2D import Encoder, Decoder
#from net_encoder_decoder1D2DResize import Encoder, Decoder
from net_encoder_decoder_vgg16 import Encoder, Decoder

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

class rgb2YCrCb(object):
    def __init__(self):
        self.ts = torchvision.transforms.ToPILImage()
        self.ts2 = transform=transforms.ToTensor()
        self.mean, self.std =[0.5,0.5,0.5], [0.25,0.25,0.25]
        self.ts3 =  torchvision.transforms.Compose([
            #torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.ToPILImage()
            #transforms.ToTensor()
        ])
        pass
    
    def __call__(self, tensor):
        tensor = self.ts3(tensor)
        orgYCrCb = cv2.cvtColor(np.float32(tensor), cv2.COLOR_BGR2YCR_CB)
        Y, Cr,Cb = cv2.split(orgYCrCb)
        CC = cv2.merge((Cr,Cb))
        CC = np.array(CC)
        #print(CC.shape)
        return CC
    
    def __repr__(self):
        return self.__class__.__name__
    
class rgb2YCrCb_(object):
    def __init__(self):
        self.ts = torchvision.transforms.ToPILImage()
        self.ts2 = transform=transforms.ToTensor()
        self.mean, self.std =[0.5,0.5,0.5], [0.25,0.25,0.25]
        self.ts3 =  torchvision.transforms.Compose([
            #torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.ToPILImage()
            #transforms.ToTensor()
        ])        
        pass
    
    def __call__(self, tensor):
        tensor = self.ts3(tensor)
        orgYCrCb = cv2.cvtColor(np.float32(tensor), cv2.COLOR_BGR2YCR_CB)
        Y, Cr,Cb = cv2.split(orgYCrCb)
        CC = cv2.merge((Cr,Cb))
        Y = np.array(Y).reshape(32*2,32*2)
        #print(Y.shape)
        return Y

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_num,train_=True, transform1 = None, transform2 = None,train = True):
                
        self.transform1 = transform1
        self.transform2 = transform2
        self.ts = torchvision.transforms.ToPILImage()
        self.ts2 = transforms.ToTensor()
        self.mean, self.std =[0.5,0.5,0.5], [0.25,0.25,0.25]
        self.ts3 =  torchvision.transforms.Compose([
            torchvision.transforms.Normalize(self.mean, self.std),
            transforms.ToTensor()
        ])
        self.train = train_
        
        self.data_dir = './'
        self.data_num = data_num
        self.data = []
        self.label = []

        # download
        CIFAR10(self.data_dir, train=True, download=True)
        #CIFAR10(self.data_dir, train=False, download=True)
        self.data =CIFAR10(self.data_dir, train=self.train, transform=self.ts2)

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
            
        #print( out_data1.shape, out_data2.shape)
        #ts(np.array(Y).reshape(64,64))
        return out_data, np.array(out_data1).reshape(1,64,64), np.array(out_data2.reshape(2,64,64)), out_label
    
class LitAutoEncoder(pl.LightningModule):

    def __init__(self, data_dir='./'):
        super().__init__()
        self.data_dir = data_dir
        self.data_num =50000 #50000
        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.dims = (32*2, 32*2)
        self.dims2 = (32*4, 32*4)
        self.mean, self.std =[0.5,0.5,0.5], [0.25,0.25,0.25]
        
        self.trans2 = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.Resize(self.dims)
        ])
        self.trans1 =  torchvision.transforms.Compose([
            torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.Resize(self.dims),
            MyAddGaussianNoise(0., 0.1),
            torchvision.transforms.Grayscale()
        ])
        
        self.trans2 =  torchvision.transforms.Compose([
            #torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.Resize(self.dims),
            rgb2YCrCb(),  #CC
            transforms.ToTensor()
        ])
        self.trans1 =  torchvision.transforms.Compose([
            #torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.Resize(self.dims),
            rgb2YCrCb_(),  #Y
            transforms.ToTensor(),
            #torchvision.transforms.Grayscale()
        ])
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x, _ = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        _,x,x_ , y = batch
        #print(x.shape, x_.shape)
        z, _ = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x_)
        self.log('train_loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        _,x, x_, y = batch
        z, _ = self.encoder(x)
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
    
def imshow(img,file='', text_=''):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy() #img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(x = 3, y = 2, s = text_, c = "red")
    plt.pause(3)
    if file != '':
        plt.savefig(file+'.png')
    plt.close()
    
ts1 = transforms.Resize((64,64))    
ts = transforms.ToPILImage()
ts2 = transforms.ToTensor()
trans2 =  transforms.Compose([
    transforms.Resize((64,64)),
    rgb2YCrCb(), #CrCb
])
trans1 =  transforms.Compose([
    transforms.Resize((64,64)),
    rgb2YCrCb_(),  #Y
])
mean, std =[0.5,0.5,0.5], [0.25,0.25,0.25]
ts3 =  transforms.Compose([
    transforms.Normalize(mean, std),
    #transforms.ToTensor()
])
    
autoencoder = LitAutoEncoder()
PATH = 'example_cifar4L100.ckpt'
pretrained_model = autoencoder.load_from_checkpoint(PATH)
pretrained_model.freeze()
pretrained_model.eval()

data_num = 50000
cifar10_full =ImageDataset(data_num, train=True, transform1=trans1, transform2=trans2)
n_train = int(len(cifar10_full)*0.1)
n_val = int(len(cifar10_full)*0.1)
n_test = len(cifar10_full)-n_train -n_val
cifar10_train, cifar10_val, cifar10_test = torch.utils.data.random_split(cifar10_full, [n_train, n_val, n_test])

trainloader = DataLoader(cifar10_train, shuffle=True, drop_last = True, batch_size=32, num_workers=0)
valloader = DataLoader(cifar10_val, shuffle=False, batch_size=32, num_workers=0)
testloader = DataLoader(cifar10_test, shuffle=False, batch_size=32, num_workers=0)

latent_dim,ver = "simpleGray2Clolarization", "color_plate1"  #####save condition
dataiter = iter(testloader)
images0,images, images1, labels = dataiter.next()



encode_img,_ = pretrained_model.encoder(images[0:32].to('cpu').reshape(32,1,32*2,32*2)) #3
decode_img = pretrained_model.decoder(encode_img)
decode_img_cpu = decode_img.cpu()
images2 = []

for i in range(32): #32
    print(i, images[i].shape, decode_img_cpu[i].shape)
    YCC_ = cv2.merge((np.array(images[i].reshape(64,64)),np.array(decode_img_cpu[i].reshape(64,64,2))))
    images2_ = cv2.cvtColor(YCC_, cv2.COLOR_YCR_CB2BGR)
    images2.append(ts3(ts2(images2_/255.)))
imshow(torchvision.utils.make_grid(images2), 'autoencode_preds_cifar10_{}_{}'.format(latent_dim,ver),text_ =' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))    
"""
for i in range(32): 
    plt.title('image2_preds')
    img = images2[i] / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy() #img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('./YCC/5piece/image_preds{}.png'.format(i))
    plt.pause(3)
    plt.clf()

    plt.title('image_gray')
    img = images[i] / 2 + 0.5     # unnormalize
    #npimg = img.detach().numpy() #img.numpy()
    plt.imshow(img.reshape(64,64), cmap = 'gray')
    plt.savefig('./YCC/5piece/image_gray{}.png'.format(i))
    plt.pause(3)
    plt.clf()
    
    plt.title('image_original_norm')
    img = ts3(ts2(ts(images0[i]))) / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy() #img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('./YCC/5piece/image_original{}_.png'.format(i))
    plt.pause(3)
    plt.clf()
    
    plt.title('image_original_')
    img = images0[i] / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy() #img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('./YCC/5piece/image_original{}.png'.format(i))
    plt.pause(3)
    plt.clf()   
    
    plt.title('image_originalx2_')
    img = ts3(ts1(ts2(ts(images0[i])))) / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy() #img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('./YCC/5piece/image_originalx2_{}.png'.format(i))
    plt.pause(3)
    plt.clf()       
"""
path_= 'YCC'
YCC = cv2.imread('YCC.jpg')
#YCC = cv2.imread('color_plate1.jpg')

image1=cv2.cvtColor(YCC, cv2.COLOR_BGR2RGB)
plt.title('image1')
plt.imshow(image1)
plt.pause(3)
plt.savefig('./YCC/'+path_+'.png')
plt.clf()

orgYCrCb = cv2.cvtColor(YCC, cv2.COLOR_BGR2YCR_CB)
Y, Cr,Cb = cv2.split(orgYCrCb)

plt.title('images[0]_')
img = images[0] / 2 + 0.5     # unnormalize
#npimg = img.detach().numpy() #img.numpy()
print('images[0]',img)
plt.imshow(img.reshape(64,64), cmap = 'gray')
plt.savefig('./YCC/'+path_+'_gray.png')
plt.pause(3)
plt.clf()
#images[0] = ts2(Y_)

Y_=ts(Y).resize((32*2,32*2))
plt.title('Y_')
img = ts2(Y_)*255. / 2 + 0.5     # unnormalize
print('Y_',img)
#npimg = img.detach().numpy() #img.numpy()
plt.imshow(img.reshape(64,64), cmap = 'gray')
plt.savefig('./YCC/'+path_+'original_gray.png')
plt.pause(3)
plt.clf()
    
Y_=ts(Y).resize((32*2,32*2))
plt.title('Y_')
plt.imshow(Y_, cmap = 'gray')
plt.savefig('./YCC/'+path_+'_mt_original.png')
plt.pause(3)
plt.clf()
#Y_ = np.array(Y_).reshape(1,64,64)

Y_2 =  ts2(Y_)*255

encode_img,_ = pretrained_model.encoder(Y_2.to('cpu').reshape(1,1,32*2,32*2)) #3
decode_img = pretrained_model.decoder(encode_img)
decode_img_cpu = decode_img.cpu()
print(Y_2.shape, decode_img_cpu.shape)
YCC_ = cv2.merge((np.array(Y_2.reshape(64,64)),np.array(decode_img_cpu.reshape(64,64,2))))
images2_ = cv2.cvtColor(YCC_, cv2.COLOR_YCR_CB2BGR)
images2 = ts3(ts2(images2_/255.))
print(images2)
plt.title('preds')
img = images2  / 5 + 0.5     # unnormalize
npimg = img.detach().numpy() #img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.savefig('./YCC/'+path_+'_preds_5.png')
plt.pause(3)
plt.clf()
