import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10 

from PIL import Image

def to_RGB(image:Image)->Image:
  if image.mode == 'RGB':return image
  image.load() # required for png.split()
  background = Image.new("RGB", image.size, (255, 255, 255))
  background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
  
  file_name = 'tmp.jpg'
  background.save(file_name, 'JPEG', quality=80)
  return cv2.open(file_name)
  #return Image.open(file_name)

class rgb2YCrCb(object):
    def __init__(self):
        self.ts = torchvision.transforms.ToPILImage()
        self.ts2 = transform=transforms.ToTensor()
        pass
    
    def __call__(self, tensor):
        tensor = self.ts(tensor)
        orgYCrCb = cv2.cvtColor(np.float32(tensor), cv2.COLOR_BGR2YCR_CB)
        Y, Cr,Cb = cv2.split(orgYCrCb)
        CC = cv2.merge((Cr,Cb))
        return CC
    
    def __repr__(self):
        return self.__class__.__name__
    
class rgb2YCrCb_(object):
    def __init__(self):
        self.ts = torchvision.transforms.ToPILImage()
        self.ts2 = transform=transforms.ToTensor()
        pass
    
    def __call__(self, tensor):
        tensor = self.ts(tensor)
        orgYCrCb = cv2.cvtColor(np.float32(tensor), cv2.COLOR_BGR2YCR_CB)
        Y, Cr,Cb = cv2.split(orgYCrCb)
        CC = cv2.merge((Cr,Cb))
        return Y
    
    def __repr__(self):
        return self.__class__.__name__      

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
        return out_data, out_data1, out_data2, out_label

    
ts = torchvision.transforms.ToPILImage()
ts2 = transform=transforms.ToTensor()
dims = (32*4, 32*4) 
mean, std =[0.5,0.5,0.5], [0.25,0.25,0.25]
trans2 = torchvision.transforms.Compose([
    #torchvision.transforms.Normalize(mean, std),
    #torchvision.transforms.Resize(dims)
])
trans1 =  torchvision.transforms.Compose([
    #torchvision.transforms.Normalize(mean, std),
    #MyAddGaussianNoise(0., 0.5),
    #torchvision.transforms.Resize(dims),
    torchvision.transforms.Grayscale()
])

trans3 =  torchvision.transforms.Compose([
    #torchvision.transforms.Normalize(mean, std),
    #MyAddGaussianNoise(0., 0.1),
    #torchvision.transforms.Resize(dims),
    rgb2YCrCb(),
])
trans4 =  torchvision.transforms.Compose([
    #torchvision.transforms.Normalize(mean, std),
    #MyAddGaussianNoise(0., 0.1),
    #torchvision.transforms.Resize(dims),
    rgb2YCrCb_(),
])

dataset = ImageDataset(8, transform1=trans4, transform2=trans3)
testloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)
for out_data, out_data1, out_data2,out_label in testloader:
    for i in range(len(out_label)):
        image =  out_data[i]
        Y = out_data1[i]
        Y = np.array(Y).reshape(32,32)
        CC_2 = out_data2[i]
        CC_2 = np.array(CC_2)
        
        
        #image_ = to_RGB(ts(image)) #jpeg
        #orgYCrCb, Y, CC = trans4((ts2(image_)))
        #print(orgYCrCb.shape,Y.shape,CC.shape)
        print(out_label[i])
        plt.imshow(Y, cmap = "gray")
        plt.title('Y')
        #print(type(orgYCrCb))
        plt.pause(1)
        
        X = np.zeros((32,32))
        #X = np.array(ts2(X).reshape(32,32))
        #print(X.shape, Y.shape, CC_2.shape)
        X = X.astype(np.uint8)
        CC_ = CC_2.astype(np.uint8)
        XCC = cv2.merge((X,CC_))
        XCC_ = cv2.cvtColor(XCC, cv2.COLOR_YCR_CB2BGR)
        plt.imshow(XCC_/255.)
        plt.title('XCC')
        plt.pause(1)
        
        print(Y.shape, CC_2.shape)
        YCC = cv2.merge((Y,CC_2))
        orgYCrCb_2 = cv2.cvtColor(YCC, cv2.COLOR_YCR_CB2BGR)
        plt.imshow(orgYCrCb_2/255.)
        plt.title('Y+Cr+Cb')
        plt.pause(1)
        
        YCC_ = cv2.merge((Y,CC_2))
        orgYCrCb_2 = cv2.cvtColor(YCC_, cv2.COLOR_YCR_CB2BGR)
        plt.imshow(orgYCrCb_2/255.)
        plt.title('Y+Cr+Cb_')
        plt.pause(1)
        
        plt.close()