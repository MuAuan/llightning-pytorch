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


ts = torchvision.transforms.ToPILImage()

image0 = cv2.imread('mayuyu.jpg')
"""
cv2.imshow('image0',image0)
plt.imshow(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
plt.pause(1)
plt.clf()
plt.close()
"""
image1=cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
plt.title('image1')
plt.imshow(image1)
plt.pause(1)
plt.clf()
"""
orgYCrCb = cv2.cvtColor(image1, cv2.COLOR_BGR2YCR_CB)
plt.title('orgYCrCb')
plt.imshow(orgYCrCb)
plt.pause(1)
plt.clf()

orgYCrCb_ = cv2.cvtColor(orgYCrCb, cv2.COLOR_YCR_CB2BGR)
plt.title('orgYCrCb_')
plt.imshow(orgYCrCb_)
plt.pause(3)
plt.clf()

Y, Cr,Cb = cv2.split(orgYCrCb)
plt.title('Y')
plt.imshow(Y, cmap = 'gray')
plt.pause(1)
plt.clf()

plt.title('Cr')
plt.imshow(Cr, cmap = 'gray')
plt.pause(1)
plt.clf()

plt.title('Cb')
plt.imshow(Cb, cmap = 'gray')
plt.pause(1)
plt.clf()

Cr_=ts(Cr).convert("RGB")
Cb_ = ts(Cb).convert("RGB")
#CC = cv2.merge((Y,Cr_,Cb_))

plt.title('CrCb_2')
plt.imshow(Cr_)
plt.pause(3)
plt.clf()

YCC = cv2.merge((Y,Cr,Cb))
orgYCrCb_2 = cv2.cvtColor(YCC, cv2.COLOR_YCR_CB2BGR)

plt.title('orgYCrCb_2')
plt.imshow(orgYCrCb_2)
plt.pause(3)
plt.clf()

orgLAB = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
plt.title('orgLAB')
plt.imshow(orgLAB)
plt.pause(1)
plt.clf()

L, A,B = cv2.split(orgLAB)
plt.title('L')
plt.imshow(L) #, cmap = 'gray')
plt.pause(1)
plt.clf()
print(L.shape,A.shape,B.shape)

plt.title('A')
plt.imshow(A)
plt.pause(1)
plt.clf()

plt.title('B')
plt.imshow(B)
plt.pause(1)
plt.clf()

LAB = cv2.merge((L,A,B))
orgLAB_2 = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)

plt.title('orgLAB_2')
plt.imshow(orgLAB_2)
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
"""

trans =  torchvision.transforms.Compose([
            #torchvision.transforms.Normalize(self.mean, self.std),
            #torchvision.transforms.Resize(self.dims),
            #MyAddGaussianNoise(0., 0.1),
            torchvision.transforms.Grayscale()
        ])
x = trans(ts(image1))
plt.title('grayscale')
plt.imshow(x, cmap = 'gray')
plt.pause(2)
plt.clf()

orgYCrCb = cv2.cvtColor(image1, cv2.COLOR_BGR2YCR_CB)
Y, Cr,Cb = cv2.split(orgYCrCb)
plt.title('Y')
plt.imshow(Y, cmap = 'gray')
plt.pause(2)
plt.clf()

xCC = cv2.merge((np.uint8(x),Cr,Cb))
orgxCrCb_2 = cv2.cvtColor(xCC, cv2.COLOR_YCR_CB2BGR)

plt.title('orgxCrCb_2')
plt.imshow(orgxCrCb_2)
plt.pause(2)
plt.clf()

orgLAB = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
L, A,B = cv2.split(orgLAB)

plt.title('L')
plt.imshow(L, cmap = 'gray')
plt.pause(2)
plt.clf()

CC = cv2.merge((Cr,Cb))
#xAB = cv2.merge((np.uint8(x),Cr,Cb))
xAB = cv2.merge((np.uint8(x),CC))
orgxAB_2 = cv2.cvtColor(xAB, cv2.COLOR_YCR_CB2BGR)

plt.title('xAB_2')
plt.imshow(orgxAB_2)
plt.pause(2)
plt.clf()

