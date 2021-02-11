import torchvision.models as models
from torchsummary import summary
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

class customize_resnet18(nn.Module):
  def __init__(self, input_size=32):
    super(customize_resnet18, self).__init__()
    
    ## First half: ResNet
    self.resnet = models.resnet18(pretrained=True) #, num_classes=10) #365) 
    #resnet = models.vgg16_bn(pretrained=True) #, num_classes=10) #365) 
    #resnet = models.wide_resnet50_2(pretrained=True) #, num_classes=10) #365) 
    #self.resnet = models.mobilenet_v2(pretrained=True)
    
    # Change first conv layer to accept single-channel (grayscale) input
    #resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    
    # Extract midlevel features from ResNet-gray
    #self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:2]) #wide_resnet50_2
    #self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0]) #VGG16_bn 
    #self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:8]) #resnet18
    
    for param in self.resnet.parameters():
        param.requires_grad = False #True #False
    
    self.f_resnet = nn.Sequential(
        #nn.ConvTranspose2d(512, 128, kernel_size=(2, 2), stride=(2, 2)),
        #nn.BatchNorm2d(128),
        #nn.ReLU(),
        #nn.Flatten(),
        #nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=np.int(1000), out_features=10, bias=True) #wide_resnet50-2 512*256
        #nn.Linear(in_features=np.int(input_size*input_size/2), out_features=10, bias=True) #resnet18, vgg16
        #*list(resnet.children())[9:10]
    )    
    
  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    #midlevel_features = self.midlevel_resnet(input)
    midlevel_features = self.resnet(input)
    output = self.f_resnet(midlevel_features)
    return output    

def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #for gpu
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    pl.seed_everything(0)

    #model = models.resnet18(pretrained=False, num_classes=10)
    #model = models.vgg16_bn(pretrained=False, num_classes=10)
    #model = models.wide_resnet50_2(pretrained=False, num_classes=10)
    model = models.mobilenet_v2(pretrained=False, num_classes=10)
    model = model.to(device) #for gpu
    print(model)

    dim = (3,256,256)
    summary(model,dim)
    

    """
    resnet = models.resnet18(num_classes=10)
    midlevel_resnet = nn.Sequential(*list(resnet.children())[0:8])
    f_resnet = nn.Sequential(*list(resnet.children())[8:10])
    model1 = midlevel_resnet.to(device) #for gpu
    print('model1=',model1)
    model2 = f_resnet.to(device) #for gpu
    print('model2=',model2)

    summary(model1,dim)
    #summary(model2,(512,1,1))
    """

    dim = (3,256,256)
    resnet_customize = customize_resnet18(256)
    model = resnet_customize.to(device) #for gpu
    print('model_customize = ',model)
    summary(model,dim)



    """
    for param in model.parameters():
        print(param)
    """    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))    