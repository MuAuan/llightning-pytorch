import torchvision.models as models
from torchsummary import summary
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

class customize_model(nn.Module):
  def __init__(self, input_size=128, sel_model='resnet18'):
    super(customize_model, self).__init__()
    
    ## Select model
    if sel_model == 'resnext50':
        self.model_ = models.resnext50_32x4d(pretrained=True)
    elif sel_model == 'vgg16':
        model_0 = models.vgg16_bn(pretrained=True) 
        self.model_ = nn.Sequential(*list(model_0.children())[0]) #VGG16_bn 
    elif sel_model == 'wide50':
        self.model_ = models.wide_resnet50_2(pretrained=True) 
    elif sel_model == 'mobilev2':
        self.model_ = models.mobilenet_v2(pretrained=True)
    elif sel_model == 'densenet121':
        self.model_ = models.densenet121(pretrained=True)
    else:
        self.model_ = models.resnet18(pretrained=True)
    
    # Change first conv layer to accept single-channel (grayscale) input
    #resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    
    # Extract midlevel features from ResNet-gray
    #self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:2]) #wide_resnet50_2
    #self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0]) #VGG16_bn 
    #self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:8]) #resnet18
    
    for i, param in enumerate(self.model_.parameters()):
        param.requires_grad = True #False
        #print(i, param.requires_grad)
    """
    for i, param in enumerate(self.model_.parameters()):
        if i >= 20:
            param.requires_grad = False #True #False
        print(i, param.requires_grad)
    
    j =0
    for name,  param in self.model_.named_parameters():
        param.requires_grad = True
        if j <= 44: #83: #140:
            param.requires_grad = False
        print(j, name, param.requires_grad)
        j +=1
    """
    if sel_model =='vgg16':
        self.f_resnet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=np.int(512*2*2), out_features=2048, bias=True), #wide_resnet50-2 512*256
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=np.int(512*2*2), out_features=1000, bias=True), #wide_resnet50-2 512*256
            nn.Linear(in_features=np.int(1000), out_features=10, bias=True) #wide_resnet50-2 512*256
            )    
    else:
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

    midlevel_features = self.model_(input)
    output = self.f_resnet(midlevel_features)
    return output

def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #for gpu
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    pl.seed_everything(0)
    """

    #model = models.resnet18(pretrained=False, num_classes=10)
    #model = models.vgg16_bn(pretrained=False, num_classes=10)
    #model = models.wide_resnet50_2(pretrained=False, num_classes=10)
    model = models.mobilenet_v2(pretrained=False, num_classes=10)
    model = model.to(device) #for gpu
    print(model)

    dim = (3,128,128)
    summary(model,dim)
    
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
    dim = (3,128,128)
    sel_model_list = ['resnet18', 'vgg16', 'wide50', 'mobilev2', 'densenet121' ]
    for list_ in sel_model_list:
        customize_ = customize_model(128, sel_model = list_)
        model = customize_.to(device) #for gpu
        print('model_customize {}= '.format(list_),model)
        if list_ == 'densenet121' or list_ == 'vgg16':
            pass
        else:
            summary(model,dim)

    """
    for param in model.parameters():
        print(param)
    """    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))    