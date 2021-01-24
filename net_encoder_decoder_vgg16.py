import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self): 
        super(Encoder, self).__init__()
        num_classes = 10
        self.block1_output = nn.Sequential (
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2_output = nn.Sequential (
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3_output = nn.Sequential (
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block4_output = nn.Sequential (
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block5_output = nn.Sequential (
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 512),  #512 * 7 * 7, 4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, 32 ),  #4096, 4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(32, num_classes)  #4096
        )
        
    def forward(self, x):
        x1 = self.block1_output(x)
        x2 = self.block2_output(x1)
        x3 = self.block3_output(x2)
        x4 = self.block4_output(x3)
        x = self.block5_output(x4)
        x0 = x.view(x.size(0), -1)
        y = self.classifier(x0)
        return x3, y
"""
class Latent(nn.Module):
    def __init__(self):
        super(Latent, self).__init__()
        self.latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 1000), 
            nn.ReLU(), 
            nn.Linear(1000, 10)
        )
    def forward(self, x):
        x = self.latent(x)
        #x = x.reshape(32,256,4,4)
        return x
"""    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2)),
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128,
                                          kernel_size = 2, stride = 2, padding = 0),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64,
                                          kernel_size = 2, stride = 2),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 16,
                                          kernel_size = 2, stride = 2),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 3,
                                          kernel_size = 2, stride = 2)
        )

    def forward(self, x):
        #x = x.reshape(32,256,4,4)
        x = self.decoder(x)
        return x