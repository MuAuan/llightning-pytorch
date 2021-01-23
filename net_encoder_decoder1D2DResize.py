import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 64,
                               kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 256, out_channels = 64,
                                          kernel_size = 2, stride = 2, padding = 0),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 16,
                                          kernel_size = 2, stride = 2),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 3,
                                          kernel_size = 2, stride = 2)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x