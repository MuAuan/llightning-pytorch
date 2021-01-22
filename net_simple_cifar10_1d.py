import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*32 * 32, 1280), 
            nn.ReLU(), 
            nn.Linear(1280, 1024)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(1024, 1280), 
            nn.ReLU(), 
            nn.Linear(1280, 3*32 * 32)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x    