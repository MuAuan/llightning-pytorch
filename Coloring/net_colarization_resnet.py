import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class ColorizationNet(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet18(num_classes=365) 
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels = 128, out_channels = 64,
                                  kernel_size = 2, stride = 2, padding = 0),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels = 64, out_channels = 2,
                                  kernel_size = 2, stride = 2),
      #nn.BatchNorm2d(16),
      #nn.ReLU(),
      #nn.ConvTranspose2d(in_channels = 16, out_channels = 2,
      #                            kernel_size = 2, stride = 2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    #output = self.upsample(midlevel_features)
    output = self.decoder(midlevel_features)
    return output