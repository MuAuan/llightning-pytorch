import torch.nn as nn
import torch.nn.functional as F

class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.Bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 1024, 5)
        self.Bn2 = nn.BatchNorm2d(1024)
        #self.conv3 = nn.Conv2d(512, 1924, 2)
        #self.Bn3 = nn.BatchNorm2d(1924)
        self.fc1 = nn.Linear(1024*5*5, 32)
        #self.fc2 = nn.Linear(1600, 400)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.Bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.Bn2(x)
        #x = self.pool(F.relu(self.conv3(x)))
        #x = self.Bn3(x)
        x = x.view(-1, 1024*5*5)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x