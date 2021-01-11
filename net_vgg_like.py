import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3,stride=1,padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2_drop = nn.Dropout2d(p=0.25)
        self.conv2_1 = nn.Conv2d(64, 128, 3,stride=1,padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3,stride=1,padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3,stride=1,padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3,stride=1,padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3,stride=1,padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, 3,stride=1,padding=1)
        self.fc1 = nn.Linear(256*4*4 , 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        #x = self.conv2_drop(self.pool(F.relu(self.conv1_2(x))))
        x = self.pool(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        #x = self.conv2_drop(self.pool(F.relu(self.conv2_2(x))))
        x = self.pool(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        #x = self.conv2_drop(self.pool(F.relu(self.conv3_4(x))))
        x = self.pool(F.relu(self.conv3_4(x)))
        x = x.view(-1, 256*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
