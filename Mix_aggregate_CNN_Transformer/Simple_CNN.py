import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses=250
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
       
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 224 * 224, 512)
        self.fc2 = nn.Linear(512, nclasses)
       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, 128 * 224 * 224)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
