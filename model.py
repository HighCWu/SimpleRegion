import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionNet(nn.Module):
    def __init__(self, inout_channels=3, n_class=32, channels=128, conv_num=2):
        super(RegionNet, self).__init__()
        
        self.conv1 = nn.Conv2d(inout_channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(conv_num-1):
            self.conv2.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(channels))
        self.conv3 = nn.Conv2d(channels, n_class, kernel_size=1, stride=1, padding=0)
        self.n_class = n_class
        
    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        y = self.bn1(y)
        for i in range(len(self.conv2)):
            y = self.conv2[i](y)
            y = F.relu(y)
            y = self.bn2[i](y)
        y = self.conv3(y)
        
        y = F.softmax(y, 1)
        y_hard = region = F.one_hot(y.argmax(1, keepdim=True), num_classes=self.n_class).transpose(1,-1)[...,0].float()

        # MapFn
        x = x[:,None,...]

        y = y[:,:,None,...]
        y = y * ((x * y).sum(dim=[-1,-2], keepdim=True) / (y.sum(dim=[-1,-2], keepdim=True) + 1e-9))
        y_soft = y.sum(dim=1) # b x c x h x w
        
        y = y_hard[:,:,None,...]
        y = y * ((x * y).sum(dim=[-1,-2], keepdim=True) / (y.sum(dim=[-1,-2], keepdim=True) + 1e-9))
        y_hard = y.sum(dim=1) # b x c x h x w

        return y_soft, y_hard, region
