import torch
import torch.nn as nn
import os

class Feature_Extractor(nn.Module):
    '''Extractor for features
    '''
    def __init__(self, in_channels):
        super().__init__()
        # conv1
        self.conv1_1 = nn.Conv3d(in_channels, 8, 3, padding=60)
        # torch.nn.Conv3d(in_channels, out_channels, kernel_size,
        # stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv3d(8, 8, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv3d(8, 16, 3, padding=15)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4
    
    def forward(self, x):
        h = x

        h = self.relu1_1(self.conv1_1(h))
        
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        return 

class RegistrationModel(nn.Module):
    ''' Feature Map Registration
    '''
    def __init__(self,in_channels):
        super().__init__()
        pass
    
    def forward(self, moving_img, fix_img):
        pass