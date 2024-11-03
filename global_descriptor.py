import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

def IdentityTensor(n: int, C: int, H: int, W: int):
    identity_matrix = torch.eye(H, W)  # Shape: (H, W)
    # Repeat this identity matrix across all channels and batch size
    identity_tensor = identity_matrix.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    identity_tensor = identity_tensor.repeat(n, C, 1, 1)  # Shape: (n, C, H, W)
    return identity_tensor

class CustomReseNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(SPADEResBlock, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(SPADEResBlock, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(SPADEResBlock, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(SPADEResBlock, 512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, mid_channels, num_blocks, stride):
        layers = []
        # Determine if downsampling is needed
        if stride != 1 or self.in_channels != mid_channels * block.expansion:
            layers.append(block(self.in_channels, mid_channels, downsample = True))

        self.in_channels = mid_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, mid_channels, downsample = False))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        IdentityTensor()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)
        
        return x
    