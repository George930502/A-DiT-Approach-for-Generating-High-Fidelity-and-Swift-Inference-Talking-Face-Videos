import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class AppearanceFeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 1, padding = 3)
        self.norm1 = nn.InstanceNorm2d(num_features = 64, affine = True)
        self.downblock2D_1 = DownBlock2D(in_C = 64, out_C = 128)
        self.downblock2D_2 = DownBlock2D(in_C = 128, out_C = 256)
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 1)
        self.resblock3d_32 = nn.ModuleList([ResBlock3D(32, 32) for _ in range(6)])

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.norm1(out), inplace = True)
        out = self.downblock2D_1(out)
        out = self.downblock2D_2(out)
        out = self.conv2(out)
        # Reshape C512 →D16xC32
        # (N, C, D, H, W)
        out_reshape = out.view(out.size(0), 32, 16, *out.shape[2:])  # (B, 32, 16, 64, 64)
        for resblock3d_32 in self.resblock3d_32:
            fs = resblock3d_32(out_reshape)
        # fs shape: (B, 32, 16, 64, 64)
        return fs  

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = AppearanceFeatureExtraction()
    print(model(x)[0])
