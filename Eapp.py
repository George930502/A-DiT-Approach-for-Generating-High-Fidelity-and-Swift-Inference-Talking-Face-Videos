import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

'''
class AppeanceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 1, padding = 3)
        self.resblock2d_128 = ResBlock2DCustom(64, 128)
        self.resblock2d_256 = ResBlock2DCustom(128, 256)
        self.resblock2d_512 = ResBlock2DCustom(256, 512)

        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv1 = nn.Conv2d(in_channels = 512, out_channels = 1536, kernel_size = 1, stride = 1, padding = 0)
        self.resblock3d_96 = nn.ModuleList([ResBlock3DCustom(96, 96) for _ in range(3)])
        
    def forward(self, x):
        out = self.conv(x)              # [1, 64, 256, 256]
        out = self.resblock2d_128(out)  # [1, 128, 256, 256]
        out = self.avgpool(out)         # [1, 128, 128, 128]
        out = self.resblock2d_256(out)  # [1, 256, 128, 128]
        out = self.avgpool(out)         # [1, 256, 64, 64]
        out = self.resblock2d_512(out)  # [1, 512, 64, 64]
        out = self.avgpool(out)         # [1, 512, 32, 32]

        out = F.group_norm(out, num_groups = 32)
        out = F.relu(out)
        out = self.conv1(out)

        # Reshape C1536 → C96xD16
        fs = out.view(out.size(0), 96, 16, *out.shape[2:])  # [1, 96, 16, 32, 32]
        for resblock3d_96 in self.resblock3d_96:
            fs = resblock3d_96(fs)

        return fs  # [1, 96, 16, 32, 32]
'''    

class AppearanceFeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 1, padding = 3)
        self.norm1 = nn.BatchNorm2d(num_features = 64)
        self.downblock2D_1 = DownBlock2D(in_C = 64, out_C = 128)
        self.downblock2D_2 = DownBlock2D(in_C = 128, out_C = 256)
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 1)
        self.resblock3d_32 = nn.ModuleList([ResBlock3D(32, 32) for _ in range(6)])

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.norm1(out))
        out = self.downblock2D_1(out)
        out = self.downblock2D_2(out)
        out = self.conv2(out)

        # Reshape C512 →D16xC32
        # (N, C, D, H, W)
        out_reshape = out.view(out.size(0), 32, 16, *out.shape[2:])  # (B, 32, 16, 64, 64)
        for resblock3d_32 in self.resblock3d_32:
            fs = resblock3d_32(out_reshape)
        
        return fs  # (B, 32, 16, 64, 64))

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = AppearanceFeatureExtraction()
    print(model(x)[0])
