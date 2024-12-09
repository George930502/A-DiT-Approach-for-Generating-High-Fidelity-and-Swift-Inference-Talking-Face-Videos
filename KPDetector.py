import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *

class CanonicalKeypointDetector(nn.Module): 
    def __init__(self, in_C: int = 3, num_kp: int = 20, temperature: int = 0.1):
        '''
        DownBlock2D_1 input columns depend on input source columns (default: 3)
        '''
        super().__init__()
        self.downblock2D_1 = DownBlock2D(in_C = in_C, out_C = 64)
        self.downblock2D_2 = DownBlock2D(in_C = 64, out_C = 128)
        self.downblock2D_3 = DownBlock2D(in_C = 128, out_C = 256)
        self.downblock2D_4 = DownBlock2D(in_C = 256, out_C = 512)
        self.downblock2D_5 = DownBlock2D(in_C = 512, out_C = 1024)

        self.conv = nn.Conv2d(in_channels = 1024, out_channels = 16384, kernel_size = 1)
        self.upblock3D_1 = UpBlock3D(in_C = 1024, out_C = 512)
        self.upblock3D_2 = UpBlock3D(in_C = 512, out_C = 256)
        self.upblock3D_3 = UpBlock3D(in_C = 256, out_C = 128)
        self.upblock3D_4 = UpBlock3D(in_C = 128, out_C = 64)
        self.upblock3D_5 = UpBlock3D(in_C = 64, out_C = 32) 

        self.temperature = temperature
        self.num_kp = num_kp
        self.conv_keypoints = nn.Conv3d(in_channels = 32, out_channels = self.num_kp, kernel_size = 7, stride = 1, padding = 3) 

    def forward(self, x):
        """
        return keypoints with shape: [B, num_kp, 3]
        """
        # U-NET model part
        out = self.downblock2D_1(x)
        out = self.downblock2D_2(out)
        out = self.downblock2D_3(out)
        out = self.downblock2D_4(out)
        out = self.downblock2D_5(out)

        out = self.conv(out)
        # Reshape C16384 →C1024xD16
        out = out.view(out.size(0), 1024, 16, *out.shape[2:])

        out = self.upblock3D_1(out)
        out = self.upblock3D_2(out)
        out = self.upblock3D_3(out)
        out = self.upblock3D_4(out)
        out = self.upblock3D_5(out)  # (B, 32, D, H, W)
        
        # extract keypoints with shape R(3x1)
        out_kp = self.conv_keypoints(out) # (B, 20, D, H, W)
        heatmap = out2heatmap(out_kp, temperature = self.temperature)
        keypoints = heatmap2kp(heatmap) 
        return keypoints
     
if __name__ == '__main__':
    x = torch.randn(5, 3, 256, 256).cuda()
    model = CanonicalKeypointDetector().cuda()
    print(model(x).shape)

