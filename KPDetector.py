import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class CanonicalKeypointDetector(nn.Module): 
    def __init__(self, in_C: int = 3):
        '''
        DownBlock2D_1 input columns depend on input source columns (default: 3)
        '''
        super().__init__()
        self.downblock2D_1 = DownBlock2D(in_C = in_C, out_C = 64)
        self.downblock2D_2 = DownBlock2D(in_C = 64, out_C = 128)
        self.downblock2D_3 = DownBlock2D(in_C = 128, out_C = 256)
        self.downblock2D_4 = DownBlock2D(in_C = 256, out_C = 512)
        self.downblock2D_5 = DownBlock2D(in_C = 512, out_C = 1024)

        self.conv1 = nn.Conv2d(in_channels = 1024, out_channels = 16384, kernel_size = 1)
        self.upblock3D_1 = UpBlock3D(in_C = 1024, out_C = 512)
        self.upblock3D_2 = UpBlock3D(in_C = 512, out_C = 256)
        self.upblock3D_3 = UpBlock3D(in_C = 256, out_C = 128)
        self.upblock3D_4 = UpBlock3D(in_C = 128, out_C = 64)
        self.upblock3D_5 = UpBlock3D(in_C = 64, out_C = 32)

        self.conv2 = nn.Conv3d(in_channels = 32, out_channels = 20, kernel_size = 7, stride = 1, padding = 3)
        self.conv3 = nn.Conv3d(in_channels = 32, out_channels = 180, kernel_size = 7, stride = 1, padding = 3)

    def forward(self, x):
        # x: [1, 3, 256, 256]
        out = self.downblock2D_1(x)
        out = self.downblock2D_2(out)
        out = self.downblock2D_3(out)
        out = self.downblock2D_4(out)
        out = self.downblock2D_5(out)

        out = self.conv1(out)
        # Reshape C16384 →C1024xD16
        out_reshape = out.view(out.size(0), 1024, 16, *out.shape[2:])  # (B, 1024, 16, 8, 8)

        detector_out = self.upblock3D_1(out_reshape)
        detector_out = self.upblock3D_2(detector_out)
        detector_out = self.upblock3D_3(detector_out)
        detector_out = self.upblock3D_4(detector_out)
        detector_out = self.upblock3D_5(detector_out)

        keypoints = self.conv2(detector_out) # (B, 20, 16, 256, 256)
        jacobians = self.conv3(detector_out) # (B, 180, 16, 256, 256)

        return keypoints, jacobians
       
if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = CanonicalKeypointDetector()
    print(model(x)[1].shape)    
        

