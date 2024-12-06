import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *

class Generator(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 16 * 32, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.InstanceNorm2d(num_features = 256, affine = True)
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 1)

        self.ResBlock2D_1 = ResBlock2D(in_C = 256, out_C = 256)
        self.ResBlock2D_2 = ResBlock2D(in_C = 256, out_C = 256)
        self.ResBlock2D_3 = ResBlock2D(in_C = 256, out_C = 256)
        self.ResBlock2D_4 = ResBlock2D(in_C = 256, out_C = 256)
        self.ResBlock2D_5 = ResBlock2D(in_C = 256, out_C = 256)
        self.ResBlock2D_6 = ResBlock2D(in_C = 256, out_C = 256)

        self.upblock2D_1 = UpBlock2D(in_C = 256, out_C = 128)
        self.upblock2D_2 = UpBlock2D(in_C = 128, out_C = 64)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 7, stride = 1, padding = 3)

    def forward(self, fs, composited_flow_field, occulsion):
        '''
        fs shape : [N, C, D, H, W]
        composited_flow_field shape: [N, D, H, W, 3]
        occulsion shape: [N, 1, H, W]
        '''
        N, _, D, H, W = fs.shape
        input = F.grid_sample(fs, composited_flow_field, align_corners = True).view(N, -1, H, W)
        output = F.leaky_relu(self.bn1(self.conv1(input)), negative_slope = 0.2, inplace = True)
        output = self.conv2(output)
        # output shape: [N, 256, H, W]
        out_mul = occulsion * output
        out = self.ResBlock2D_1(out_mul)
        out = self.ResBlock2D_2(out)
        out = self.ResBlock2D_3(out)
        out = self.ResBlock2D_4(out)
        out = self.ResBlock2D_5(out)
        out = self.ResBlock2D_6(out)
        out = self.upblock2D_1(out)
        out = self.upblock2D_2(out)
        # out shape: [N, 64, 4 * H, 4 * W]
        generated_image = self.conv3(out)
        # generated_image shape: [N, 3, 4 * H, 4 * W] => reshape back to original image size (B, 3, 256, 256)
        return generated_image
     
if __name__ == '__main__':
    fs = torch.randn(1, 32, 16, 64, 64).cuda()
    composited_flow_field = torch.randn(1, 16, 64, 64, 3).cuda()
    occulsion = torch.randn(1, 1, 64, 64).cuda()
    model = Generator().cuda()
    output = model(fs, composited_flow_field, occulsion)
    print(output.shape)