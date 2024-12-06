import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *

class MotionFieldEstimator(nn.Module):
    def __init__(self, C1: int = 32, C2: int = 4, num_kp: int = 20):

        super().__init__()
        self.compress = nn.Conv3d(in_channels = C1, out_channels = C2, kernel_size = 1)

        self.downblock3D_1 = DownBlock3D(in_C = (C2 + 1) * (num_kp + 1), out_C = 64)
        self.downblock3D_2 = DownBlock3D(in_C = 64, out_C = 128)
        self.downblock3D_3 = DownBlock3D(in_C = 128, out_C = 256)
        self.downblock3D_4 = DownBlock3D(in_C = 256, out_C = 512)
        self.downblock3D_5 = DownBlock3D(in_C = 512, out_C = 1024)
        self.upblock3D_1 = UpBlock3D(in_C = 1024, out_C = 512)
        self.upblock3D_2 = UpBlock3D(in_C = 512, out_C = 256)
        self.upblock3D_3 = UpBlock3D(in_C = 256, out_C = 128)
        self.upblock3D_4 = UpBlock3D(in_C = 128, out_C = 64)
        self.upblock3D_5 = UpBlock3D(in_C = 64, out_C = 32) 

        self.mask_conv = nn.Conv3d(in_channels = (C2 + 1) *  (num_kp + 1) + C1, out_channels = 21, kernel_size = 7, stride = 1, padding = 3)
        self.occlusion_conv = nn.Conv2d(in_channels = 16 * ((C2 + 1) *  (num_kp + 1) + C1), out_channels = 1, kernel_size = 7, stride = 1, padding = 3)

    def forward(self, fs, kp_s, kp_d, Rs, Rd):
        '''
        fs shape: [N, C, D, H, W]
        kp_s shape: [N, 20, 3]
        kp_d shape: [N, 20, 3]
        Rs shape: [N, 3, 3]
        Rd shape: [N, 3, 3]
        '''
        N, _, D, H, W = fs.shape
        fs_compressed = self.compress(fs)
        heatmap_representation = create_heatmap_representations(fs_compressed, kp_s, kp_d) # shape: [N, 21, 1, 16, 64, 64]
        sparse_motion = create_sparse_motions(fs_compressed, kp_s, kp_d, Rs, Rd)  # shape: [N, 21, 16, 64, 64, 3]
        deformed_source = create_deformed_source_image(fs_compressed, sparse_motion)  # shape: [N, 21, 4, 16, 64, 64]
        input = torch.cat([heatmap_representation, deformed_source], dim = 2).view(N, -1, D, H, W)  # shape: [N, 21 * 5, 16, 64, 64]
        # input shape: [N, 105, 16, 64, 64]
        # U-net structure
        out = self.downblock3D_1(input)
        out = self.downblock3D_2(out)
        out = self.downblock3D_3(out)
        out = self.downblock3D_4(out)
        out = self.downblock3D_5(out)
        out = self.upblock3D_1(out)
        out = self.upblock3D_2(out)
        out = self.upblock3D_3(out)
        out = self.upblock3D_4(out)
        out = self.upblock3D_5(out)  
        # output shape: (N, 32, 16, 64, 64)
        x = torch.cat([input, out], dim = 1)  # shape: (N, 137, 16, 64, 64)
        mask = self.mask_conv(x)  # shape: (N, 21, 16, 64, 64)
        mask = F.softmax(mask, dim = 1).unsqueeze(-1) # shape: (N, 21, 16, 64, 64, 1)
        composited_flow_field = (sparse_motion * mask).sum(dim = 1) # shape: (N, 16, 64, 64, 3)
        occlusion_mask = self.occlusion_conv(x.view(N, -1, H, W))
        occlusion_mask = torch.sigmoid(occlusion_mask) # shape: (N, 1, 64, 64)
        return composited_flow_field, occlusion_mask

     
if __name__ == '__main__':
    fs = torch.randn(1, 32, 16, 64, 64).cuda()
    kp_s = torch.randn(1, 20, 3).cuda()
    kp_d = torch.randn(1, 20, 3).cuda()
    Rs = torch.randn(1, 3, 3).cuda()
    Rd = torch.randn(1, 3, 3).cuda()
    model = MotionFieldEstimator().cuda()
    output = model(fs, kp_s, kp_d, Rs, Rd)
    print(output[0].shape, output[1].shape)