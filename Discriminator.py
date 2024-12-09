import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, num_kp: int = 20):
        
        super().__init__()
        self.layers = nn.ModuleList([
            DownBlock2D_discriminator(in_C = 3 + num_kp, out_C = 64, norm = False, pool = True),
            DownBlock2D_discriminator(in_C = 64, out_C = 128, norm = True, pool = True),
            DownBlock2D_discriminator(in_C = 128, out_C = 256, norm = True, pool = True),
            DownBlock2D_discriminator(in_C = 256, out_C = 512, norm = True, pool = False),
            spectral_norm(nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4))
        ])

    def forward(self, x, kp):
        '''
        x shape: [N, 3, H, W]
        kp shape: [N, num_kp, 3]
        '''
        heatmap = kp2gaussian_2d(kp.detach()[:, :, :2], x.shape[2:])  # shape: [N, num_kp, H, W]
        x = torch.cat([x, heatmap], dim = 1)
        res = [x]
        for layer in self.layers:
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1: -1]
        return output, features
        
if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256).cuda()
    kp = torch.randn(1, 20, 3).cuda()
    model = Discriminator().cuda()
    output = model(x, kp)
    print(output[0].shape)
    for feature in output[1]:
        print(feature.shape)