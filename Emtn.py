import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
import numpy as np

class HeadPoseEstimator_ExpressionDeformationEstimator(nn.Module): 
    def __init__(self, num_kp: int = 20, n_bins: int = 66):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        self.norm1 = nn.InstanceNorm2d(num_features = 64, affine = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        #self.conv2 = nn.Conv2d(in_channels= 64, out_channels = 256, kernel_size=1)
        #self.norm2 = nn.BatchNorm2d(num_features = 256)
        
        self.ResBottleneckBlock_256_downsample = ResBottleneck(in_C = 64, out_C = 256, stride = 1)  
        # default stride is 1
        self.ResBottleneckBlock_256_1 = ResBottleneck(in_C = 256, out_C = 256)  
        self.ResBottleneckBlock_256_2 = ResBottleneck(in_C = 256, out_C = 256)
        self.ResBottleneckBlock_256_3 = ResBottleneck(in_C = 256, out_C = 256)

        self.ResBottleneckBlock_512_downsample = ResBottleneck(in_C = 256, out_C = 512, stride = 2)

        self.ResBottleneckBlock_512_1 = ResBottleneck(in_C = 512, out_C = 512)
        self.ResBottleneckBlock_512_2 = ResBottleneck(in_C = 512, out_C = 512)
        self.ResBottleneckBlock_512_3 = ResBottleneck(in_C = 512, out_C = 512)

        self.ResBottleneckBlock_1024_downsample = ResBottleneck(in_C =  512, out_C = 1024, stride = 2)

        self.ResBottleneckBlock_1024_1 = ResBottleneck(in_C = 1024, out_C = 1024)
        self.ResBottleneckBlock_1024_2 = ResBottleneck(in_C = 1024, out_C = 1024)
        self.ResBottleneckBlock_1024_3 = ResBottleneck(in_C = 1024, out_C = 1024)
        self.ResBottleneckBlock_1024_4 = ResBottleneck(in_C = 1024, out_C = 1024)
        self.ResBottleneckBlock_1024_5 = ResBottleneck(in_C = 1024, out_C = 1024)

        self.ResBottleneckBlock_2048_downsample = ResBottleneck(in_C = 1024, out_C = 2048, stride = 2)

        self.ResBottleneckBlock_2048_1 = ResBottleneck(in_C = 2048, out_C = 2048)
        self.ResBottleneckBlock_2048_2 = ResBottleneck(in_C = 2048, out_C = 2048)
        self.avgpool = nn.AvgPool2d(kernel_size = 7) 

        self.yaw = nn.Linear(in_features = 2048, out_features = 66)
        self.pitch = nn.Linear(in_features = 2048, out_features = 66)
        self.roll = nn.Linear(in_features = 2048, out_features = 66)
        self.translation = nn.Linear(in_features = 2048, out_features = 3)
        self.deformation = nn.Linear(in_features = 2048, out_features = 60)

        self.num_kp = num_kp
        self.n_bins = n_bins
        self.idx_tensor = torch.FloatTensor(list(range(self.n_bins))).unsqueeze(0).cuda()
    
    def forward(self, x):
        '''
        '''
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace = True)
        out = self.maxpool(out)
        
        out = self.ResBottleneckBlock_256_downsample(out)  # [B, 256, 64, 64]
        out = self.ResBottleneckBlock_256_1(out)
        out = self.ResBottleneckBlock_256_2(out)
        out = self.ResBottleneckBlock_256_3(out)

        out = self.ResBottleneckBlock_512_downsample(out)  # [B, 512, 32, 32]

        out = self.ResBottleneckBlock_512_1(out)
        out = self.ResBottleneckBlock_512_2(out)
        out = self.ResBottleneckBlock_512_3(out)

        out = self.ResBottleneckBlock_1024_downsample(out) # [B, 1024, 16, 16]

        out = self.ResBottleneckBlock_1024_1(out)
        out = self.ResBottleneckBlock_1024_2(out)
        out = self.ResBottleneckBlock_1024_3(out)
        out = self.ResBottleneckBlock_1024_4(out)
        out = self.ResBottleneckBlock_1024_5(out)

        out = self.ResBottleneckBlock_2048_downsample(out) # [B, 2048, 8, 8]
        out = self.ResBottleneckBlock_2048_1(out)
        out = self.ResBottleneckBlock_2048_2(out)

        out = self.avgpool(out)  # [B, 2048, 1, 1]
        out = out.view(out.shape[0], -1)  # [B, 2048]

        yaw = F.softmax(self.yaw(out), dim = 1)   # size: [B, 66]
        yaw = (yaw * self.idx_tensor).sum(dim = 1)  # size: [B,]
        pitch = F.softmax(self.pitch(out), dim = 1)   # size: [B, 66]
        pitch = (pitch * self.idx_tensor).sum(dim = 1) # size: [B,]
        roll = F.softmax(self.roll(out), dim = 1)   # size: [B, 66]
        roll = (roll * self.idx_tensor).sum(dim = 1) # size: [B, ]
        
        yaw = (yaw - self.n_bins // 2) * 3 * np.pi / 180  # turn to radian for calculation
        pitch = (pitch - self.n_bins // 2) * 3 * np.pi / 180
        roll = (roll - self.n_bins // 2) * 3 * np.pi / 180

        translation = self.translation(out)  # size: [B, 3]
        deformation = self.deformation(out)  # size: [B, 60]
        deformation = deformation.view(x.shape[0], -1, 3)  # size: [B, 60, 3]

        return yaw, pitch, roll, translation, deformation
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = HeadPoseEstimator_ExpressionDeformationEstimator().cuda()
    print(model(x.cuda())) 
