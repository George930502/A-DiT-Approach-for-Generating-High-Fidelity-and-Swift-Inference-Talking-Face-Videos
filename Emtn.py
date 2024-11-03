import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class MotionEncoder(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        self.norm1 = nn.BatchNorm2d(num_features = 64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.ResBottleneckBlock_256_1 = ResBottleneck(in_C = 64, out_C = 256)
        self.ResBottleneckBlock_256_2 = ResBottleneck(in_C = 256, out_C = 256)
        self.ResBottleneckBlock_256_3 = ResBottleneck(in_C = 256, out_C = 256)

        self.ResBottleneckBlock_512_downsample = ResBottleneck(in_C = 256, out_C = 512, stride = 2)

        self.ResBottleneckBlock_512_1 = ResBottleneck(in_C = 512, out_C = 512)
        self.ResBottleneckBlock_512_2 = ResBottleneck(in_C = 512, out_C = 512)
        self.ResBottleneckBlock_512_3 = ResBottleneck(in_C = 512, out_C = 512)

        self.ResBottleneckBlock_1024_downsample = ResBottleneck(in_C = 512, out_C = 1024, stride = 2)

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
        self.delta = nn.Linear(in_features = 2048, out_features = 60)
    
    # The full angle range is divided into 66 bins for rotation angles, and the network predicts which bin the target angle is in
    def forward(self, x):
        # x: [1, 3, 256, 256]
        out = self.conv1(x)  # [1, 64, 128, 128]
        out = self.norm1(out)
        out = F.relu(out)
        out = self.maxpool(out)  # [1, 64, 64, 64]

        out = self.ResBottleneckBlock_256_1(out)
        out = self.ResBottleneckBlock_256_2(out)
        out = self.ResBottleneckBlock_256_3(out)

        out = self.ResBottleneckBlock_512_downsample(out)
        out = self.ResBottleneckBlock_512_1(out)
        out = self.ResBottleneckBlock_512_2(out)
        out = self.ResBottleneckBlock_512_3(out)

        out = self.ResBottleneckBlock_1024_downsample(out)
        out = self.ResBottleneckBlock_1024_1(out)
        out = self.ResBottleneckBlock_1024_2(out)
        out = self.ResBottleneckBlock_1024_3(out)
        out = self.ResBottleneckBlock_1024_4(out)
        out = self.ResBottleneckBlock_1024_5(out)

        out = self.ResBottleneckBlock_2048_downsample(out)
        out = self.ResBottleneckBlock_2048_1(out)
        out = self.ResBottleneckBlock_2048_2(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        yaw = F.softmax(self.yaw(out), dim = 1)   # size: [batch, 66]
        pitch = F.softmax(self.pitch(out), dim = 1)   # size: [batch, 66]
        roll = F.softmax(self.roll(out), dim = 1)   # size: [batch, 66]

        #z_pose = torch.cat((yaw, pitch, roll), dim = 0)
        z_dyn = self.delta(out)   # size: [batch, 60]

        return yaw, pitch, roll, z_dyn

if __name__ == '__main__':
    x = torch.randn(5, 3, 256, 256)
    model = MotionEncoder().cuda()
    print(model(x.cuda())[3].shape)   
