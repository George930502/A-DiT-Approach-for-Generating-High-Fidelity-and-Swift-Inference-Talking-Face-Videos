import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class MotionEncoder(nn.Module): 
    def __init__(self, num_kp: int = 20):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        self.norm1 = nn.BatchNorm2d(num_features = 64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.conv2 = nn.Conv2d(in_channels= 64, out_channels = 256, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(num_features = 256)

        self.ResBottleneckBlock_256_1 = ResBottleneck(in_C = 256, out_C = 256)
        self.ResBottleneckBlock_256_2 = ResBottleneck(in_C = 256, out_C = 256)
        self.ResBottleneckBlock_256_3 = ResBottleneck(in_C = 256, out_C = 256)

        self.conv3 = nn.Conv2d(in_channels= 256, out_channels = 512, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(num_features = 512)

        self.ResBottleneckBlock_512_downsample = ResBottleneck(in_C = 512, out_C = 512, stride = 2)

        self.ResBottleneckBlock_512_1 = ResBottleneck(in_C = 512, out_C = 512)
        self.ResBottleneckBlock_512_2 = ResBottleneck(in_C = 512, out_C = 512)
        self.ResBottleneckBlock_512_3 = ResBottleneck(in_C = 512, out_C = 512)

        self.conv4 = nn.Conv2d(in_channels= 512, out_channels = 1024, kernel_size=1)
        self.norm4 = nn.BatchNorm2d(num_features = 1024)

        self.ResBottleneckBlock_1024_downsample = ResBottleneck(in_C = 1024, out_C = 1024, stride = 2)

        self.ResBottleneckBlock_1024_1 = ResBottleneck(in_C = 1024, out_C = 1024)
        self.ResBottleneckBlock_1024_2 = ResBottleneck(in_C = 1024, out_C = 1024)
        self.ResBottleneckBlock_1024_3 = ResBottleneck(in_C = 1024, out_C = 1024)
        self.ResBottleneckBlock_1024_4 = ResBottleneck(in_C = 1024, out_C = 1024)
        self.ResBottleneckBlock_1024_5 = ResBottleneck(in_C = 1024, out_C = 1024)

        self.conv5 = nn.Conv2d(in_channels = 1024, out_channels = 2048, kernel_size=1)
        self.norm5 = nn.BatchNorm2d(num_features = 2048)

        self.ResBottleneckBlock_2048_downsample = ResBottleneck(in_C = 2048, out_C = 2048, stride = 2)

        self.ResBottleneckBlock_2048_1 = ResBottleneck(in_C = 2048, out_C = 2048)
        self.ResBottleneckBlock_2048_2 = ResBottleneck(in_C = 2048, out_C = 2048)
        
        '''
        self.yaw = nn.Linear(in_features = 2048, out_features = 66)
        self.pitch = nn.Linear(in_features = 2048, out_features = 66)
        self.roll = nn.Linear(in_features = 2048, out_features = 66)
        self.translation = nn.Linear(in_features = 2048, out_features = 3)
        self.deformation = nn.Linear(in_features = 2048, out_features = 60)
        '''
        self.num_kp = num_kp
        self.fc_translation = nn.Linear(in_features = 2048, out_features = 3)
        self.fc_deformation = nn.Linear(in_features = 2048, out_features = 3 * self.num_kp)
    
    # The full angle range is divided into 66 bins for rotation angles, and the network predicts which bin the target angle is in
    def forward(self, x):
        '''
        '''
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.ResBottleneckBlock_256_1(out)
        out = self.ResBottleneckBlock_256_2(out)
        out = self.ResBottleneckBlock_256_3(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.ResBottleneckBlock_512_downsample(out)

        out = self.ResBottleneckBlock_512_1(out)
        out = self.ResBottleneckBlock_512_2(out)
        out = self.ResBottleneckBlock_512_3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = self.ResBottleneckBlock_1024_downsample(out)

        out = self.ResBottleneckBlock_1024_1(out)
        out = self.ResBottleneckBlock_1024_2(out)
        out = self.ResBottleneckBlock_1024_3(out)
        out = self.ResBottleneckBlock_1024_4(out)
        out = self.ResBottleneckBlock_1024_5(out)

        out = self.conv5(out)
        out = self.norm5(out)

        out = self.ResBottleneckBlock_2048_downsample(out)
        out = self.ResBottleneckBlock_2048_1(out)
        out = self.ResBottleneckBlock_2048_2(out)   # shape (B, 2048, 16, 16)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)

        '''
        yaw = F.softmax(self.yaw(out), dim = 1)   # size: [batch, 66]
        pitch = F.softmax(self.pitch(out), dim = 1)   # size: [batch, 66]
        roll = F.softmax(self.roll(out), dim = 1)   # size: [batch, 66]
        translation = self.translation(out)   # size: [batch, 3]
        deformation = self.deformation(out)   # size: [batch, 60]

        return yaw, pitch, roll, translation, deformation
        '''
        translation = self.fc_translation(out)
        deformation = self.fc_deformation(out)

        translation = translation.unsqueeze(-2).repeat(1, self.num_kp, 1)  # torch.Size([1, 20, 3])
        deformation = deformation.view(-1, self.num_kp, 3)  # torch.Size([1, 20, 3])

        # return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 'translation': translation, 'deformation': deformation}
        return {'translation': translation, 'deformation': deformation}
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = MotionEncoder().cuda()
    print(model(x.cuda())) 
