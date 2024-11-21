import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *

class CanonicalKeypointDetector(nn.Module): 
    def __init__(self, in_C: int = 3, num_kp: int = 20, temperature: int = 1, estimate_jacobian: bool = False):
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

        if estimate_jacobian:
            self.conv_jacobian = nn.Conv3d(in_channels = 32, out_channels = 9 * self.num_kp, kernel_size = 7, stride = 1, padding = 3)
        else:
            self.conv_jacobian = None

    @staticmethod
    def gaussian2kp(heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)  # shape: (B, 20, D, H, W, 1)
        # make_coordinate_grid creates a (D, H, W, 3) shape tensor 
        grid = make_coordinate_grid(shape[2:], dtype=heatmap.type()).unsqueeze(0).unsqueeze(0).to(heatmap.device) # shape: (1, 1, D, H, W, 3)
        keypoints = (heatmap * grid).sum(dim=(2, 3, 4))  # elementwise multiplication
        kp = {'keypoints': keypoints}  # shape: (B, 20, 3)

        return kp

    def forward(self, x):
        '''
        return dict with shape: {
            "keypoints": (B, 20(num_kp), 3),
            "jacobian": (B, 20(num_kp), 3, 3)
        }
        '''
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
        keypoints = self.conv_keypoints(out) # (B, 20, D, H, W)
        final_shape = keypoints.shape

        heatmap = keypoints.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=-1)
        heatmap = F.softmax(heatmap, dim=-1)
        heatmap = heatmap.view(*final_shape)

        result = self.gaussian2kp(heatmap)

        # extract jacobian for loss function calculation
        if self.conv_jacobian is not None:
            jacobian_map = self.conv_jacobian(out)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_kp, 9, final_shape[2], final_shape[3], final_shape[4])  # (B, 20, 9, D, H, W)
            heatmap = heatmap.unsqueeze(2)  # (B, 20, 1, D, H, W)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)  # (B, 20, 9, 1)
            jacobian = jacobian.reshape(jacobian.shape[0], jacobian.shape[1], 3, 3)  # (B, 20, 3, 3)
            result['jacobian'] = jacobian
        
        return result
       
if __name__ == '__main__':
    x = torch.randn(5, 3, 512, 512)
    model = CanonicalKeypointDetector(estimate_jacobian=True)
    result = model(x)
    print(result['keypoints'].shape, result['jacobian'].shape)    
        

