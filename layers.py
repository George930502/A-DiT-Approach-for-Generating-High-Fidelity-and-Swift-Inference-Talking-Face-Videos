import torch.nn as nn
import torch.nn.functional as F
import torch

# In the architecture of our base model, we replace all BatchNorms with GroupNorms, 
# and all convolutional layers, except the first and the last ones, are used with weight standardization

# redefine convolution

class Conv2d_WS(nn.Conv2d):
    '''
    Inherited from nn.Conv2d class
    '''
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim = True)
        weight_std = weight.std(dim=[1, 2, 3], keepdim = True) + 1e-5  # Adding a small epsilon to avoid division by zero
        weight_normalized = (weight - weight_mean) / weight_std
        output = F.conv2d(x, weight_normalized, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

class ResBlock2DCustom(nn.Module):
    # num_groups can be a parameter
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv2d_ws = Conv2d_WS(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv2d_res = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        out = F.group_norm(x, num_groups = 32)
        out = F.relu(out)
        out = self.conv2d_ws(out)
        out = F.group_norm(out, num_groups = 32)
        out = F.relu(out)
        out = self.conv2d(out)
        out_prev = self.conv2d_res(x)
        return out + out_prev
    

class Conv3d_WS(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        
    def forward(self, x):
        weight = self.conv.weight
        weight_mean = weight.mean(dim=[1, 2, 3, 4], keepdim = True)
        weight_std = weight.std(dim=[1, 2, 3, 4], keepdim = True) + 1e-5  # Adding a small epsilon to avoid division by zero
        weight_normalized = (weight - weight_mean) / weight_std
        output = F.conv3d(x, weight_normalized, self.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return output
    
class ResBlock3DCustom(nn.Module):
    # num_groups can be a parameter
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv3d_ws = Conv3d_WS(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv3d_res = nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv3d = nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        out = F.group_norm(x, num_groups = 32)
        out = F.relu(out)
        out = self.conv3d_ws(out)
        out = F.group_norm(out, num_groups = 32)
        out = F.relu(out)
        out = self.conv3d(out)
        out_prev = self.conv3d_res(x)
        return out + out_prev

class ResBlock2D(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_features, out_channels = out_features, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = nn.Conv2d(in_channels = out_features, out_channels = out_features, kernel_size = kernel_size, stride = stride, padding = padding)
        self.norm1 = nn.BatchNorm2d(num_features = in_features)
        self.norm2 = nn.BatchNorm2d(num_features = out_features)

    def forward(self, x):
        out = F.relu(self.norm1(x))
        out = self.conv1(out)
        out = F.relu(self.norm2(out))
        out = self.conv2(out)
        out += x
        return out

class ResBlock3D(nn.Module):
    '''
    To extract temporal feature:
    input size: (N, Cin, F, H, W)
    output size: (N, Cout, Fout, Hout, Wout)
    '''
    def __init__(self, in_C: int, out_C: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels = in_C, out_channels = out_C, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv3d(in_channels = out_C, out_channels = out_C, kernel_size = 3, padding = 1)
        self.norm1 = nn.BatchNorm3d(num_features = in_C)
        self.norm2 = nn.BatchNorm3d(num_features = out_C)

    def forward(self, x):
        out = F.relu(self.norm1(x))
        out = self.conv1(out)
        out = F.relu(self.norm2(out))
        out = self.conv2(out)
        out += x
        return out
    
class UpBlock3D(nn.Module):
    '''
    Upsampling block for use in encoder
    '''
    def __init__(self, in_C: int, out_C: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels = in_C, out_channels = out_C, kernel_size = 3, stride = 1, padding = 1)
        self.norm = nn.BatchNorm3d(num_features = out_C)  # affine is default to "True"

    def forward(self, x):
        # x = F.interpolate(x, scale_factor = (1, 2, 2), mode = 'trilinear')    # Upsample by a factor of 2
        _, _, d, h, w = x.shape
        x = F.interpolate(x, size = (d, 2 * h, 2 * w))
        out = self.conv(x)
        out = F.relu(self.norm(out))
        return out

class DownBlock2D(nn.Module):
    '''
    Downsampling block for use in encoder
    '''
    def __init__(self, in_C: int, out_C: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_C, out_channels = out_C, kernel_size = 3, stride = 1, padding = 1)
        self.norm = nn.BatchNorm2d(num_features = out_C)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out
    


class ResBottleneck(nn.Module):
    def __init__(self, in_C: int, out_C: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.in_C = in_C
        self.out_C = out_C
        self.conv1 = nn.Conv2d(in_channels = in_C, out_channels = out_C // 4, kernel_size = 1)
        self.norm1 = nn.BatchNorm2d(num_features = out_C // 4)
        self.conv2 = nn.Conv2d(in_channels = out_C // 4, out_channels = out_C // 4, kernel_size = 3, stride = stride, padding = 1)
        self.norm2 = nn.BatchNorm2d(num_features = out_C // 4)
        self.conv3 = nn.Conv2d(in_channels = out_C // 4, out_channels = out_C, kernel_size = 1)
        self.norm3 = nn.BatchNorm2d(num_features = out_C)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_C, out_C, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(num_features = out_C))
 
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.norm1(out))
        out = self.conv2(out)
        out = F.relu(self.norm2(out))
        out = self.conv3(out)
        out = self.norm3(out)
        # residual connection
        if self.stride != 1:
            shortcut = self.shortcut(x)
            out += shortcut
        else:
            out += x
        return F.relu(out)


class SPADE(nn.Module):
    # 𝑛 denotes the dimension of a convolutional layer (either 2D or 3D) 
    # and x denotes the number of output channels
    def __init__(self, in_C: int, out_C: int):
        super().__init__()
        self.instanceNorm2d = nn.InstanceNorm2d(in_C)
        self.weight = nn.Conv2d(in_channels = in_C, out_channels = out_C, kernel_size = 3, padding = 1)
        self.bias = nn.Conv2d(in_channels = in_C, out_channels = out_C, kernel_size = 3, padding = 1)
        #self.upsample = upsample

        #if self.upsample:
        #    self.upsampling = nn.Upsample(scale_factor = 2, mode = 'bilinear')
    
    def forward(self, x, identity_tensor):
        x_normalized = self.instanceNorm2d(x)
        identity_sliced = identity_tensor[: x.size(0)]  # Get the batch size of the input

        weight = self.weight(identity_sliced)
        bias = self.bias(identity_sliced)

        #if self.upsample:
        #    weight = self.upsampling(weight)
        #    bias = self.upsampling(bias)
        
        y = torch.mul(x_normalized, weight) + bias  # element-wise multiplication
        return y

class SPADEResBlock(nn.Module):
    expansion = 4
    def __init__(self, in_C: int, out_C: int, stride: int, downsample: bool = False):
        super().__init__()
        middle_channels = min(in_C, out_C)

        self.spade1 = SPADE(in_C, middle_channels)
        self.conv1 = nn.Conv2d(middle_channels, middle_channels, kernel_size = 3, padding = 1)

        self.spade2 = SPADE(middle_channels, middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_C * 4, kernel_size = 3, padding = 1)
        
        self.downsample = downsample
        if self.downsample:
            self.conv_shortcut = nn.Conv2d(in_C, out_C, kernel_size = 1, stride = stride)
        
    def forward(self, x, identity_tensor):
        # SPADE tensor shape H x W is at most 64 × 64
        # identity_tensor: torch.Size([n, C, H, W])
        x_1 = F.relu(self.conv1(self.spade1(x, identity_tensor)))
        x_2 = F.relu(self.conv2(self.spade2(x_1, identity_tensor)))

        if self.downsample:
            shortcut = self.conv_shortcut(x)
            y = F.relu(x_2 + shortcut)
        else:
            y = F.relu(x_2 + x)
        return y



if __name__ == "__main__":
    # Create a random input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 128, 128)  

    # Create a convolutional layer with weight standardization
    identity_tensor = torch.randn(2, input_tensor.size(1), 64, 64)

    # Forward pass
    model = Conv2d_WS(3, 64, 3)
    output = model(input_tensor)

    print("Output shape:", output.shape)  # Output shape
