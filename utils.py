import torch

def make_coordinate_grid(spatial_size, dtype):
    """
    Create a meshgrid [-1,1] x [-1,1] x [-1,1] of given spatial_size.
    """
    d, h, w = spatial_size
    z = torch.arange(d).type(dtype)
    x = torch.arange(w).type(dtype)
    y = torch.arange(h).type(dtype)

    z = (2 * (z / (d - 1)) - 1)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    xx = x.view(1, 1, -1).repeat(d, h, 1)
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    zz = z.view(-1, 1, 1).repeat(1, h, w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)  # shape: (x, y, z, 3)
    return meshed

def kp2gaussian(spatial_size, kp_variance):
    """
    Transform keypoints into gaussian like representation
    """
    mean = torch.randn(5, 20, 3)
    number_of_leading_dimensions = len(mean.shape) - 1   # 2

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    coordinate_grid = coordinate_grid.to(mean.device)

    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape  
    coordinate_grid = coordinate_grid.view(*shape)  # shape: (1, 1, x, y, z, 3)

    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)  
    coordinate_grid = coordinate_grid.repeat(*repeats)  # shape: (B, 20, x, y, z, 3)

    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)  # shape: (B, 20, 1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)   # shape: (B, 20, x, y, z)
    return out

if __name__ == '__main__':
    kp2gaussian(spatial_size = (16, 52, 52), kp_variance = 1) 