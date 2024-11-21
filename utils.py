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
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)  # shape: (d, h, w, 3)
    return meshed

if __name__ == '__main__':
    make_coordinate_grid(spatial_size = (16, 52, 52), dtype = torch.float32) 