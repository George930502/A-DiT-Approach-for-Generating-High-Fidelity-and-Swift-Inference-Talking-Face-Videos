import torch
import torch.nn.functional as F

def make_coordinate_grid_3d(spatial_size):
    """
    Create a meshgrid [-1,1] x [-1,1] x [-1,1] of given spatial_size.
    """
    d, h, w = spatial_size
    z = torch.arange(d).cuda()
    x = torch.arange(w).cuda()
    y = torch.arange(h).cuda()

    z = (2 * (z / (d - 1)) - 1)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    xx = x.view(1, 1, -1).repeat(d, h, 1)  # The value of each element is the coordinate of the x-axis
    yy = y.view(1, -1, 1).repeat(d, 1, w)  # The value of each element is the coordinate of the y-axis
    zz = z.view(-1, 1, 1).repeat(1, h, w)  # The value of each element is the coordinate of the z-axis

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)  # returned shape: (x, y, z, 3)
    return meshed

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform keypoints into gaussian like representation
    """
    mean = kp['keypoints']
    number_of_leading_dimensions = len(mean.shape) - 1

    coordinate_grid = make_coordinate_grid_3d(spatial_size, mean.type())
    coordinate_grid = coordinate_grid.to(mean.device)

    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out

def out2heatmap(out, temperature = 0.1):
    final_shape = out.shape
    heatmap = out.view(final_shape[0], final_shape[1], -1)
    # When temperature is small such as 𝑇 = 0.1, Softmax will amplify the largest value and generate a sharp distribution (close to one-hot)
    heatmap = F.softmax(heatmap / temperature, dim = -1)  
    heatmap = heatmap.view(*final_shape)
    return heatmap

def heatmap2kp(heatmap):
    shape = heatmap.shape
    grid = make_coordinate_grid_3d(shape[2:]).unsqueeze(0).unsqueeze(0)
    # The specific location of key points is determined using a weighted average calculation of the coordinate grid
    kp = (heatmap.unsqueeze(-1) * grid).sum(dim=(2, 3, 4))
    return kp

def kp2gaussian_3d(kp, spatial_size, kp_variance = 0.01):
    '''
    Generate a 3D Gaussian distribution associated with each keypoints, representing the spatial influence range corresponding to the keypoints.
    kp shape: [N, num_kp, 3]
    spatial shape: [D, H, W]
    '''
    N, K = kp.shape[:2]
    coordinate_grid = make_coordinate_grid_3d(spatial_size).view(1, 1, *spatial_size, 3).repeat(N, K, 1, 1, 1, 1)
    mean = kp.view(N, K, 1, 1, 1, 3)
    mean_sub = coordinate_grid - mean  # euclidean metric
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)  # apply gaussian formula
    return out

def create_heatmap_representations(fs, kp_s, kp_d):
    '''
    fs shape: [N, C, D, H, W]
    kp_s and kp_d shape: [N, num_kp, 3]
    '''
    spatial_size = fs.shape[2:]  # extract [D, H, W] dims from fs
    heatmap_d = kp2gaussian_3d(kp_d, spatial_size)
    heatmap_s = kp2gaussian_3d(kp_s, spatial_size)
    heatmap = heatmap_d - heatmap_s
    # adding background feature
    zeros = torch.zeros(heatmap.shape[0], 1, *spatial_size).cuda()  # shape: [N, 1, D, H, W]
    heatmap = torch.cat([zeros, heatmap], dim = 1)   # shape: [N, K + 1, D, H, W]
    heatmap = heatmap.unsqueeze(2)   # shape: [N, K + 1, 1, D, H, W]
    return heatmap


def create_sparse_motions(fs, kp_s, kp_d, Rs, Rd):
    N, _, D, H, W = fs.shape
    K = kp_s.shape[1]
    identity_grid = make_coordinate_grid_3d((D, H, W)).view(1, 1, D, H, W, 3).repeat(N, 1, 1, 1, 1, 1)  # shape: [N, 1, D, H, W, 3]
    coordinate_grid = identity_grid.repeat(1, K, 1, 1, 1, 1) - kp_d.view(N, K, 1, 1, 1, 3)  # shape: [N, K, D, H, W, 3]
    jacobian = torch.matmul(Rs, torch.inverse(Rd)).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)  # shape: [N, 1, 1, 1, 1, 3, 3]
    coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1)).squeeze(-1)
    driving_to_source = coordinate_grid + kp_s.view(N, K, 1, 1, 1, 3)
    # adding background feature
    sparse_motions = torch.cat([identity_grid, driving_to_source], dim = 1)  # shape: [N, K + 1, D, H, W, 3]
    return sparse_motions


def create_deformed_source_image(fs, sparse_motions):
    '''
    perform warping operation of fs and flows (from keypoints)
    '''
    N, _, D, H, W = fs.shape
    K = sparse_motions.shape[1] - 1
    source_repeat = fs.unsqueeze(1).repeat(1, K + 1, 1, 1, 1, 1).view(N * (K + 1), -1, D, H, W)  # shape: [N * (K + 1), C, D, H, W]
    sparse_motions = sparse_motions.view((N * (K + 1), D, H, W, -1)) # shape: [N * (K + 1), D, H, W, 3]
    sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners = True)  # shape: [N * (K + 1), C, D, H, W]
    sparse_deformed = sparse_deformed.view((N, K + 1, -1, D, H, W))  # shape: [N, K + 1, C, D, H, W]
    return sparse_deformed


if __name__ == '__main__':
    print(heatmap2kp(torch.randn(5, 20, 16, 256, 256)).shape)