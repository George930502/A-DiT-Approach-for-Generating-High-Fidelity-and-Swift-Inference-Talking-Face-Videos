import torch
import torch.nn.functional as F

def make_coordinate_grid_2d(spatial_size):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(h).cuda()
    y = torch.arange(w).cuda()
    x = 2 * (x / (h - 1)) - 1
    y = 2 * (y / (w - 1)) - 1
    xx = x.view(-1, 1).repeat(1, w)
    yy = y.view(1, -1).repeat(h, 1)
    meshed = torch.cat([yy.unsqueeze(2), xx.unsqueeze(2)], 2)
    return meshed  # returned shape: (x, y, 2)

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

def kp2gaussian_2d(kp, spatial_size, kp_variance=0.01):
    '''
    Generate a 2D Gaussian distribution associated with each keypoints, representing the spatial influence range corresponding to the keypoints.
    kp shape: [N, num_kp, 3]
    spatial shape: [H, W]
    '''
    N, K = kp.shape[:2]
    coordinate_grid = make_coordinate_grid_2d(spatial_size).view(1, 1, *spatial_size, 2).repeat(N, K, 1, 1, 1)
    mean = kp.view(N, K, 1, 1, 2)
    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out  # shape: [N, num_kp, H, W]


def rotation_matrix_x(theta):
    theta = theta.view(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([c, z, s], 2),
            torch.cat([z, o, z], 2),
            torch.cat([-s, z, c], 2),
        ], dim = 1)

def rotation_matrix_y(theta):
    theta = theta.view(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([o, z, z], 2),
            torch.cat([z, c, -s], 2),
            torch.cat([z, s, c], 2),
        ], dim = 1)


def rotation_matrix_z(theta):
    theta = theta.view(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([c, -s, z], 2),
            torch.cat([s, c, z], 2),
            torch.cat([z, z, o], 2),
        ], dim = 1)

def transformkp(yaw, pitch, roll, canonical_kp, translation, deformation):
    '''
    yaw shape: [B]
    pitch shape: [B]
    roll.shape: [B]
    canonical_kp shape: [B, 20, 3]
    translation shape: [B, 3]
    deformation shape: [B, 20, 3]
    '''
    rot_mat = rotation_matrix_x(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_z(roll)  # shape: [B, 3, 3]
    transformed_kp = torch.matmul(rot_mat.unsqueeze(1), canonical_kp.unsqueeze(-1)).squeeze(-1) + translation.unsqueeze(1) + deformation
    return transformed_kp, rot_mat


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


def apply_imagenet_normalization(input):
    mean = input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (input - mean) / std
    return output

def apply_vggface_normalization(input):
    mean = input.new_tensor([129.1863, 104.7624, 93.5940]).view(1, 3, 1, 1)
    std = input.new_tensor([1, 1, 1]).view(1, 3, 1, 1)
    output = (input * 255 - mean) / std
    return output

if __name__ == '__main__':
    kp = torch.randn(1, 20, 2).cuda()
    print(kp2gaussian_2d(kp, torch.Size([256, 256])).shape)
    #print(rotation_matrix_x(torch.tensor([-0.0228])).shape)