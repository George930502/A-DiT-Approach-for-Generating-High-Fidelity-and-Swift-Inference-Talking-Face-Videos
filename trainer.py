import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from Eapp import AppearanceFeatureExtraction
from Emtn import HeadPoseEstimator_ExpressionDeformationEstimator
from KPDetector import CanonicalKeypointDetector
from MotionFieldEstimator import MotionFieldEstimator
from Generator import Generator
from Emtn_gt import Hopenet
from Discriminator import Discriminator
from utils import *
from losses import *

class Transform:
    """
    Random tps transformation for equivariance constraints.
    reference: https://github.com/AliaksandrSiarohin/first-order-model/blob/master/modules/model.py
    In our experiments TX←Y is implemented using randomly sampled thin plate splines. We sample spline parameters from normal distributions with zero mean and variance
    equal to 0.005 for deformation component and 0.05 for the affine component. For deformation component we use uniform 5 * 5 grid
    """
    def __init__(self, bs, sigma_affine = 0.05, sigma_tps = 0.005, points_tps = 5):
        noise = torch.normal(mean = 0, std = sigma_affine * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        self.control_points = make_coordinate_grid_2d((points_tps, points_tps))
        self.control_points = self.control_points.unsqueeze(0)
        self.control_params = torch.normal(mean = 0, std = sigma_tps * torch.ones([bs, 1, points_tps ** 2]))

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:]).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, align_corners = True, padding_mode = "reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        control_points = self.control_points.type(coordinates.type())
        control_params = self.control_params.type(coordinates.type())
        distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
        distances = torch.abs(distances).sum(-1)

        result = distances ** 2
        result = result * torch.log(distances + 1e-6)
        result = result * control_params
        result = result.sum(dim = 2).view(self.bs, coordinates.shape[1], 1)
        transformed = transformed + result

        return transformed


class VideoSynthesisModel(nn.Module):
    def __init__(self, 
                 AFE: AppearanceFeatureExtraction,
                 CKD: CanonicalKeypointDetector,
                 HPE_EDE: HeadPoseEstimator_ExpressionDeformationEstimator,
                 MFE: MotionFieldEstimator,
                 Generator: Generator,
                 Discriminator: Discriminator,
                 pretrained_path = "pretrained/hopenet_robust_alpha1.pkl",
                 num_bins = 66):
        
        super().__init__()
        self.AFE = AFE
        self.CKD = CKD
        self.HPE_EDE = HPE_EDE
        self.MFE = MFE
        self.Generator = Generator
        self.Discriminator = Discriminator

        self.weights = {
            "P": 10,
            "G": 1,
            "F": 10,
            "E": 20,
            "K": 10,
            "H": 20,
            "D": 5,
        }
        self.losses = {
            "P": PerceptualLoss(),
            "G": GANLoss(),
            "F": FeatureMatchingLoss(),
            "E": EquivarianceLoss(),
            "K": KeypointPriorLoss(),
            "H": HeadPoseLoss(),
            "D": DeformationPriorLoss(),
        }

        pretrained_HPNet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins).cuda()
        pretrained_HPNet.load_state_dict(torch.load(pretrained_path, map_location = torch.device("cuda:0"), weights_only = True))
        for parameter in pretrained_HPNet.parameters():
            parameter.requires_grad = False   # for evaluation only
        self.pretrained_HPNet = pretrained_HPNet

    def forward(self, source_img, drive_img):
        fs = self.AFE(source_img)
        kp_c = self.CKD(source_img)
        # For equivariance loss calculation
        transform = Transform(drive_img.shape[0])
        transformed_drive_img = transform.transform_frame(drive_img)
        cated = torch.cat([source_img, drive_img, transformed_drive_img], dim = 0)
        yaw, pitch, roll, translation, deformation = self.HPE_EDE(cated)
        [yaw_s, yaw_d, yaw_trans_d], [pitch_s, pitch_d, pitch_trans_d], [roll_s, roll_d, roll_trans_d] = (
            torch.chunk(yaw, 3, dim = 0),
            torch.chunk(pitch, 3, dim = 0),
            torch.chunk(roll, 3, dim = 0),
        )
        [translation_s, translation_d, translation_trans_d], [deformation_s, deformation_d, deformation_trans_d] = (
            torch.chunk(translation, 3, dim = 0),
            torch.chunk(deformation, 3, dim = 0),
        )
        kp_s, Rs = transformkp(yaw_s, pitch_s, roll_s, kp_c, translation_s, deformation_s)
        kp_d, Rd = transformkp(yaw_d, pitch_d, roll_d, kp_c, translation_d, deformation_d)
        composited_flow_field, occlusion_mask = self.MFE(fs, kp_s, kp_d, Rs, Rd)
        generated_drive_img = self.Generator(fs, composited_flow_field, occlusion_mask)
        _, features_real_drive = self.Discriminator(drive_img, kp_d)
        output_fake_drive, features_fake_drive = self.Discriminator(generated_drive_img, kp_d)
        
        transformed_kp, _ = transformkp(yaw_trans_d, pitch_trans_d, roll_trans_d, kp_c, translation_trans_d, deformation_trans_d)
        reverse_kp = transform.warp_coordinates(transformed_kp[:, :, :2])

        with torch.no_grad():
            self.pretrained_HPNet.eval()
            real_yaw, real_pitch, real_roll = self.pretrained_HPNet(F.interpolate(apply_imagenet_normalization(cated), size = (224, 224)))
        
        # calculate loss function
        loss = {
            "P": self.weights["P"] * self.losses["P"](generated_drive_img, drive_img),
            "G": self.weights["G"] * self.losses["G"](output_fake_drive, True, False),
            "F": self.weights["F"] * self.losses["F"](features_real_drive, features_fake_drive),
            "E": self.weights["E"] * self.losses["E"](kp_d, reverse_kp),
            "K": self.weights["K"] * self.losses["K"](kp_d),
            "H": self.weights["H"] * self.losses["H"](yaw, pitch, roll, real_yaw, real_pitch, real_roll),
            "D": self.weights["D"] * self.losses["D"](deformation_d),
        }

        return loss, generated_drive_img, transformed_drive_img, kp_s, kp_d, transformed_kp, occlusion_mask

class DiscriminatorFull(nn.Module):
    def __init__(self, Discriminator: Discriminator):
        super().__init__()
        self.Discriminator = Discriminator
        self.weights = {
            "G": 1,
        }
        self.losses = {
            "G": GANLoss(),
        }

    def forward(self, drive_img, generated_drive_img, kp_d):
        output_real_drive, _ = self.Discriminator(drive_img, kp_d)
        output_fake_drive, _ = self.Discriminator(generated_drive_img.detach(), kp_d)
        loss = {
            "G1": self.weights["G"] * self.losses["G"](output_fake_drive, False, True),
            "G2": self.weights["G"] * self.losses["G"](output_real_drive, True, True),
        }
        return loss


def print_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    size_in_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    size_in_mb = size_in_bytes / (1024 * 1024)
    print(f"Total number of parameters: {num_params}")
    print(f"Size of the model: {size_in_mb:.2f} MB")


if __name__ == '__main__':
    g_models = {"AFE": AppearanceFeatureExtraction(), "CKD": CanonicalKeypointDetector(), "HPE_EDE": HeadPoseEstimator_ExpressionDeformationEstimator(), "MFE": MotionFieldEstimator(), "Generator": Generator()}
    d_models = {"Discriminator": Discriminator()}
    source_img = torch.randn(1, 3, 256, 256).cuda()
    drive_img = torch.randn(1, 3, 256, 256).cuda()
    model = VideoSynthesisModel(**g_models, **d_models).cuda()
    output = model(source_img, drive_img)
    print(output[0])
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    print(output[4].shape)
    print(output[5].shape)
    print(output[6].shape)
    print_model_size(model)


