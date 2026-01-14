import torch
from einops import einsum

from core import Camera, GaussianSplats


def project_gaussians_to_camera(splats: GaussianSplats, camera: Camera):
    """
    Transform Gaussian splats from world space to camera space

    Args:
        splats: GaussianSplats instance
        camera: Camera instance

    Returns:
        positions_cam: [N, 3] positions in camera space
        depths: [N] depth values (z-coordinate in camera space)
    """

    positions_world = splats._positions

    w2c = camera.w2c

    # Convert to homogeneous coordinates
    N = positions_world.shape[0]
    positions_hom = torch.cat(
        [positions_world, torch.ones(N, 1, device=positions_world.device)], dim=1
    )  # [N, 4]

    # Transform to camera space
    positions_cam_hom = positions_hom @ w2c.T  # [N, 4] @ [4, 4] = [N, 4]

    # Divide by w to get Cartesian coordinates
    w = positions_cam_hom[:, 3:4]  # [N, 1]
    positions_cam = positions_cam_hom[:, :3] / w  # [N, 3]

    # Extract depths (z-coordinate in camera space)
    depths = positions_cam[:, 2]  # [N]

    return positions_cam, depths


def project_points_to_image(positions_cam: torch.Tensor, camera: Camera):
    K = camera.intrinsic_matrix

    valid_mask = positions_cam[:, 2] < -0.01

    positions_image_hom = positions_cam @ K.T  # Project: [N, 3] @ [3, 3]^T = [N, 3]
    positions_image = positions_image_hom / positions_image_hom[:, 2:3]
    points_2d = positions_image[:, :2]  # [N, 2]

    # Filter for points reasonably close to image bounds
    # Use a generous margin (e.g., 50 pixels) for Gaussian tails
    margin = 50
    in_bounds = (
        (points_2d[:, 0] > -margin)
        & (points_2d[:, 0] < camera.width + margin)
        & (points_2d[:, 1] > -margin)
        & (points_2d[:, 1] < camera.height + margin)
    )
    valid_mask = valid_mask & in_bounds

    return points_2d, valid_mask


def project_cov_to_2d(
    covs_3d: torch.Tensor, positions_cam: torch.Tensor, camera: Camera
):
    """
    Project 3D covariance matrices to 2D

    Args:
        covs_3d: [N, 3, 3] covariance matrices in world space
        positions_cam: [N, 3] positions in camera space
        camera: Camera instance

    Returns:
        covs_2d: [N, 2, 2] covariance matrices in image space
    """

    N = covs_3d.shape[0]
    X, Y, Z = positions_cam[:, 0], positions_cam[:, 1], positions_cam[:, 2]
    # Compute Jacobian of perspective projection [N, 2, 3]
    # J = [ fx/Z    0     -fx*X/Z² ]
    #     [   0    fy/Z   -fy*Y/Z² ]
    J = torch.zeros(N, 2, 3, device=covs_3d.device)
    J[:, 0, 0] = camera.fx / Z
    J[:, 0, 2] = -camera.fx * X / (Z * Z)
    J[:, 1, 1] = camera.fy / Z
    J[:, 1, 2] = -camera.fy * Y / (Z * Z)

    w2c_rotation = camera.w2c[:3, :3]
    covs_cam = einsum(w2c_rotation, covs_3d, w2c_rotation, "i j, n j k, l k -> n i l")
    covs_2d = J @ covs_cam @ J.transpose(-2, -1)

    return covs_2d
