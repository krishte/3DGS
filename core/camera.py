import numpy as np
import torch


class Camera(torch.nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        c2w: torch.Tensor,  # [4, 4] camera-to-world matrix
        gt_image: torch.Tensor,
    ):
        super().__init__()
        # Image dimensions
        self.width = width
        self.height = height

        # Intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.c2w: torch.Tensor
        self.w2c: torch.Tensor
        self.gt_image: torch.Tensor
        self.register_buffer("c2w", c2w)
        self.register_buffer("w2c", torch.inverse(c2w))
        self.register_buffer("gt_image", gt_image)

    @property
    def position(self):
        """Camera position in world coordinates"""
        return self.c2w[:3, 3]

    @property
    def rotation(self):
        """Camera rotation matrix (world frame)"""
        return self.c2w[:3, :3]

    @property
    def intrinsic_matrix(self):
        """Returns K matrix [3, 3]"""
        K = torch.zeros(3, 3, device=self.c2w.device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K[2, 2] = 1.0
        return K

    @classmethod
    def from_nerf_json(cls, width, height, fov_x, transform_matrix, gt_image):
        """Create from NeRF-style JSON format"""
        # Compute focal length from FOV
        fx = width / (2 * np.tan(fov_x / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2

        # NeRF gives camera-to-world directly
        c2w = torch.tensor(transform_matrix, dtype=torch.float32)

        return cls(width, height, fx, fy, cx, cy, c2w, gt_image)
