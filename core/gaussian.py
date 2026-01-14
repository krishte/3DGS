import torch


class GaussianSplats(torch.nn.Module):
    def __init__(self, num_splats):
        super().__init__()
        self._num_splats = num_splats
        self._positions = torch.nn.Parameter(2.0 * torch.rand(num_splats, 3) - 1.0)
        self._scales = torch.nn.Parameter(torch.randn(num_splats, 3) * 0.01 - 4.6)

        # Quaternions - start near identity [1,0,0,0]
        quats = torch.zeros(num_splats, 4)
        quats[:, 0] = 1.0
        self._quaternions = torch.nn.Parameter(quats + torch.randn(num_splats, 4) * 0.1)

        # Opacities - start semi-transparent
        self._opacity_logits = torch.nn.Parameter(
            torch.ones(num_splats) * -2.0
        )  # sigmoid(-2) ≈ 0.12

        # Single rgb per splat in [0,1]
        self._colors = torch.nn.Parameter(torch.rand(num_splats, 1, 3))

    def get_opacities(self):
        return torch.sigmoid(self._opacity_logits)

    def get_scales(self):
        return torch.diag_embed(torch.exp(self._scales))

    def get_rotations(self):
        """
        q: [N, 4] tensor of unit quaternions [w, x, y, z]
        returns: [N, 3, 3] rotation matrices
        """
        # Normalize
        q = self._quaternions / self._quaternions.norm(dim=-1, keepdim=True)

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # Build rotation matrix
        R = torch.stack(
            [
                torch.stack(
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    dim=-1,
                ),
                torch.stack(
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    dim=-1,
                ),
                torch.stack(
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    dim=-1,
                ),
            ],
            dim=-2,
        )

        return R

    def get_covariances_3d(self):
        R = self.get_rotations()
        S = self.get_scales()

        RS = R @ S  # [N, 3, 3]
        return RS @ RS.transpose(-2, -1)  # [N, 3, 3]
