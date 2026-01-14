import torch

from core import GaussianSplats


def setup_optimizer(gaussian_splats: GaussianSplats) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates for different parameter groups

    Args:
        gaussian_splats: GaussianSplats model (nn.Module)

    Returns:
        Configured Adam optimizer
    """
    return torch.optim.Adam(
        [
            {"params": gaussian_splats._positions, "lr": 0.00016, "name": "positions"},
            {
                "params": gaussian_splats._quaternions,
                "lr": 0.001,
                "name": "quaternions",
            },
            {"params": gaussian_splats._scales, "lr": 0.005, "name": "scales"},
            {
                "params": gaussian_splats._opacity_logits,
                "lr": 0.05,
                "name": "opacities",
            },
            {"params": gaussian_splats._colors, "lr": 0.0025, "name": "colors"},
        ],
        eps=1e-15,
    )
