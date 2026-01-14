from typing import Any, Dict

import torch

from core import GaussianSplats


class DensificationController:
    """
    Manages adaptive Gaussian densification via splitting and cloning.

    Paper schedule:
    - Track gradients continuously
    - Densify every 100 iterations from iteration 500 to 15,000
    - Split: high gradient + large scale → divide into 2 smaller Gaussians
    - Clone: high gradient + small scale → duplicate Gaussian
    """

    def __init__(
        self,
        gradient_threshold: float = 0.0002,
        scale_threshold: float = 0.01,
        densify_interval: int = 100,
        densify_start: int = 500,
        densify_end: int = 15000,
    ):
        self.gradient_threshold = gradient_threshold
        self.scale_threshold = scale_threshold
        self.densify_interval = densify_interval
        self.densify_start = densify_start
        self.densify_end = densify_end

        # Gradient accumulation
        self.gradient_accum = None  # [N] - accumulated positional gradients
        self.gradient_count = 0

    def track_gradients(self, splats: GaussianSplats):
        """
        Accumulate positional gradients for densification decisions.
        Call this after backward() but before optimizer.zero_grad().

        Args:
            splats: GaussianSplats with computed gradients
        """
        if splats._positions.grad is None:
            return

        # Compute gradient magnitude: ||grad||_2 for each Gaussian
        grad_norm = torch.norm(splats._positions.grad, dim=1)  # [N]

        if self.gradient_accum is None:
            self.gradient_accum = grad_norm
        else:
            self.gradient_accum += grad_norm

        self.gradient_count += 1

    def should_densify(self, iteration: int) -> bool:
        """Check if densification should run at this iteration"""
        if iteration < self.densify_start or iteration > self.densify_end:
            return False
        return (iteration - self.densify_start) % self.densify_interval == 0

    def densify(self, splats: GaussianSplats, optimizer) -> Dict[str, Any]:
        """
        Perform densification: split large high-gradient Gaussians,
        clone small high-gradient Gaussians.

        Returns:
            Statistics: {'num_split': int, 'num_cloned': int, 'total_added': int}
        """
        if self.gradient_accum is None or self.gradient_count == 0:
            return {"num_split": 0, "num_cloned": 0, "total_added": 0}

        # Average accumulated gradients
        avg_gradient = self.gradient_accum / self.gradient_count

        # Get current scales (convert from log-space)
        scales = torch.exp(splats._scales)  # [N, 3]
        max_scale = scales.max(dim=1).values  # [N] - largest axis per Gaussian

        # Identify candidates
        high_gradient = avg_gradient > self.gradient_threshold
        large_scale = max_scale > self.scale_threshold

        # Split: high gradient + large scale
        split_mask = high_gradient & large_scale
        num_split = split_mask.sum().item()

        # Clone: high gradient + small scale
        clone_mask = high_gradient & ~large_scale
        num_cloned = clone_mask.sum().item()

        # Perform splitting
        if num_split > 0:
            self._split_gaussians(splats, split_mask, optimizer)

        # Perform cloning
        if num_cloned > 0:
            self._clone_gaussians(splats, clone_mask, optimizer)

        # Reset gradient accumulation
        self.gradient_accum = None
        self.gradient_count = 0

        return {
            "num_split": num_split,
            "num_cloned": num_cloned,
            "total_added": num_split + num_cloned,
        }

    def _split_gaussians(self, splats: GaussianSplats, mask: torch.Tensor, optimizer):
        """
        Split Gaussians: create 2 new Gaussians from each selected one.
        New Gaussians have:
        - Positions: offset along principal axis
        - Scales: reduced by factor of 1.6
        - Other params: copied
        """
        # Extract parameters of Gaussians to split
        split_positions = splats._positions.data[mask]  # [K, 3]
        split_scales = splats._scales.data[mask]  # [K, 3]
        split_quaternions = splats._quaternions.data[mask]  # [K, 4]
        split_opacities = splats._opacity_logits.data[mask]  # [K]
        split_colors = splats._colors.data[mask]  # [K, 1, 3]

        K = split_positions.shape[0]

        # Sample offset along principal axis (largest scale dimension)
        # Create 2 new Gaussians per split
        offsets = torch.randn(K, 2, 3, device=split_positions.device) * 0.5
        new_positions = split_positions.unsqueeze(1) + offsets  # [K, 2, 3]
        new_positions = new_positions.reshape(-1, 3)  # [2K, 3]

        # Reduce scale by 1.6x
        new_scales = split_scales.unsqueeze(1).repeat(1, 2, 1) - torch.log(
            torch.tensor(1.6)
        )
        new_scales = new_scales.reshape(-1, 3)  # [2K, 3]

        # Duplicate other parameters
        new_quaternions = split_quaternions.unsqueeze(1).repeat(1, 2, 1).reshape(-1, 4)
        new_opacities = split_opacities.unsqueeze(1).repeat(1, 2).reshape(-1)
        new_colors = split_colors.unsqueeze(1).repeat(1, 2, 1, 1).reshape(-1, 1, 3)

        # Add new Gaussians to the model
        self._add_gaussians(
            splats,
            optimizer,
            new_positions,
            new_scales,
            new_quaternions,
            new_opacities,
            new_colors,
        )

        # Remove original split Gaussians
        self._remove_gaussians(splats, optimizer, mask)

    def _clone_gaussians(self, splats: GaussianSplats, mask: torch.Tensor, optimizer):
        """
        Clone Gaussians: duplicate selected Gaussians with small random offset.
        """
        # Extract parameters
        clone_positions = splats._positions.data[mask]
        clone_scales = splats._scales.data[mask]
        clone_quaternions = splats._quaternions.data[mask]
        clone_opacities = splats._opacity_logits.data[mask]
        clone_colors = splats._colors.data[mask]

        # Add small random offset to positions
        offset = torch.randn_like(clone_positions) * 0.01
        new_positions = clone_positions + offset

        # Add cloned Gaussians
        self._add_gaussians(
            splats,
            optimizer,
            new_positions,
            clone_scales,
            clone_quaternions,
            clone_opacities,
            clone_colors,
        )

    def _add_gaussians(
        self,
        splats: GaussianSplats,
        optimizer,
        positions,
        scales,
        quaternions,
        opacities,
        colors,
    ):
        """Add new Gaussians to the model and update optimizer state"""
        # Concatenate new parameters
        splats._positions.data = torch.cat([splats._positions.data, positions], dim=0)
        splats._scales.data = torch.cat([splats._scales.data, scales], dim=0)
        splats._quaternions.data = torch.cat(
            [splats._quaternions.data, quaternions], dim=0
        )
        splats._opacity_logits.data = torch.cat(
            [splats._opacity_logits.data, opacities], dim=0
        )
        splats._colors.data = torch.cat([splats._colors.data, colors], dim=0)

        # Update optimizer state (add zeros for new parameters)
        # This is complex - need to extend Adam state dictionaries
        # See helper function below
        self._extend_optimizer_state(optimizer, positions.shape[0])

    def _remove_gaussians(self, splats: GaussianSplats, optimizer, mask: torch.Tensor):
        """Remove Gaussians indicated by mask"""
        keep_mask = ~mask

        splats._positions.data = splats._positions.data[keep_mask]
        splats._scales.data = splats._scales.data[keep_mask]
        splats._quaternions.data = splats._quaternions.data[keep_mask]
        splats._opacity_logits.data = splats._opacity_logits.data[keep_mask]
        splats._colors.data = splats._colors.data[keep_mask]

        # Update optimizer state
        self._prune_optimizer_state(optimizer, keep_mask)

    def _extend_optimizer_state(self, optimizer, num_new: int):
        """Extend optimizer state for new Gaussians (initialize with zeros)"""
        # For each parameter group in Adam optimizer
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    # Adam has 'exp_avg' and 'exp_avg_sq' state
                    if "exp_avg" in state:
                        zeros = torch.zeros(
                            num_new,
                            *param.shape[1:],
                            device=param.device,
                            dtype=param.dtype,
                        )
                        state["exp_avg"] = torch.cat([state["exp_avg"], zeros], dim=0)
                    if "exp_avg_sq" in state:
                        zeros = torch.zeros(
                            num_new,
                            *param.shape[1:],
                            device=param.device,
                            dtype=param.dtype,
                        )
                        state["exp_avg_sq"] = torch.cat(
                            [state["exp_avg_sq"], zeros], dim=0
                        )

    def _prune_optimizer_state(self, optimizer, keep_mask: torch.Tensor):
        """Prune optimizer state for removed Gaussians"""
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    if "exp_avg" in state:
                        state["exp_avg"] = state["exp_avg"][keep_mask]
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"] = state["exp_avg_sq"][keep_mask]
