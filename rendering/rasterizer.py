import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from core import Camera, GaussianSplats

from .projection import (
    project_cov_to_2d,
    project_gaussians_to_camera,
    project_points_to_image,
)


class GaussianRasterizer:

    def __init__(self):
        pass

    def render_pixels(
        self,
        points_2d,  # [batch_size, max_count+1, 2]
        covs_2d,  # [batch_size, max_count+1, 2, 2]
        colors,  # [batch_size, max_count+1, 3]
        opacities,  # [batch_size, max_count+1]
        tiled_pixel_coords: torch.Tensor,  # [batch_size, tile_size, tile_size, 2]
        tile_size: int = 16,
    ):
        device = points_2d.device
        batch_size = points_2d.shape[0]

        # Reshape for broadcasting
        pixel_coords_expanded = tiled_pixel_coords.unsqueeze(-2)
        points_2d_expanded = points_2d.unsqueeze(1).unsqueeze(1)

        # Compute differences: [batch_size, tile_size, tile_size, max_count+1, 2]
        diffs = pixel_coords_expanded - points_2d_expanded

        # Add epsilon for numerical stability
        eps = 1e-4
        covs_2d_reg = covs_2d + eps * torch.eye(2, device=device).unsqueeze(0)
        covs_2d_inv = torch.linalg.inv(covs_2d_reg)  # [batch_size, max_count+1, 2, 2]

        # Compute Mahalanobis distance: [batch_size, tile_size, tile_size, max_count+1]
        diffs_cov = torch.einsum(
            "ahwnj,anjk->ahwnk", diffs, covs_2d_inv
        )  # [batch_size, tile_size, tile_size, max_count+1, 2]
        mahalanobis = torch.einsum(
            "ahwnj,ahwnj->ahwn", diffs_cov, diffs
        )  # [batch_size, tile_size, tile_size, max_count+1]

        # Gaussian weights: [batch_size, tile_size, tile_size, max_count+1]
        gaussian_weights = torch.exp(-0.5 * mahalanobis)

        # Alpha values: [batch_size, tile_size, tile_size, max_count+1]
        alphas = (
            opacities.unsqueeze(1).unsqueeze(1) * gaussian_weights
        )  # [batch_size, tile_size, tile_size, max_count+1]
        alphas = torch.clamp(alphas, 0.0, 1.0)

        # Compute transmittance (front-to-back): T_i = prod(1 - alpha_j) for j < i
        one_minus_alpha = 1.0 - alphas
        transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones(batch_size, tile_size, tile_size, 1, device=device),
                    one_minus_alpha,
                ],
                dim=-1,
            ),
            dim=-1,
        )[
            :, :, :, :-1
        ]  # [batch_size, tile_size, tile_size, max_count+1]

        # Compute contributions: [batch_size, tile_size, tile_size, max_count+1, 3]
        colors_expanded = colors.unsqueeze(1).unsqueeze(
            1
        )  # [batch_size, 1, 1, max_count+1, 3]
        contributions = (
            transmittance.unsqueeze(-1) * alphas.unsqueeze(-1) * colors_expanded
        )

        # Sum over Gaussians so output: [batch_size, tile_size, tile_size, 3]
        rendered_image = contributions.sum(dim=-2)

        return rendered_image

    def render_tile_batch(
        self,
        splats_per_tiles_mask: torch.Tensor,  # [batch_size, N]
        tiled_pixel_coords: torch.Tensor,  # [batch_size, tile_size, tile_size, 2]
        points_2d,  # [N, 2]
        covs_2d,  # [N, 2, 2]
        colors,  # [N, 3]
        opacities,  # [N]
    ):
        """Fully vectorized batched selection - no Python loops"""
        batch_size, N = splats_per_tiles_mask.shape
        device = splats_per_tiles_mask.device

        splat_indices = []
        for i in range(batch_size):
            indices = torch.nonzero(splats_per_tiles_mask[i], as_tuple=False).squeeze(
                -1
            )
            splat_indices.append(indices)
        splat_indices = pad_sequence(
            splat_indices, batch_first=True, padding_value=N
        )  # [batch_size, max_count]

        if splat_indices.shape[1] == 0:
            tile_size = tiled_pixel_coords.shape[1]
            return torch.zeros(
                (batch_size, tile_size, tile_size, 3),
                dtype=torch.float32,
                device=device,
            )

        # Add padding Gaussian
        points_2d = torch.cat(
            [points_2d, torch.zeros((1, 2), device=device, dtype=points_2d.dtype)]
        )
        covs_2d = torch.cat(
            [covs_2d, torch.eye(2, device=device, dtype=covs_2d.dtype).unsqueeze(0)]
        )
        colors = torch.cat(
            [colors, torch.zeros((1, 3), device=device, dtype=colors.dtype)]
        )
        opacities = torch.cat(
            [opacities, torch.zeros(1, device=device, dtype=opacities.dtype)]
        )

        # Gather using indices (padded indices point to padding Gaussian)
        selected_points = points_2d[splat_indices]  # [batch_size, max_count, 2]
        selected_covs = covs_2d[splat_indices]  # [batch_size, max_count, 2, 2]
        selected_colors = colors[splat_indices]  # [batch_size, max_count, 3]
        selected_opacities = opacities[splat_indices]  # [batch_size, max_count]

        return self.render_pixels(
            selected_points,
            selected_covs,
            selected_colors,
            selected_opacities,
            tiled_pixel_coords,
        )

    def gaussian_bounding_boxes(self, points_2d: torch.Tensor, covs_2d: torch.Tensor):
        # Process in chunks to avoid CUDA solver batch size limitations
        N = covs_2d.shape[0]
        chunk_size = 10000  # Process 10k at a time
        max_deviations = []

        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunk_covs = covs_2d[i:end_idx]

            # Compute max eigenvalue for this chunk
            eigenvalues = torch.linalg.eigvalsh(chunk_covs)  # [chunk_size, 2]
            max_eig = eigenvalues.max(dim=-1, keepdim=True).values  # [chunk_size, 1]
            chunk_max_dev = 3.0 * torch.sqrt(max_eig).squeeze(-1)  # [chunk_size]
            max_deviations.append(chunk_max_dev)

        max_deviation = torch.cat(max_deviations, dim=0)  # [N]

        return torch.stack(
            [
                points_2d[:, 0] - max_deviation,
                points_2d[:, 0] + max_deviation,
                points_2d[:, 1] - max_deviation,
                points_2d[:, 1] + max_deviation,
            ],
            dim=-1,
        )  # [N, 4]

    def assign_gaussians_to_tiles(
        self, image_width, image_height, tile_size, bounding_boxes: torch.Tensor
    ):
        device = bounding_boxes.device

        tile_ranges = bounding_boxes / tile_size
        tile_ranges_int = torch.stack(
            [
                torch.floor(tile_ranges[:, 0]),
                torch.ceil(tile_ranges[:, 1]),
                torch.floor(tile_ranges[:, 2]),
                torch.ceil(tile_ranges[:, 3]),
            ],
            dim=-1,
        ).int()

        num_tiles_x = int(np.ceil(image_width / tile_size))
        num_tiles_y = int(np.ceil(image_height / tile_size))

        # Create tile coordinate grids: [num_tiles_x, num_tiles_y]
        tile_x_coords = torch.arange(num_tiles_x, device=device).unsqueeze(
            1
        )  # [num_tiles_x, 1]
        tile_y_coords = torch.arange(num_tiles_y, device=device).unsqueeze(
            0
        )  # [1, num_tiles_y]

        # Broadcast comparison: [num_tiles_x, num_tiles_y, N]
        # tile_ranges_int shape: [N, 4] where columns are [x_min, x_max, y_min, y_max]
        x_min = tile_ranges_int[:, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        x_max = tile_ranges_int[:, 1].unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        y_min = tile_ranges_int[:, 2].unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        y_max = tile_ranges_int[:, 3].unsqueeze(0).unsqueeze(0)  # [1, 1, N]

        # Vectorized comparison
        x_in_range = (tile_x_coords.unsqueeze(-1) >= x_min) & (
            tile_x_coords.unsqueeze(-1) <= x_max
        )
        y_in_range = (tile_y_coords.unsqueeze(-1) >= y_min) & (
            tile_y_coords.unsqueeze(-1) <= y_max
        )

        splats_per_tile_mask = x_in_range & y_in_range  # [num_tiles_x, num_tiles_y, N]

        return splats_per_tile_mask  # [num_tiles_width, num_tiles_height, N]

    def batch_tiles(
        self,
        splats_per_tiles_mask: torch.Tensor,
        camera: Camera,
        batch_size: int = 100,
        tile_size: int = 16,
    ):
        num_tiles_width, num_tiles_height = splats_per_tiles_mask.shape[:2]
        splats_per_tile_count = splats_per_tiles_mask.sum(
            dim=-1
        ).flatten()  # [num_tiles_width * num_tiles_height]

        splats_per_tiles_mask_flattened = splats_per_tiles_mask.flatten(
            0, 1
        )  # [num_tiles_width*num_tiles_height, N]
        sorted_indices = torch.argsort(splats_per_tile_count)
        num_tiles = splats_per_tile_count.shape[0]
        assert num_tiles % batch_size == 0
        N = splats_per_tiles_mask_flattened.shape[-1]
        batched_splats_per_tiles_mask = splats_per_tiles_mask_flattened[
            sorted_indices
        ].reshape((num_tiles // batch_size, batch_size, N))
        batched_sorted_indices = sorted_indices.reshape(
            (num_tiles // batch_size, batch_size)
        )

        device = splats_per_tiles_mask.device
        y_coords = torch.arange(camera.height, device=device, dtype=torch.float32)
        x_coords = torch.arange(camera.width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        pixel_coords = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        tiled_pixel_coords = pixel_coords.reshape(
            num_tiles_height,
            tile_size,  # Tile rows, pixels per tile row
            num_tiles_width,
            tile_size,  # Tile cols, pixels per tile col
            2,
        )  # [num_tiles_height, tile_size, num_tiles_width, tile_size, 2]

        # Step 2: Transpose to group tiles together
        tiled_pixel_coords = tiled_pixel_coords.permute(0, 2, 1, 3, 4)
        # [num_tiles_height, num_tiles_width, tile_size, tile_size, 2]

        # Step 3: Flatten tiles into single dimension
        tiled_pixel_coords = tiled_pixel_coords.reshape(
            num_tiles, tile_size, tile_size, 2
        )
        # [num_tiles, tile_size, tile_size, 2]

        # Step 4: Batch the tiles
        batched_tiled_pixel_coords = tiled_pixel_coords.reshape(
            num_tiles // batch_size, batch_size, tile_size, tile_size, 2
        )  # [num_batches, batch_size, tile_size, tile_size, 2]

        return (
            batched_splats_per_tiles_mask,
            batched_sorted_indices,
            batched_tiled_pixel_coords,
        )

    def render(self, gaussian_splats: GaussianSplats, camera: Camera):
        positions_cam, depths = project_gaussians_to_camera(gaussian_splats, camera)
        points_2d, valid_mask = project_points_to_image(positions_cam, camera)
        covs_2d = project_cov_to_2d(
            gaussian_splats.get_covariances_3d(), positions_cam, camera
        )

        valid_points_2d = points_2d[valid_mask]
        valid_covs_2d = covs_2d[valid_mask]
        valid_depths = depths[valid_mask]
        valid_colors = gaussian_splats._colors[valid_mask].squeeze(1)
        valid_opacities = gaussian_splats.get_opacities()[valid_mask]

        sorted_indices = torch.argsort(valid_depths, descending=True)

        image = self._render(
            valid_points_2d[sorted_indices],
            valid_covs_2d[sorted_indices],
            valid_colors[sorted_indices],
            valid_opacities[sorted_indices],
            camera,
        )

        return image

    def _render(self, points_2d, covs_2d, colors, opacities, camera: Camera):
        gaussian_bounding_boxes = self.gaussian_bounding_boxes(points_2d, covs_2d)
        splats_per_tile_mask = self.assign_gaussians_to_tiles(
            camera.width, camera.height, 16, gaussian_bounding_boxes
        )
        (
            batched_splats_per_tiles_mask,
            batched_sorted_indices,
            batched_tiled_pixel_coords,
        ) = self.batch_tiles(splats_per_tile_mask, camera, 100, 16)
        # Initialize output image
        device = points_2d.device
        rendered_image = torch.zeros(camera.height, camera.width, 3, device=device)

        tile_size = 16
        num_tiles_x = (camera.width + tile_size - 1) // tile_size
        num_tiles_y = (camera.height + tile_size - 1) // tile_size

        # Process each batch
        num_batches = batched_splats_per_tiles_mask.shape[0]
        for batch_idx in range(num_batches):
            batch_splats_per_tile_mask = batched_splats_per_tiles_mask[batch_idx]
            batch_sorted_indices = batched_sorted_indices[batch_idx]  # [batch_size]
            batch_tiled_pixel_coords = batched_tiled_pixel_coords[batch_idx]
            # Render batch of tiles
            batch_tile_renders = self.render_tile_batch(
                batch_splats_per_tile_mask,
                batch_tiled_pixel_coords,
                points_2d,
                covs_2d,
                colors,
                opacities,
            )  # [batch_size, tile_size, tile_size, 3]
            # Place each rendered tile back into the image
            for i, flat_tile_idx in enumerate(batch_sorted_indices):
                # Convert flat tile index to 2D tile coordinates
                tile_y = flat_tile_idx // num_tiles_x
                tile_x = flat_tile_idx % num_tiles_x

                # Compute pixel coordinates for this tile
                y_start = tile_y * tile_size
                y_end = min(y_start + tile_size, camera.height)
                x_start = tile_x * tile_size
                x_end = min(x_start + tile_size, camera.width)

                # Copy tile to output image
                rendered_image[y_start:y_end, x_start:x_end] = batch_tile_renders[i]

        return rendered_image
