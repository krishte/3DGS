import random

import torch
from tqdm import tqdm

from core import Scene
from rendering import GaussianRasterizer

from .loss import compute_loss
from .optimizer import setup_optimizer
from .scheduler import create_scheduler


def train(
    scene: Scene,
    rasterizer: GaussianRasterizer,
    num_iterations: int = 30000,
):
    """
    Main training loop

    Args:
        scene: Scene containing cameras and ground truth images
        rasterizer: GaussianRasterizer for rendering
        num_iterations: Total training iterations
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Move model to device
    scene.to(device)

    # Setup optimizer and scheduler
    optimizer = setup_optimizer(scene.gaussian_splats)
    scheduler = create_scheduler(optimizer, max_iterations=num_iterations)

    # Training loop
    print(f"Starting training for {num_iterations} iterations...")
    progress_bar = tqdm(range(num_iterations), desc="Training")

    for iteration in progress_bar:
        # Sample random training camera
        camera = random.choice(scene.cameras)
        gt_image = camera.gt_image  # [H, W, 3]
        # Render
        rendered_image = rasterizer.render(scene.gaussian_splats, camera)  # [H, W, 3]

        # Compute loss
        losses = compute_loss(rendered_image, gt_image)
        total_loss = losses["total"]

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(scene.gaussian_splats.parameters(), max_norm=1.0)
        optimizer.step()

        # Update learning rates
        scheduler.step()

        # Logging
        progress_bar.set_postfix(
            {
                "loss": f"{total_loss.item():.4f}",
                "l1": f"{losses['l1'].item():.4f}",
                "ssim": f"{losses['ssim'].item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )
