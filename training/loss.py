import torch
from pytorch_msssim import ssim as pytorch_ssim


def l1_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss between rendered and ground truth images

    Args:
        rendered: [H, W, 3] rendered image
        gt: [H, W, 3] ground truth image

    Returns:
        L1 loss (scalar)
    """
    return torch.abs(rendered - gt).mean()


def compute_loss(rendered: torch.Tensor, gt: torch.Tensor) -> dict:
    """
    Compute combined loss for Gaussian Splatting

    Args:
        rendered: [H, W, 3] rendered image
        gt: [H, W, 3] ground truth image
        lambda_dssim: weight for D-SSIM loss (default 0.2)

    Returns:
        Dictionary containing:
            - 'total': total loss
            - 'l1': L1 loss component
            - 'dssim': D-SSIM loss component
    """
    # rendered/gt: [H, W, 3] → [1, 3, H, W] for pytorch-msssim
    rendered_bhwc = rendered.permute(2, 0, 1).unsqueeze(0)
    gt_bhwc = gt.permute(2, 0, 1).unsqueeze(0)

    l1 = l1_loss(rendered, gt)
    ssim_val = pytorch_ssim(rendered_bhwc, gt_bhwc, data_range=1.0)

    total = 0.8 * l1 + 0.2 * (1.0 - ssim_val)
    return {"total": total, "l1": l1, "ssim": ssim_val}
