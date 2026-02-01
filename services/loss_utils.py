"""
Loss functions for 3D Gaussian Splatting training.
"""

import torch
import torch.nn.functional as F


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    channel: int = 3,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Args:
        pred: Predicted image (H, W, C) or (B, H, W, C)
        target: Target image (H, W, C) or (B, H, W, C)
        window_size: Size of the Gaussian window
        sigma: Standard deviation of Gaussian window
        channel: Number of channels

    Returns:
        SSIM value (scalar tensor, higher is better, max=1.0)
    """
    # Handle unbatched input
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # Rearrange from (B, H, W, C) to (B, C, H, W) for conv2d
    pred = pred.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)

    # Create Gaussian window
    window = _create_gaussian_window(window_size, sigma, channel, pred.device)

    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variances
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def _create_gaussian_window(
    window_size: int,
    sigma: float,
    channel: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a 2D Gaussian window for SSIM computation."""
    # 1D Gaussian
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2

    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # 2D Gaussian (outer product)
    window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)

    # Expand for all channels
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()

    return window


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 (mean absolute error) loss between images.

    Args:
        pred: Predicted image
        target: Target image

    Returns:
        L1 loss (scalar tensor)
    """
    return torch.abs(pred - target).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
) -> torch.Tensor:
    """
    Compute combined L1 + SSIM loss for 3DGS training.

    Args:
        pred: Predicted image (H, W, C)
        target: Target image (H, W, C)
        lambda_l1: Weight for L1 loss
        lambda_ssim: Weight for SSIM loss

    Returns:
        Combined loss (scalar tensor)
    """
    l1 = l1_loss(pred, target)
    ssim = compute_ssim(pred, target)
    ssim_loss = 1.0 - ssim

    return lambda_l1 * l1 + lambda_ssim * ssim_loss


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted image [0, 1]
        target: Target image [0, 1]

    Returns:
        PSNR in dB (higher is better)
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()
