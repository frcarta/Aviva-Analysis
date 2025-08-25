import torch
import torch.nn.functional as F
import math


def degrade_image(img_hr: torch.Tensor, r: int) -> torch.Tensor:
    """
    Degrades a high-resolution image by Gaussian filtering and downsampling.

    Args:
        img_hr (torch.Tensor): High-res image, shape (C, H, W)
        r (int): Downsampling factor

    Returns:
        img_lr (torch.Tensor): Low-res image, shape (C, H//r, W//r)
    """
    with torch.no_grad():
        H, W = img_hr.shape[2], img_hr.shape[3]
        sigma = 0.5 * r
        radius = int(math.ceil(3 * sigma))
        kernel_size = 2 * radius + 1

        # Create 1D Gaussian kernel
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss /= gauss.sum()

        # Create 2D separable kernel
        kernel2d = gauss[:, None] @ gauss[None, :]
        kernel2d = kernel2d.expand(1, 1, kernel_size, kernel_size)

        # Pad image
        img_padded = F.pad(img_hr, (radius, radius, radius, radius), mode="reflect")

        # Convolve with Gaussian
        img_filtered = F.conv2d(img_padded, kernel2d)

        # Downsample by factor r
        img_lr = img_filtered[0, :, ::r, ::r]

        return img_lr
