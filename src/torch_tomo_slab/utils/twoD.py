"""2D image processing utilities for tomographic data.

This module provides specialized 2D image processing functions used in
the tomographic segmentation pipeline, including robust normalization
and local variance computation.
"""

import torch
import torch.nn.functional as F

from torch_tomo_slab import config, constants


def robust_normalization(data: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor using robust statistics (median and percentile range).

    Parameters
    ----------
    data : torch.Tensor
        Input tensor to normalize.

    Returns
    -------
    torch.Tensor
        Normalized tensor with median=0 and scaled by 5th-95th percentile range.
        If range is too small, returns median-centered data without scaling.
    """
    data = data.float()
    p5, p95 = torch.quantile(data, 0.05), torch.quantile(data, 0.95)
    if p95 - p5 < 1e-5: return data - torch.median(data)
    return (data - torch.median(data)) / (p95 - p5)

def local_variance_2d(image: torch.Tensor) -> torch.Tensor:
    """
    Calculate local variance of 2D image using convolution.

    Parameters
    ----------
    image : torch.Tensor
        Input 2D image tensor.

    Returns
    -------
    torch.Tensor
        Local variance map of same size as input.
        Computed using kernel size from config.AUGMENTATION_CONFIG['LOCAL_VARIANCE_KERNEL_SIZE'].
    """
    kernel_size = constants.AUGMENTATION_CONFIG['LOCAL_VARIANCE_KERNEL_SIZE']
    image_float = image.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size ** 2)
    padding = kernel_size // 2
    local_mean = F.conv2d(image_float, kernel, padding=padding)
    local_mean_sq = F.conv2d(image_float ** 2, kernel, padding=padding)
    return torch.clamp(local_mean_sq - local_mean ** 2, min=0).squeeze(0).squeeze(0)
