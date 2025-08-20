# src/torch_tomo_slab/utils/twoD.py
import torch
import torch.nn.functional as F

from .. import config

def robust_normalization(data: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a tensor by its median and the 5th-95th percentile range.
    """
    data = data.float()
    p5, p95 = torch.quantile(data, 0.05), torch.quantile(data, 0.95)
    if p95 - p5 < 1e-5: return data - torch.median(data)
    return (data - torch.median(data)) / (p95 - p5)

def local_variance_2d(image: torch.Tensor) -> torch.Tensor:
    """
    Calculates the local variance of a 2D image tensor.
    """
    kernel_size = config.AUGMENTATION_CONFIG['LOCAL_VARIANCE_KERNEL_SIZE']
    image_float = image.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size ** 2)
    padding = kernel_size // 2
    local_mean = F.conv2d(image_float, kernel, padding=padding)
    local_mean_sq = F.conv2d(image_float ** 2, kernel, padding=padding)
    return torch.clamp(local_mean_sq - local_mean ** 2, min=0).squeeze(0).squeeze(0)