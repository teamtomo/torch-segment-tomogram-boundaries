"""3D volume processing utilities for tomographic data.

This module provides utilities for 3D volume manipulation including
resizing, padding, and interpolation operations used in the tomographic
processing pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_and_pad_3d(tensor: torch.Tensor, target_shape: tuple, mode: str) -> torch.Tensor:
    """
    Resize and pad 3D tensor to target shape with appropriate interpolation.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 3D tensor of shape (D, H, W).
    target_shape : tuple
        Target shape as (depth, height, width).
    mode : str
        Interpolation mode - 'image' for trilinear (continuous data) or 
        'label' for nearest (discrete/label data).

    Returns
    -------
    torch.Tensor
        Resized and padded tensor with exact target_shape.
        Padding uses median value for 'image' mode, 0 for 'label' mode.
    """
    is_label = (mode == 'label')
    input_dtype = tensor.dtype
    tensor = tensor.float()

    tensor_5d = tensor.unsqueeze(0).unsqueeze(0)
    resized_5d = F.interpolate(
        tensor_5d,
        size=target_shape,
        mode='trilinear' if not is_label else 'nearest',
        align_corners=False if not is_label else None
    )
    resized_tensor = resized_5d.squeeze(0).squeeze(0)

    if is_label:
        resized_tensor = resized_tensor.to(input_dtype)

    shape = resized_tensor.shape
    discrepancy = [max(0, ts - s) for ts, s in zip(target_shape, shape)]
    if not any(d > 0 for d in discrepancy):
        return resized_tensor

    padding = []
    for d in reversed(discrepancy):
        pad_1, pad_2 = d // 2, d - (d // 2)
        padding.extend([pad_1, pad_2])

    padding_value = torch.median(tensor).item() if not is_label else 0
    return F.pad(resized_tensor, tuple(padding), mode='constant', value=padding_value)


def gpu_gaussian_blur_3d(tensor: torch.Tensor, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Apply 3D Gaussian blur using GPU-accelerated convolution.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 3D tensor to blur.
    sigma : float
        Standard deviation for Gaussian kernel.
    device : torch.device
        PyTorch device for computation.

    Returns
    -------
    torch.Tensor
        Blurred tensor of same shape as input.
    """
    kernel_size = 2 * math.ceil(3.5 * sigma) + 1
    coords = torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype) - kernel_size // 2
    g = coords.pow(2); kernel_1d = torch.exp(-g / (2 * sigma**2))
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    kernel_3d /= kernel_3d.sum()
    kernel_5d = kernel_3d.unsqueeze(0).unsqueeze(0)
    conv = nn.Conv3d(1, 1, kernel_size, padding='same', bias=False).to(device)
    conv.weight.data = kernel_5d
    return conv(tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def apply_slab_blending(volume_3d: torch.Tensor, slab_size: int, device: torch.device, axis: str = "H", show_progress: bool = True) -> torch.Tensor:
    """
    Apply optimized slab blending for temporal consistency.

    Parameters
    ----------
    volume_3d : torch.Tensor
        Input prediction volume tensor of shape (D, H, W).
    slab_size : int
        Size of temporal slab for blending adjacent slices.
    device : torch.device
        PyTorch device for computation.
    axis : str, default="H"
        Axis name for logging.
    show_progress : bool, default=True
        Whether to show progress bar.

    Returns
    -------
    torch.Tensor
        Blended prediction volume of same shape as input.
    """
    if slab_size <= 1:
        return volume_3d
        
    D, H, W = volume_3d.shape
    hann_window = torch.hann_window(slab_size, periodic=False, device=device)
    final_slices = []
    
    iterator = range(H)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"Slab Blending ({axis} axis)", leave=False, ncols=80)
    
    for i in iterator:
        half_slab = slab_size // 2
        start = max(0, i - half_slab)
        end = min(H, i + half_slab + 1)
        
        # Extract slab around current slice
        slab = volume_3d[:, start:end, :]  # Shape: (D, slab_length, W)
        
        # Calculate window indices for current position
        win_start = max(0, half_slab - i)
        win_end = min(slab_size, half_slab + (H - i))
        current_window = hann_window[win_start:win_end]
        
        # Apply weighted averaging along the temporal dimension
        if current_window.sum() > 0:
            weighted_slice = torch.einsum('dwh,d->wh', slab.permute(1, 0, 2), current_window) / current_window.sum()
        else:
            weighted_slice = volume_3d[:, i, :]  # Fallback to original slice
        
        final_slices.append(weighted_slice)
    
    return torch.stack(final_slices, dim=1)
