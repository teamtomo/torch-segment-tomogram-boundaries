"""3D volume processing utilities for tomographic data.

This module provides utilities for 3D volume manipulation including
resizing, padding, and interpolation operations used in the tomographic
processing pipeline.
"""

import torch
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
