# src/torch_tomo_slab/utils/threeD.py
import torch
import torch.nn.functional as F


def resize_and_pad_3d(tensor: torch.Tensor, target_shape: tuple, mode: str) -> torch.Tensor:
    """
    Resizes and pads a 3D tensor to a target shape.

    Args:
        tensor: The input 3D tensor.
        target_shape: A tuple (D, H, W) for the target shape.
        mode: 'image' for continuous data (uses trilinear) or 'label' for discrete data (uses nearest).

    Returns:
        The resized and padded tensor.
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