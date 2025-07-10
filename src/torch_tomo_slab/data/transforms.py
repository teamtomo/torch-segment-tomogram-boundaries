from typing import Tuple
from torch_tomo_slab import config
import torchio as tio

def get_transforms(patch_size: Tuple[int, int],
                   is_training: bool = True) -> tio.Transform:
    """
    Create TorchIO transform pipeline for 2D medical image segmentation.
    """

    if is_training:
        transforms = tio.Compose([
            # Spatial transforms (applied to both image and label)
            tio.RandomFlip(axes=('Left', 'Posterior'), p=0.5),  # Horizontal/Vertical flip
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=(-10, 10),
                translation=(-0.1, 0.1),
                p=0.5
            ),

            # Intensity transforms (only applied to image, not label)
            tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
            tio.RandomNoise(std=(0, 0.05), p=0.3),
            tio.RandomBlur(std=(0, 1), p=0.3),
            # Patch sampling for training
            tio.CropOrPad((*patch_size, 1)),  # Ensure consistent size
        ])
    else:
        # Validation/test: minimal transforms
        transforms = tio.Compose([
            tio.ZNormalization(masking_method=None),
            tio.CropOrPad((*patch_size, 1)),
        ])

    return transforms