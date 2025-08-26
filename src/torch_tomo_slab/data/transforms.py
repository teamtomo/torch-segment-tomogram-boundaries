from typing import Any, Dict

import albumentations as A
import cv2
import numpy as np

from torch_tomo_slab import constants
from torch_tomo_slab.data.weight_maps import generate_boundary_weight_map


class AddBoundaryWeightMap(A.core.transforms_interface.ImageOnlyTransform):

    def __init__(self, high_weight: float = 10.0, base_weight: float = 1.0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.high_weight = high_weight
        self.base_weight = base_weight

    def apply(self, img, **params) -> Dict[str, Any]:
        return generate_boundary_weight_map(img, self.high_weight, self.base_weight)


class BalancedCrop(A.DualTransform):
    """
    Simple crop that avoids pure empty (0) or pure filled (1) patches.
    
    This transform ensures patches have a minimum percentage of both 0s and 1s,
    preventing the network from training on completely empty or completely filled patches.
    
    Parameters
    ----------
    height : int
        Height of the crop
    width : int
        Width of the crop  
    min_fill_ratio : float
        Minimum percentage of 1s required (0.1 = 10%)
    max_fill_ratio : float  
        Maximum percentage of 1s allowed (0.9 = 90%)
    max_attempts : int
        Maximum attempts to find a balanced crop before falling back to random
    """
    
    def __init__(self, height: int, width: int, min_fill_ratio: float = 0.1, 
                 max_fill_ratio: float = 0.9, max_attempts: int = 50, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.min_fill_ratio = min_fill_ratio
        self.max_fill_ratio = max_fill_ratio
        self.max_attempts = max_attempts
    
    def apply(self, img, crop_coords, **params):
        """Apply crop to image."""
        x1, y1, x2, y2 = crop_coords
        return img[y1:y2, x1:x2]
    
    def apply_to_mask(self, mask, crop_coords, **params):
        """Apply crop to mask."""
        x1, y1, x2, y2 = crop_coords
        return mask[y1:y2, x1:x2]
    
    def get_params_dependent_on_targets(self, params):
        """
        Generate crop coordinates that avoid pure 0 or pure 1 patches.
        """
        mask = params['mask']
        h, w = mask.shape[:2]
        
        # Ensure we can make a crop
        if w < self.width or h < self.height:
            return {'crop_coords': (0, 0, min(w, self.width), min(h, self.height))}
        
        max_x = w - self.width
        max_y = h - self.height
        
        # Try to find a balanced crop (reduced attempts to prevent OOM)
        for attempt in range(self.max_attempts):
            # Random crop location
            x1 = np.random.randint(0, max_x + 1)
            y1 = np.random.randint(0, max_y + 1)
            x2 = x1 + self.width
            y2 = y1 + self.height
            
            # Check fill ratio of this crop (more efficient calculation)
            crop_mask = mask[y1:y2, x1:x2]
            fill_ratio = np.mean(crop_mask, dtype=np.float32)
            
            # Accept if within desired range
            if self.min_fill_ratio <= fill_ratio <= self.max_fill_ratio:
                return {'crop_coords': (x1, y1, x2, y2)}
        
        # Fall back to random crop if no balanced crop found
        x1 = np.random.randint(0, max_x + 1)
        y1 = np.random.randint(0, max_y + 1)
        x2 = x1 + self.width
        y2 = y1 + self.height
        
        return {'crop_coords': (x1, y1, x2, y2)}
    
    @property
    def targets_as_params(self):
        return ["image", "mask"]


def get_transforms(is_training: bool = True, use_balanced_crop: bool = True) -> A.Compose:
    if is_training:
        transform_list = [
            # Rectangular padding with edge-preserving border
            A.PadIfNeeded(
                min_height=constants.AUGMENTATION_CONFIG['PAD_HEIGHT'],
                min_width=constants.AUGMENTATION_CONFIG['PAD_WIDTH'],
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ),

            # Geometric transforms that preserve edges (NO TRANSPOSE)
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT_101),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            # Intensity transforms (gentle)
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),

            # EDGE-PRESERVING CoarseDropout with inpainting
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.06, 0.12),
                hole_width_range=(0.06, 0.12),
                fill='inpaint_telea',
                fill_mask=None,
                p=0.4
            ),

            # ADD: Final crop to ensure consistent dimensions
            BalancedCrop(
                height=constants.AUGMENTATION_CONFIG['CROP_HEIGHT'],  # e.g., 512
                width=constants.AUGMENTATION_CONFIG['CROP_WIDTH'],  # e.g., 512
                min_fill_ratio=0.05,
                max_fill_ratio=0.95,
                max_attempts=20,
                p=1.0
            )
        ]
    else:
        transform_list = [
            A.PadIfNeeded(
                min_height=constants.AUGMENTATION_CONFIG['PAD_HEIGHT'],
                min_width=constants.AUGMENTATION_CONFIG['PAD_WIDTH'],
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ),
            # ADD: Consistent crop for validation too
            A.CenterCrop(
                height=constants.AUGMENTATION_CONFIG['CROP_HEIGHT'],
                width=constants.AUGMENTATION_CONFIG['CROP_WIDTH'],
                p=1.0
            )
        ]

    return A.Compose(transform_list)

            # Balanced rectangular crop
            BalancedCrop(
                height=constants.AUGMENTATION_CONFIG['CROP_HEIGHT'],
                width=constants.AUGMENTATION_CONFIG['CROP_WIDTH'],
                min_fill_ratio=0.05,
                max_fill_ratio=0.95,
                max_attempts=20,
                p=1.0
            )
        ]
    else:
        transform_list = [
            A.PadIfNeeded(
                min_height=constants.AUGMENTATION_CONFIG['PAD_HEIGHT'],
                min_width=constants.AUGMENTATION_CONFIG['PAD_WIDTH'],
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            )
        ]

    return A.Compose(transform_list)
