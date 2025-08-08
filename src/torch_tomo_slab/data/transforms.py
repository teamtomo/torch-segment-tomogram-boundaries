# src/torch_tomo_slab/data/transforms.py

from typing import Dict

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

# Define the final training size in one place for consistency.
FINAL_CROP_SIZE = 512

def get_transforms(is_training: bool = True) -> A.Compose:
    """
    Defines and returns the augmentation pipeline using Albumentations.
    This implements the robust "Pad -> Augment -> Crop" strategy.
    """
    if is_training:
        # --- ROBUST TRAINING PIPELINE ---
        transform_list = [
            # 1. PAD THE IMAGE TO A LARGE SQUARE (1024x1024)
            A.PadIfNeeded(
                min_height=1024,
                min_width=1024,
                # CORE FIX: Use the integer value 0 for BORDER_CONSTANT.
                border_mode=0,
                value=0,
                mask_value=0,
                p=1.0
            ),

            # 2. AUGMENT FREELY ON THE SQUARE CANVAS
            A.Rotate(
                limit=90,
                p=0.7,
                # CORE FIX: Use the integer value 0 for BORDER_CONSTANT.
                border_mode=0,
                value=0,
                mask_value=0
            ),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),

            # 3. CROP A FIXED-SIZE SQUARE FOR TRAINING
            A.RandomCrop(height=FINAL_CROP_SIZE, width=FINAL_CROP_SIZE, p=1.0),

            # 4. CONVERT TO TENSOR
            ToTensorV2(),
        ]
    else:
        # --- VALIDATION PIPELINE ---
        transform_list = [
            A.CenterCrop(height=512, width=512, p=1.0),
            A.Resize(height=FINAL_CROP_SIZE, width=FINAL_CROP_SIZE, p=1.0),
            ToTensorV2(),
        ]
    
    return A.Compose(transform_list)
