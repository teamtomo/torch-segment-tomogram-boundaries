from typing import Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

# This wrapper class makes the albumentations pipeline seamlessly compatible 
# with your existing PyTorch Lightning DataModule and Dataset.
class AlbumentationsWrapper:
    """
    A wrapper to apply Albumentations transformations to the sample dictionary
    {'image': tensor, 'label': tensor} produced by your PTFileDataset.
    """
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Albumentations works with NumPy arrays and specific dictionary keys.
        # We convert the image tensor from (C, H, W) to a NumPy array (H, W, C).
        image_np = sample['image'].numpy().transpose(1, 2, 0)
        
        # We convert the label tensor from (1, H, W) to a NumPy array (H, W).
        mask_np = sample['label'].numpy().squeeze()

        # Apply the transformations. Albumentations requires the keys 'image' and 'mask'.
        transformed = self.transforms(image=image_np, mask=mask_np)

        # The ToTensorV2() transform at the end of the pipeline handles the conversion
        # back to PyTorch Tensors and moves the channel dimension to the front for the image.
        transformed_image = transformed['image']
        
        # The transformed mask will be a (H, W) tensor, so we add the channel dimension
        # back to match the expected shape of (1, H, W).
        transformed_mask = transformed['mask'].unsqueeze(0)

        # Update the sample dictionary with the augmented data.
        sample['image'] = transformed_image
        sample['label'] = transformed_mask
        
        return sample

def get_transforms(is_training: bool = True) -> AlbumentationsWrapper:
    """
    Returns a composition of transformations for data augmentation using Albumentations.
    
    Args:
        is_training (bool): If True, returns a robust set of augmentations for training.
                            If False, returns a minimal pipeline for validation/testing.
    """
    if is_training:
        # Define a rich set of augmentations for the training set to combat overfitting.
        # Each transform is applied with a given probability `p`.
        transform_list = A.Compose([
            # --- Geometric Transforms ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.7,
                border_mode=0 # Pad with black
            ),
            
            # --- Pixel-level Transforms ---
            # Adjust brightness and contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            # Add Gaussian noise
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),

            # --- Blur Transforms ---
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.GaussianBlur(p=0.8),
            ], p=0.5), # Apply one of the blur types with 50% probability

            # This must be the last transform in the pipeline. It converts the
            # NumPy arrays to PyTorch Tensors and handles channel ordering.
            ToTensorV2(),
        ])
    else:
        # For validation, we only need to convert the data to a tensor.
        # No random augmentations should be applied to get a consistent evaluation.
        transform_list = A.Compose([
            ToTensorV2(),
        ])
    
    # Wrap the Albumentations pipeline to make it compatible with your dataloader
    return AlbumentationsWrapper(transform_list)
