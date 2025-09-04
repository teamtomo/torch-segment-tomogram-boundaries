import albumentations as A
import cv2
import numpy as np

from torch_tomo_slab import constants
from torch_tomo_slab.data.weight_maps import generate_boundary_weight_map


def scale_to_0_1(img, **kwargs):
    """Scales a numpy array to the [0, 1] range."""
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val > 1e-6:
        img = (img - min_val) / (max_val - min_val)
    return img.astype(np.float32)


class AddBoundaryWeightMap(A.ImageOnlyTransform):

    def __init__(self, high_weight: float = 10.0, base_weight: float = 1.0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.high_weight = high_weight
        self.base_weight = base_weight

    def apply(self, img, **params):
        return generate_boundary_weight_map(img, self.high_weight, self.base_weight)

def get_transforms(is_training: bool = True, use_balanced_crop: bool = True) -> A.Compose:
    """
    Dimension-consistent augmentation pipeline for 256x512 tomography images.
    
    Key improvements:
    - Guaranteed output dimensions via Resize + CenterCrop strategy
    - Removed VerticalFlip (inappropriate for tomography orientation)  
    - Conservative scaling to prevent dimension instability
    - RandomCrop for natural zoom variation
    
    Args:
        is_training: If True, apply stochastic augmentations for regularization
        use_balanced_crop: Ignored (legacy parameter for API compatibility)
    """
    
    if is_training:
        # Dimension-first augmentation strategy for training stability
        transform_list = [
            # === STEP 1: NORMALIZE ===
            A.Lambda(image=scale_to_0_1, name="scale_to_0_1", p=1.0),
            
            # === STEP 2: ESTABLISH CONSISTENT DIMENSIONS ===
            # Resize with buffer then crop to guarantee exact output size
            A.Resize(
                height=constants.AUGMENTATION_CONFIG['RESIZE_BUFFER_HEIGHT'],  # 288
                width=constants.AUGMENTATION_CONFIG['RESIZE_BUFFER_WIDTH'],    # 576
                p=1.0
            ),
            A.CenterCrop(
                height=constants.AUGMENTATION_CONFIG['TARGET_HEIGHT'],         # 256
                width=constants.AUGMENTATION_CONFIG['TARGET_WIDTH'],           # 512
                p=1.0
            ),
            
            # === STEP 3: SAFE SPATIAL AUGMENTATIONS ===
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # REMOVED VerticalFlip - orientation matters in tomography
            
            # Conservative rotation for rectangular images
            A.Rotate(
                limit=constants.AUGMENTATION_CONFIG['ROTATE_LIMIT'],  # 8 degrees
                p=0.5,  # Reduced probability
                border_mode=cv2.BORDER_REFLECT_101,
                interpolation=cv2.INTER_LINEAR
            ),
            
            # === STEP 4: CONTROLLED SCALE VARIATION ===
            # RandomCrop provides natural zoom variation
            A.RandomCrop(
                height=constants.AUGMENTATION_CONFIG['TARGET_HEIGHT'],         # 256
                width=constants.AUGMENTATION_CONFIG['TARGET_WIDTH'],           # 512
                p=0.5
            ),
            
            # Conservative affine scaling (reduced from 0.8-1.2 to 0.95-1.05)
            A.Affine(
                scale=constants.AUGMENTATION_CONFIG['AFFINE_SCALE_RANGE'],      # (0.95, 1.05)
                p=0.5,  # Reduced probability
                border_mode=cv2.BORDER_REFLECT_101
            ),
            
            # === STEP 5: INTENSITY AUGMENTATIONS ===
            A.RandomBrightnessContrast(
                brightness_limit=constants.AUGMENTATION_CONFIG['BRIGHTNESS_CONTRAST_LIMIT'],  # 0.1
                contrast_limit=constants.AUGMENTATION_CONFIG['BRIGHTNESS_CONTRAST_LIMIT'],    # 0.1
                p=0.5  # Reduced probability
            ),
            
            # Gamma correction variation
            A.RandomGamma(
                gamma_limit=constants.AUGMENTATION_CONFIG['GAMMA_LIMIT'],  # (90, 110)
                p=0.5
            ),
            
            # === STEP 6: NOISE AND BLUR ===
            #A.GaussNoise(
            #    var_limit=constants.AUGMENTATION_CONFIG['NOISE_VAR_LIMIT'],  # (5, 15)
            #    p=0.25
            #),
            
            A.GaussianBlur(
                blur_limit=constants.AUGMENTATION_CONFIG['BLUR_LIMIT'],  # 3
                p=0.4
            ),
            
            # === STEP 7: OCCLUSION/DROPOUT (KEEP PROVEN EFFECTIVE) ===
            # Coarse dropout - larger holes to force learning robust features
            #A.CoarseDropout(
            #    num_holes_range=constants.AUGMENTATION_CONFIG['COARSE_DROPOUT_HOLES'],  # (3, 8)
            #    hole_height_range=constants.AUGMENTATION_CONFIG['COARSE_DROPOUT_SIZE'], # (0.03, 0.08)
            #    hole_width_range=constants.AUGMENTATION_CONFIG['COARSE_DROPOUT_SIZE'],  # (0.03, 0.08)
            #    fill=0,
            #    fill_mask=None,  # Don't fill mask holes
            #    p=0.4
            #),
            
            # Grid dropout - systematic occlusion patterns  
            #A.GridDropout(
            #    ratio=constants.AUGMENTATION_CONFIG['GRID_DROPOUT_RATIO'],  # 0.3
            #    p=0.3
            #),
        ]
    else:
        # Validation: Apply same dimensional preprocessing for consistency
        transform_list = [
            A.Lambda(image=scale_to_0_1, name="scale_to_0_1", p=1.0),
            A.Resize(
                height=constants.AUGMENTATION_CONFIG['RESIZE_BUFFER_HEIGHT'],  # 288
                width=constants.AUGMENTATION_CONFIG['RESIZE_BUFFER_WIDTH'],    # 576
                p=1.0
            ),
            A.CenterCrop(
                height=constants.AUGMENTATION_CONFIG['TARGET_HEIGHT'],         # 256
                width=constants.AUGMENTATION_CONFIG['TARGET_WIDTH'],           # 512
                p=1.0
            ),
        ]
    
    return A.Compose(transform_list)
