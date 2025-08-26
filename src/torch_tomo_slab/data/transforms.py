import albumentations as A
import cv2

from torch_tomo_slab import constants
from torch_tomo_slab.data.weight_maps import generate_boundary_weight_map


class AddBoundaryWeightMap(A.ImageOnlyTransform):

    def __init__(self, high_weight: float = 10.0, base_weight: float = 1.0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.high_weight = high_weight
        self.base_weight = base_weight

    def apply(self, img, **params):
        return generate_boundary_weight_map(img, self.high_weight, self.base_weight)

def get_transforms(is_training: bool = True, use_balanced_crop: bool = True) -> A.Compose:
    """
    Clean augmentation pipeline - no padding needed, no CLAHE for multi-channel.
    
    Args:
        is_training: If True, apply stochastic augmentations for regularization
        use_balanced_crop: Ignored (legacy parameter for API compatibility)
    """
    
    if is_training:
        # Enhanced augmentations to combat overfitting without cropping
        transform_list = [
            # === SPATIAL AUGMENTATIONS ===
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Conservative rotation for rectangular images
            A.Rotate(
                limit=constants.AUGMENTATION_CONFIG['ROTATE_LIMIT'],  # 10 degrees
                p=0.4,
                border_mode=cv2.BORDER_REFLECT_101,
                interpolation=cv2.INTER_LINEAR
            ),
            
            # Elastic deformation for shape variation
            A.ElasticTransform(
                alpha=constants.AUGMENTATION_CONFIG['ELASTIC_ALPHA'],    # 50
                sigma=constants.AUGMENTATION_CONFIG['ELASTIC_SIGMA'],    # 5
                p=0.4
            ),
            
            # Grid distortion for local geometric variation
            A.GridDistortion(
                num_steps=5,
                distort_limit=constants.AUGMENTATION_CONFIG['GRID_DISTORTION_LIMIT'],  # 0.2
                p=0.3,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            
            # === INTENSITY AUGMENTATIONS ===
            A.RandomBrightnessContrast(
                brightness_limit=constants.AUGMENTATION_CONFIG['BRIGHTNESS_CONTRAST_LIMIT'],  # 0.2
                contrast_limit=constants.AUGMENTATION_CONFIG['BRIGHTNESS_CONTRAST_LIMIT'],    # 0.2
                p=0.5
            ),
            
            # Gamma correction variation
            A.RandomGamma(
                gamma_limit=constants.AUGMENTATION_CONFIG['GAMMA_LIMIT'],  # (80, 120)
                p=0.3
            ),
            
            # === NOISE AND BLUR ===
            A.GaussNoise(
                var_limit=constants.AUGMENTATION_CONFIG['NOISE_VAR_LIMIT'],  # (10, 30)
                p=0.25
            ),
            
            A.GaussianBlur(
                blur_limit=constants.AUGMENTATION_CONFIG['BLUR_LIMIT'],  # 3
                p=0.2
            ),
            
            # === OCCLUSION/DROPOUT ===
            # Coarse dropout - larger holes to force learning robust features
            A.CoarseDropout(
                num_holes_range=constants.AUGMENTATION_CONFIG['COARSE_DROPOUT_HOLES'],  # (3, 8)
                hole_height_range=constants.AUGMENTATION_CONFIG['COARSE_DROPOUT_SIZE'], # (0.03, 0.08)
                hole_width_range=constants.AUGMENTATION_CONFIG['COARSE_DROPOUT_SIZE'],  # (0.03, 0.08)
                fill=0,
                fill_mask=None,  # Don't fill mask holes
                p=0.4
            ),
            
            # Grid dropout - systematic occlusion patterns  
            A.GridDropout(
                ratio=constants.AUGMENTATION_CONFIG['GRID_DROPOUT_RATIO'],  # 0.3
                p=0.3
            ),
        ]
    else:
        # Validation: No augmentations - clean data only
        transform_list = []
    
    return A.Compose(transform_list)
