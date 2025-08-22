from typing import Any, Dict

import albumentations as A

from torch_tomo_slab import constants, config
from torch_tomo_slab.data.weight_maps import generate_boundary_weight_map


class AddBoundaryWeightMap(A.core.transforms_interface.ImageOnlyTransform):

    def __init__(self, high_weight: float = 10.0, base_weight: float = 1.0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.high_weight = high_weight
        self.base_weight = base_weight

    def apply(self, img, **params) -> Dict[str, Any]:
        return generate_boundary_weight_map(img, self.high_weight, self.base_weight)

def get_transforms(is_training: bool = True) -> A.Compose:
    if is_training:
        transform_list = [
            A.PadIfNeeded(min_height=constants.AUGMENTATION_CONFIG['PAD_SIZE'], min_width=constants.AUGMENTATION_CONFIG['PAD_SIZE'], border_mode=0, fill=0, fill_mask=0, p=1.0),
            A.Rotate(limit=constants.AUGMENTATION_CONFIG['ROTATE_LIMIT'], p=0.7, border_mode=0, fill=0, fill_mask=0),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=constants.AUGMENTATION_CONFIG['BRIGHTNESS_CONTRAST_LIMIT'], contrast_limit=constants.AUGMENTATION_CONFIG['BRIGHTNESS_CONTRAST_LIMIT'], p=0.4),
            A.GaussNoise(std_range=constants.AUGMENTATION_CONFIG['GAUSS_NOISE_STD_RANGE'], p=0.3),
            A.GaussianBlur(blur_limit=constants.AUGMENTATION_CONFIG['GAUSS_BLUR_LIMIT'], p=0.3),
#            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.3),
            A.CoarseDropout(p=0.3, num_holes_range=(1,8), hole_height_range=(0.1,0.2), hole_width_range=(0.1,0.2), fill=0),
            A.RandomCrop(height=constants.AUGMENTATION_CONFIG['CROP_SIZE'], width=constants.AUGMENTATION_CONFIG['CROP_SIZE'], p=1.0),
        ]
    else:
        transform_list = [
            A.CenterCrop(height=constants.AUGMENTATION_CONFIG['CROP_SIZE'], width=constants.AUGMENTATION_CONFIG['CROP_SIZE'], p=1.0),
            A.Resize(height=constants.AUGMENTATION_CONFIG['CROP_SIZE'], width=constants.AUGMENTATION_CONFIG['CROP_SIZE'], p=1.0),
        ]
    return A.Compose(transform_list)
