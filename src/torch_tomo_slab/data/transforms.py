from typing import Dict
import albumentations as A
from .weight_maps import generate_boundary_weight_map
from .. import constants

class AddBoundaryWeightMap(A.core.transforms_interface.ImageOnlyTransform):

    def __init__(self, high_weight: float = 10.0, base_weight: float = 1.0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.high_weight = high_weight
        self.base_weight = base_weight

    def apply(self, img, **params) -> Dict[str, any]:
        return generate_boundary_weight_map(img, self.high_weight, self.base_weight)

def get_transforms(is_training: bool = True) -> A.Compose:
    if is_training:
        transform_list = [
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, fill=0, fill_mask=0, p=1.0),
            A.Rotate(limit=90, p=0.7, border_mode=0, fill=0, fill_mask=0),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(std_range=(0.0, 0.05), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomCrop(height=constants.FINAL_CROP_SIZE, width=constants.FINAL_CROP_SIZE, p=1.0),
        ]
    else:
        transform_list = [
            A.CenterCrop(height=constants.FINAL_CROP_SIZE, width=constants.FINAL_CROP_SIZE, p=1.0),
            A.Resize(height=constants.FINAL_CROP_SIZE, width=constants.FINAL_CROP_SIZE, p=1.0),
        ]
    return A.Compose(transform_list)
