from typing import Dict, Tuple, List
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from torch_tomo_slab import config

class JointTransform:
    """
    Applies the same random geometric transform to an image and its label map.
    Operates on a dictionary of tensors.
    """

    def __init__(self):
        # These are just for storing parameters, the logic is custom.
        self.affine_params = T.RandomAffine.get_params(
            degrees=(-10, 10),
            translate=(0.1, 0.1),
            scale_ranges=(0.9, 1.1),
            shears=None,
            img_size=[1, 1]  # Placeholder, will be replaced
        )

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image, label = sample['image'], sample['label']

        # 1. Random Flips (50% chance for each)
        if torch.rand(1) < 0.5:
            image = F.hflip(image)
            label = F.hflip(label)
        if torch.rand(1) < 0.5:
            image = F.vflip(image)
            label = F.vflip(label)

        # 2. Random Affine (applied together)
        affine_params = T.RandomAffine.get_params(
            degrees=(-15, 15),
            translate=(0.1, 0.1),
            scale_ranges=(0.9, 1.1),
            shears=None,
            img_size=list(image.shape[-2:])
        )
        image = F.affine(image, *affine_params, interpolation=InterpolationMode.BILINEAR)
        # Use NEAREST for the label to preserve integer class values
        label = F.affine(label, *affine_params, interpolation=InterpolationMode.NEAREST)

        # 3. Intensity transforms (applied only to image)
        # Using ColorJitter on a 2-channel scientific image is possible but can have
        # unexpected effects. Let's use more direct methods.
        # Add random noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(image) * 0.05
            image = image + noise

        # Random blur
        if torch.rand(1) < 0.3:
            image = F.gaussian_blur(image, kernel_size=3, sigma=(0.1, 1.0))

        sample['image'] = image
        sample['label'] = label
        return sample



def get_transforms(is_training: bool = True) -> T.Compose:
    """
    Create a torchvision transform pipeline for 2D medical image segmentation.
    With pre-normalization, this pipeline now only contains augmentations.
    """
    transform_list: List[callable] = []

    if is_training:
        transform_list.append(JointTransform())

    # --- REMOVED: Normalization is no longer applied here. ---
    # transform_list.append(NormalizeSample())

    # If no transforms are needed (i.e., validation), return None or an empty Compose.
    if not transform_list:
        return None
        
    return T.Compose(transform_list)
