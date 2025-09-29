import albumentations as A
import cv2
import numpy as np

from torch_tomo_slab import constants
from numpy.fft import fft2, ifft2, fftshift, ifftshift


class MissingWedge(A.ImageOnlyTransform):
    """
    Applies a missing wedge artifact to a 2D image in Fourier space.

    This augmentation simulates the data loss seen in tomography. It does so by:
    1. Transforming the image to Fourier space (frequency domain).
    2. Creating a wedge-shaped mask at a random orientation.
    3. Setting the frequency components within the wedge to zero.
    4. Transforming the image back to the spatial domain.

    Args:
        angle_range (tuple): A (min, max) tuple for the missing wedge angle in degrees.
                             A larger angle means more data is removed.
        p (float): The probability of applying the transform.
    """

    def __init__(self, angle_range=(40.0, 60.0), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.angle_range = angle_range

    def apply(self, img, **params):
        # 1. Randomly determine the parameters for this specific image
        wedge_angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        # Randomize the orientation of the wedge from 0 to 180 degrees
        rotation_angle = np.random.uniform(0, 180)

        # 2. Perform the forward Fourier Transform
        fft_img = fftshift(fft2(img))
        h, w = img.shape

        # 3. Create a coordinate grid in Fourier space
        ch, cw = h / 2, w / 2
        y_coords, x_coords = np.indices(img.shape)
        x_coords = x_coords - cw
        y_coords = y_coords - ch

        # 4. Rotate the coordinate system to orient the wedge
        rotation_rad = np.deg2rad(rotation_angle)
        cos_theta = np.cos(rotation_rad)
        sin_theta = np.sin(rotation_rad)
        x_rotated = x_coords * cos_theta + y_coords * sin_theta

        # 5. Create the wedge mask using the rotated coordinates
        # The wedge is defined by an angle around the y-axis in the rotated frame
        wedge_angle_rad = np.deg2rad(wedge_angle)
        mask = np.abs(np.arctan2(x_rotated, y_coords * sin_theta - x_coords * cos_theta)) > (wedge_angle_rad / 2)

        # 6. Apply the mask to erase data in the frequency domain
        fft_img[~mask] = 0

        # 7. Perform the inverse Fourier Transform
        img_reconstructed = ifft2(ifftshift(fft_img))

        # Return the real part, ensuring it has the same data type
        return np.real(img_reconstructed).astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("angle_range",)


def scale_to_0_1(img, **kwargs):
    """Scales a numpy array to the [0, 1] range."""
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val > 1e-6:
        img = (img - min_val) / (max_val - min_val)
    return img.astype(np.float32)

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
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Conservative rotation for rectangular images
            A.Rotate(
                limit=constants.AUGMENTATION_CONFIG['ROTATE_LIMIT'],
                p=0.5,
                border_mode=cv2.BORDER_REFLECT_101,
                interpolation=cv2.INTER_LINEAR
            ),
            A.RandomCrop(
                height=constants.AUGMENTATION_CONFIG['TARGET_HEIGHT'],
                width=constants.AUGMENTATION_CONFIG['TARGET_WIDTH'],
                p=0.5
            ),
            A.Affine(
                scale=constants.AUGMENTATION_CONFIG['AFFINE_SCALE_RANGE'],
                p=0.5,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.RandomBrightnessContrast(
                brightness_limit=constants.AUGMENTATION_CONFIG['BRIGHTNESS_CONTRAST_LIMIT'],
                contrast_limit=constants.AUGMENTATION_CONFIG['BRIGHTNESS_CONTRAST_LIMIT'],
                p=0.5  # Reduced probability
            ),

            A.RandomGamma(
                gamma_limit=constants.AUGMENTATION_CONFIG['GAMMA_LIMIT'],  # (90, 110)
                p=0.5
            ),

            MissingWedge(angle_range=(30, 70.0), p=0.5),
            

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
