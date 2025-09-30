import numpy as np
from scipy.ndimage import binary_erosion


def generate_boundary_weight_map(mask: np.ndarray, high_weight: float = 2.5, base_weight: float = 1.0) -> np.ndarray:
    """
    Generates a weight map where pixels on the boundary of objects have a higher weight
    Args:
        mask: The ground truth mask, assumed to be a 2D numpy array with integer values.
        high_weight: The weight to assign to boundary pixels.
        base_weight: The base weight for all other pixels.

    Returns
    -------
        A 2D numpy array (weight map) of the same size as the mask.
    """
    # Ensure mask is binary (0 or 1)
    binary_mask = (mask > 0).astype(np.uint8)

    # Erode the mask to find the inner boundary
    eroded_mask = binary_erosion(binary_mask, iterations=1)

    # The boundary is the difference between the original mask and the eroded one
    boundary = binary_mask - eroded_mask

    # Create the weight map
    weight_map = np.full(mask.shape, base_weight, dtype=np.float32)
    weight_map[boundary == 1] = high_weight

    # Normalize so the mean stays close to 1 and gradients remain stable
    mean_val = weight_map.mean()
    if mean_val > 0:
        weight_map = weight_map / mean_val

    return weight_map
