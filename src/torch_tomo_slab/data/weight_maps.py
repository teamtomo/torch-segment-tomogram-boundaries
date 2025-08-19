# src/torch_tomo_slab/data/weight_maps.py

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt as distance_transform

def generate_boundary_weight_map(mask: np.ndarray, high_weight: float = 10.0, base_weight: float = 1.0) -> np.ndarray:
    """
    Generates a weight map where pixels on the boundary of objects have a higher weight.
    Args:
        mask: The ground truth mask, assumed to be a 2D numpy array with integer values.
        high_weight: The weight to assign to boundary pixels.
        base_weight: The base weight for all other pixels.
    Returns:
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
    
    return weight_map
