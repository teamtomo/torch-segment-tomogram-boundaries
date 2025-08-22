"""Simple sampling utilities for tomographic segmentation.

This module provides basic sampling utilities that may be used for 
future sampling strategies if needed.
"""
from typing import List
import numpy as np
import torch
from torch.utils.data import Sampler


# Placeholder for future sampling strategies if needed
def compute_fill_ratios(pt_file_paths: List, threshold: float = 0.5) -> np.ndarray:
    """
    Compute fill ratios (percentage of 1s) for a list of .pt files.
    
    Parameters
    ----------
    pt_file_paths : List
        List of paths to .pt files containing image/label pairs
    threshold : float
        Threshold for binarizing labels (default: 0.5)
    
    Returns
    -------
    np.ndarray
        Array of fill ratios, one per .pt file
    """
    ratios = []
    
    for pt_path in pt_file_paths:
        data = torch.load(pt_path, map_location='cpu')
        label_np = data['label'].numpy().squeeze()
        
        # Calculate fill ratio
        binary_mask = (label_np > threshold).astype(np.float32)
        fill_ratio = binary_mask.mean()
        ratios.append(fill_ratio)
    
    return np.array(ratios)