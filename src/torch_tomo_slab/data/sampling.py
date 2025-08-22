"""Boundary-aware sampling utilities for tomographic segmentation.

This module provides intelligent sampling strategies that address the class imbalance
problem by oversampling boundary-rich patches and implementing content-aware cropping.
"""
from typing import List, Tuple
import numpy as np
from scipy.ndimage import binary_erosion
import torch
from torch.utils.data import Sampler


def calculate_boundary_score(mask: np.ndarray, crop_size: int = 64, stride: int = 32) -> List[Tuple[int, int, float]]:
    """
    Calculate boundary scores for all possible crop locations in a mask.
    
    Parameters
    ----------
    mask : np.ndarray
        2D binary mask array
    crop_size : int
        Size of the square crop
    stride : int
        Stride between crop centers
    
    Returns
    -------
    List[Tuple[int, int, float]]
        List of (row, col, boundary_score) tuples for each valid crop location
    """
    h, w = mask.shape
    scores = []
    
    # Generate all possible crop centers
    for i in range(crop_size // 2, h - crop_size // 2, stride):
        for j in range(crop_size // 2, w - crop_size // 2, stride):
            # Extract crop region
            crop_mask = mask[i - crop_size // 2:i + crop_size // 2,
                            j - crop_size // 2:j + crop_size // 2]
            
            if crop_mask.shape != (crop_size, crop_size):
                continue
                
            # Calculate boundary score for this crop
            score = _compute_boundary_density(crop_mask)
            scores.append((i, j, score))
    
    return scores


def _compute_boundary_density(crop_mask: np.ndarray) -> float:
    """
    Compute boundary density score for a crop.
    
    Parameters
    ----------
    crop_mask : np.ndarray
        2D binary crop of the mask
    
    Returns
    -------
    float
        Boundary density score (0-1, higher means more boundary content)
    """
    if crop_mask.sum() == 0 or crop_mask.sum() == crop_mask.size:
        # All empty or all filled - no boundaries
        return 0.0
    
    # Convert to binary
    binary_mask = (crop_mask > 0).astype(np.uint8)
    
    # Find boundaries using erosion
    eroded = binary_erosion(binary_mask, iterations=1)
    boundary = binary_mask - eroded
    
    # Calculate boundary density
    boundary_pixels = boundary.sum()
    total_pixels = crop_mask.size
    
    # Also consider mask fill ratio - patches with ~50% fill tend to have more boundaries
    fill_ratio = binary_mask.mean()
    fill_penalty = 1.0 - abs(fill_ratio - 0.5) * 2  # Peak at 50% fill
    
    boundary_density = (boundary_pixels / total_pixels) * fill_penalty
    
    return float(boundary_density)


def compute_patch_boundary_scores(pt_file_paths: List, crop_size: int = 64) -> np.ndarray:
    """
    Pre-compute boundary scores for all patches across the dataset.
    
    Parameters
    ----------
    pt_file_paths : List
        List of paths to .pt files containing image/label pairs
    crop_size : int
        Size of crops to analyze
    
    Returns
    -------
    np.ndarray
        Array of boundary scores, one per .pt file
    """
    scores = []
    
    for pt_path in pt_file_paths:
        data = torch.load(pt_path, map_location='cpu')
        label_np = data['label'].numpy().squeeze()
        
        # Calculate mean boundary score for this image
        patch_scores = calculate_boundary_score(label_np, crop_size=crop_size)
        if patch_scores:
            mean_score = np.mean([score for _, _, score in patch_scores])
        else:
            mean_score = 0.0
        
        scores.append(mean_score)
    
    return np.array(scores)


class BoundaryAwareSampler(Sampler):
    """
    Sampler that oversamples images with high boundary content.
    
    This sampler analyzes the boundary content of each image and creates
    sampling weights that favor boundary-rich images, helping to balance
    the training distribution.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from
    crop_size : int
        Size of crops used in training
    boundary_weight : float
        Weight multiplier for high-boundary samples (default: 3.0)
    num_samples : int, optional
        Number of samples per epoch. If None, uses dataset length.
    """
    
    def __init__(self, dataset, crop_size: int = 64, boundary_weight: float = 3.0, num_samples: int = None):
        self.dataset = dataset
        self.crop_size = crop_size
        self.boundary_weight = boundary_weight
        self.num_samples = num_samples or len(dataset)
        
        # Pre-compute boundary scores for all samples
        print("Computing boundary scores for intelligent sampling...")
        self.boundary_scores = compute_patch_boundary_scores(
            dataset.pt_file_paths, crop_size=crop_size
        )
        
        # Create sampling weights
        self._create_sampling_weights()
    
    def _create_sampling_weights(self):
        """Create sampling weights based on boundary scores."""
        # Base weight is 1.0, boundary-rich samples get higher weight
        weights = np.ones(len(self.boundary_scores))
        
        # Find high-boundary samples (top 30% by boundary score)
        threshold = np.percentile(self.boundary_scores, 70)
        high_boundary_mask = self.boundary_scores >= threshold
        
        # Apply boundary weighting
        weights[high_boundary_mask] *= self.boundary_weight
        
        # Normalize weights
        self.sampling_weights = weights / weights.sum()
        
        print(f"Boundary sampler: {high_boundary_mask.sum()}/{len(weights)} samples "
              f"marked as boundary-rich (weight: {self.boundary_weight:.1f}x)")
    
    def __iter__(self):
        # Sample with replacement using boundary-aware weights
        indices = np.random.choice(
            len(self.dataset),
            size=self.num_samples,
            replace=True,
            p=self.sampling_weights
        )
        return iter(indices)
    
    def __len__(self):
        return self.num_samples