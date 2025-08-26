"""Project constants and file path configurations.

This module defines file paths, data processing constants, and other fixed
parameters used throughout the tomographic segmentation pipeline.
"""
from torch_tomo_slab.config import MODEL_CONFIG

VALIDATION_FRACTION: float = 0.2
# --- Data Processing Constants ---
# Final crop size for training transforms
FINAL_CROP_SIZE = 512

# Resize volume-labels to these dims
TARGET_VOLUME_SHAPE = (256, 512, 512)

# Number of 2D slices to extract from each 3D volume in script 02
NUM_SECTIONS_PER_VOLUME = 256

MODEL_ENCODER = MODEL_CONFIG['encoder_name']
MONITOR_METRIC: str = "val_dice"
LOG_EVERY_N_STEPS: int = 10
CHECK_VAL_EVERY_N_EPOCH: int = 1
# --- STOCHASTIC WEIGHT AVERAGING (SWA) ---
USE_SWA: bool = True

# --- AUGMENTATION PARAMETERS (ENHANCED FOR REGULARIZATION) ---
AUGMENTATION_CONFIG: dict[str, any] = {
    # No padding needed - data is preprocessed to consistent size
    
    # Geometric augmentation parameters
    'ROTATE_LIMIT': 20,              # Moderate rotation for rectangular images
    'ELASTIC_ALPHA': 50,             # Stronger elastic deformation 
    'ELASTIC_SIGMA': 5,              # Smoothness parameter
    'GRID_DISTORTION_LIMIT': 0.2,   # Grid distortion strength
    
    # Intensity augmentation parameters  
    'BRIGHTNESS_CONTRAST_LIMIT': 0.2,  # More aggressive intensity changes
    'GAMMA_LIMIT': (80, 120),          # Gamma variation range
    'NOISE_VAR_LIMIT': (10, 30),       # Gaussian noise strength
    'BLUR_LIMIT': 3,                   # Gaussian blur kernel limit
    
    # Dropout/occlusion parameters
    'COARSE_DROPOUT_HOLES': (3, 8),        # Number of holes range
    'COARSE_DROPOUT_SIZE': (0.03, 0.08),   # Hole size range
    'GRID_DROPOUT_RATIO': 0.3,             # Grid dropout coverage
    'GRID_DROPOUT_UNIT_SIZE': (8, 16),     # Grid unit size range
}
LOCAL_VARIANCE_KERNEL_SIZE: int = 5