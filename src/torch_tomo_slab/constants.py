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

# --- AUGMENTATION PARAMETERS (NEW) ---
AUGMENTATION_CONFIG: dict[str, any] = {
    'PAD_HEIGHT': 512,     # Rectangular padding
    'PAD_WIDTH': 768,      # Different aspect ratio
    'CROP_HEIGHT': 256,    # Rectangular crops
    'CROP_WIDTH': 384,     # Preserve aspect relationships
    'ROTATE_LIMIT': 15,    # Reduced to preserve edge geometry
    'BRIGHTNESS_CONTRAST_LIMIT': 0.15,  # Gentler intensity changes
    'MIN_FILL_RATIO': 0.05,  # More lenient for edge cases
    'MAX_FILL_RATIO': 0.95,
    'MAX_CROP_ATTEMPTS': 20,
}