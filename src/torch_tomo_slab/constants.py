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
    'CROP_SIZE': 256,                   # Square training patches from 256x512 orthogonal views
    'PAD_SIZE': 512,                    # Pad shorter dimension (256) to match longer (512) for square crops
    'ROTATE_LIMIT': 30,                 # Rotation limited to +/- 30 degrees
    'BRIGHTNESS_CONTRAST_LIMIT': 0.15,  # Keep brightness/contrast adjustments
    # 'GAUSS_NOISE_STD_RANGE': (0.0, 0.04), # Disabled
    # 'GAUSS_BLUR_LIMIT': (3, 5),         # Disabled as it can soften edges
    # 'ELASTIC_ALPHA': 10,                # Disabled
    # 'ELASTIC_SIGMA': 3,                 # Disabled
    'LOCAL_VARIANCE_KERNEL_SIZE': 5,
    # Balanced crop parameters (simple fill ratio filtering)
    'MIN_FILL_RATIO': 0.05,             # Minimum percentage of 1s in patch (5% = avoid pure empty)
    'MAX_FILL_RATIO': 0.95,             # Maximum percentage of 1s in patch (95% = avoid pure filled)
    'MAX_CROP_ATTEMPTS': 10,            # Maximum attempts to find balanced crop before fallback
}