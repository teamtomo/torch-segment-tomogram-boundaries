"""Project constants and file path configurations.

This module defines file paths, data processing constants, and other fixed
parameters used throughout the tomographic segmentation pipeline.
"""
from torch_tomo_slab.config import BASE_DATA_PATH
from torch_tomo_slab.config import MODEL_CONFIG

# --- Core Project Paths ---
# Input data directories for script 01 and 02
REFERENCE_TOMOGRAM_DIR = BASE_DATA_PATH / "data_in" / "volumes"

# Output data directories from scripts
MASK_OUTPUT_DIR = BASE_DATA_PATH / "data_in" / "boundary_mask_voumes"
PREPARED_DATA_BASE_DIR = BASE_DATA_PATH / "prepared_data"
TRAIN_DATA_DIR = PREPARED_DATA_BASE_DIR / "train"
VAL_DATA_DIR = PREPARED_DATA_BASE_DIR / "val"

# Flag for script 01 to generate diagnostic volumes
MULTIPLY_TOMO_MASK = True

# --- Data Processing Constants ---
# Final crop size for training transforms
FINAL_CROP_SIZE = 512
# Resize volume-labels to these dims
TARGET_VOLUME_SHAPE = (256, 512, 512)

# Number of 2D slices to extract from each 3D volume in script 02
NUM_SECTIONS_PER_VOLUME = 256

MODEL_ENCODER = MODEL_CONFIG['encoder_name']
