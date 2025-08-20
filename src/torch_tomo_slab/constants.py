from pathlib import Path

# --- Core Project Paths ---
# Base path for all data. Please adjust this to your system's layout.
BASE_DATA_PATH = Path("/home/pranav/data/training/torch-tomo-slab")

# Input data directories for script 01 and 02
IMOD_MODEL_DIR = BASE_DATA_PATH / "data_in" / "mods"
REFERENCE_TOMOGRAM_DIR = BASE_DATA_PATH / "data_in" / "volumes"

# Output data directories from scripts
MASK_OUTPUT_DIR = BASE_DATA_PATH / "data_in" / "boundary_mask_voumes"
PREPARED_DATA_BASE_DIR = BASE_DATA_PATH / "prepared_data"
TRAIN_DATA_DIR = PREPARED_DATA_BASE_DIR / "train"
VAL_DATA_DIR = PREPARED_DATA_BASE_DIR / "val"

# Flag for script 01 to generate diagnostic volumes
MULTIPLY_TOMO_MASK = True

# --- Data Processing Constants ---
# Resize volume-labels to these dims
TARGET_VOLUME_SHAPE = (256, 512, 512)

# Number of 2D slices to extract from each 3D volume in script 02
NUM_SECTIONS_PER_VOLUME = 256
