from pathlib import Path

BASE_DATA_PATH = Path("/home/pranav/data/training/torch-tomo-slab")  # Adjust this to your project's root data folder

# Input data for script 01 and 02
#RAW_DATA_PATH = BASE_DATA_PATH/"data_in"/""
IMOD_MODEL_DIR = BASE_DATA_PATH/"data_in"/"mods"
REFERENCE_TOMOGRAM_DIR = BASE_DATA_PATH/"data_in"/"volumes"

# Output data from scripts
MASK_OUTPUT_DIR =  BASE_DATA_PATH / "data_in" / "boundary_mask_voumes"
PREPARED_DATA_DIR = BASE_DATA_PATH / "prepared_data" / "labels"  # Output of script 02, input for train.py
MULTIPLY_TOMO_MASK = True

# Number of 2D slices to extract from each 3D volume
NUM_SECTIONS_PER_VOLUME = 120

# Kernel size for the local variance calculation
LOCAL_VARIANCE_KERNEL_SIZE = 5

# --- TRAINING PARAMETERS ---
# Model Architecture (from segmentation-models-pytorch)
MODEL_ARCH = "Unet"
MODEL_ENCODER = "resnet34"

# Training Hyperparameters
LEARNING_RATE = 1e-3
LOSS_FUNCTION = "dice+bce"  # Options: 'dice', 'bce', 'dice+bce'
LOSS_WEIGHTS = (0.5,0.5)
MAX_EPOCHS = 100
PRECISION='bf16-mixed'
# Dataloader & Augmentation
PATCH_SIZE = 128
OVERLAP = 64
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 16
NUM_WORKERS = 8  # Adjust based on your machine's CPUs

# Patch Sampling Strategy
SAMPLES_PER_VOLUME = 200  # Number of patches to extract per 2D image
ALPHA_FOR_DROPPING = 0.75  # Controls how aggressively to drop empty patches. 0=no drop, 1=aggressive.
VALIDATION_PATCH_SAMPLING = True  # Set to False for validation on full images, True for patches

# --- TRAINER CONFIGURATION ---
ACCELERATOR = "auto"  # Let PyTorch Lightning detect GPU/MPS/CPU
DEVICES = 2
LOG_EVERY_N_STEPS = 10
CHECK_VAL_EVERY_N_EPOCH = 5

# --- NEW: LEARNING RATE SCHEDULER CONFIGURATION ---
USE_LR_SCHEDULER = True
# Number of epochs to slowly ramp up the learning rate from 0
SCHEDULER_WARMUP_EPOCHS = 10
# The maximum learning rate is the one we already defined
# SCHEDULER_MAX_LR = LEARNING_RATE
# The minimum learning rate at the end of the cosine cycle
SCHEDULER_MIN_LR = 1e-6

# --- CALLBACK CONFIGURATION ---
# The metric to monitor for checkpointing and early stopping
MONITOR_METRIC = "val_loss"

# Early Stopping configuration
EARLY_STOPPING_PATIENCE = 15  # Stop after 15 validation epochs with no improvement
EARLY_STOPPING_MIN_DELTA = 0.001 # Minimum change to be considered an improvement

# Model Checkpoint configuration
CHECKPOINT_SAVE_TOP_K = 1 # Save only the single best model

# Stochastic Weight Averaging (SWA) configuration
USE_SWA = True
SWA_LEARNING_RATE = 1e-4 # SWA learning rate is typically smaller
SWA_START_EPOCH_FRACTION = 0.75 # Start SWA in the last 25% of epochs
