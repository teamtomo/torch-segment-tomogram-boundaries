import torch
from pathlib import Path

# --- DATA ---
BASE_DATA_PATH = Path("/home/pranav/data/training/torch-tomo-slab")

# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE = 1e-4
MAX_EPOCHS = 50
PRECISION = 'bf16-mixed'
VALIDATION_FRACTION = 0.2

# --- MODEL ARCHITECTURE ---
MODEL_CONFIG = {
    'arch': "Unet",
    'encoder_name': "resnet18",
    'encoder_weights': None,
    'encoder_depth': 5,
    'decoder_channels': [256, 128, 64, 32, 16],
    'decoder_attention_type': 'scse',
    'classes': 1,
    'in_channels': 2,
}

# --- Loss Function Configuration ---
# 'name': Can be a single loss or multiple losses combined with '+'.
# 'weights': A list of weights for combined losses. Must match the number of losses.
# 'params': A nested dictionary for loss-specific hyperparameters (now unused).
LOSS_CONFIG = {
    'name': 'weighted_bce',  # Options: 'dice', 'bce', 'dice+bce', 'boundary', 'weighted_bce'.
    'weights': [0.5, 0.5],     # Only used for combined losses like 'dice+bce'
    'params': {}             # Simplified: No longer need focal/tversky params
}

# --- DATALOADER & AUGMENTATION ---
BATCH_SIZE: int = 64
NUM_WORKERS: int = 8

# --- AUGMENTATION PARAMETERS (NEW) ---
AUGMENTATION_CONFIG = {
    'CROP_SIZE': 512,
    'PAD_SIZE': 1024,
    'ROTATE_LIMIT': 90,
    'BRIGHTNESS_CONTRAST_LIMIT': 0.2,
    'GAUSS_NOISE_STD_RANGE': (0.0, 0.05),
    'GAUSS_BLUR_LIMIT': (3, 7),
    'LOCAL_VARIANCE_KERNEL_SIZE': 5
}

# --- DYNAMIC TRAINING MANAGEMENT ---
USE_DYNAMIC_MANAGER = True
EMA_ALPHA = 0.3                # Smoothing factor for validation metric (higher means more responsive)
SWA_TRIGGER_PATIENCE = 6       # Epochs of plateau before starting SWA
EARLY_STOP_PATIENCE = 8       # Epochs of no improvement (after SWA) before stopping
EARLY_STOP_MIN_DELTA = 0.005   # Minimum change to be considered an improvement

# --- FALLBACK: STANDARD CALLBACKS ---
STANDARD_EARLY_STOPPING_PATIENCE = 15
STANDARD_SWA_START_FRACTION = 0.75

# --- PL TRAINER & INFRASTRUCTURE ---
ACCELERATOR = "auto"
DEVICES = torch.cuda.device_count() if torch.cuda.is_available() else 1
MONITOR_METRIC = "val_dice"
LOG_EVERY_N_STEPS = 10
CHECK_VAL_EVERY_N_EPOCH = 1

# --- LR SCHEDULER ---
USE_LR_SCHEDULER = True
SCHEDULER_MONITOR = "val_dice"
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.1
SCHEDULER_MIN_LR = 1e-8

# --- STOCHASTIC WEIGHT AVERAGING (SWA) ---
USE_SWA = True
SWA_LEARNING_RATE = 0.1 * LEARNING_RATE # 10% of the main LR is a good starting point

# --- CHECKPOINTING ---
CHECKPOINT_SAVE_TOP_K = 1