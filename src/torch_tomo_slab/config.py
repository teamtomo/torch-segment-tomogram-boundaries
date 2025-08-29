"""Configuration parameters for tomographic boundary segmentation.

This module centralizes all configuration parameters including model architecture,
training hyperparameters, data paths, loss functions, and augmentation settings.
All parameters can be overridden programmatically for experiments.
"""
import os
import torch
from pathlib import Path

# --- DATA ---
BASE_DATA_PATH = Path(os.environ.get("TORCH_TOMO_SLAB_DATA", "data/"))

# Input data directories for script 01 and 02
TOMOGRAM_DIR: Path = BASE_DATA_PATH / "data_in" / "volumes"

# Output data directories from scripts
MASKS_DIR: Path = BASE_DATA_PATH / "data_in" / "boundary_mask_voumes"
PREPARED_DATA_BASE_DIR: Path = BASE_DATA_PATH / "prepared_data"
TRAIN_DATA_DIR: Path = PREPARED_DATA_BASE_DIR / "train"
VAL_DATA_DIR: Path = PREPARED_DATA_BASE_DIR / "val"

# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE: float = 2e-4  # Reduced for stability
MAX_EPOCHS: int = 50
PRECISION: str = '32'
WARMUP_EPOCHS: int = 5

# --- MODEL ARCHITECTURE ---
MODEL_CONFIG: dict[str, any] = {
    'arch': "Unet",
    'encoder_name': "resnet18",
    'encoder_weights': None,
    'encoder_depth': 5,
    'decoder_channels': [256, 128, 64, 32,16],
    'decoder_attention_type': 'scse',
    'classes': 1,
    'in_channels': 1,
    'dropout': 0.4,  # Increased dropout for overfitting prevention
}

# --- Loss Function Configuration ---
# 'name': Can be a single loss or multiple losses combined with '+'.
# 'weights': A list of weights for combined losses. Must match the number of losses.
# 'params': A nested dictionary for loss-specific hyperparameters (now unused).
LOSS_CONFIG: dict[str, any] = {
    'name': 'weighted_bce',  # Options: 'dice', 'bce', 'dice+bce', 'boundary', 'weighted_bce'.
    'weights': [0.5, 0.5],     # Only used for combined losses like 'dice+bce'
    'params': {
        'gradient_weight': 10,  # Add label smoothing to reduce overconfidence
    }
}

# --- DATALOADER & AUGMENTATION ---
BATCH_SIZE: int = 32
NUM_WORKERS: int = 8

# --- DYNAMIC TRAINING MANAGEMENT ---
USE_DYNAMIC_MANAGER: bool = True
EMA_ALPHA: float = 0.1                 # Lower for more stability (less responsive to noise)
SWA_TRIGGER_PATIENCE: int = 4       # Reduced patience - stop overfitting sooner
EARLY_STOP_PATIENCE: int = 3       # Aggressive early stopping to prevent overfitting
EARLY_STOP_MIN_DELTA: float = 1e-3   # Higher threshold - only continue if significant improvement

# --- FALLBACK: STANDARD CALLBACKS ---
STANDARD_EARLY_STOPPING_PATIENCE: int = 6
STANDARD_SWA_START_FRACTION: float = 0.75

# --- PL TRAINER & INFRASTRUCTURE ---
ACCELERATOR: str = "auto"
DEVICES: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
SWA_LEARNING_RATE: float = 0.1 * LEARNING_RATE # 10% of the main LR is a good starting point

# --- CHECKPOINTING ---
CKPT_SAVE_PATH: Path = BASE_DATA_PATH/"checkpoints"
CHECKPOINT_SAVE_TOP_K: int = 1

# --- OPTIMIZER ---
OPTIMIZER_CONFIG: dict[str, any] = {
    "name": "AdamW",
    "params": {
        "lr": LEARNING_RATE,
        "weight_decay": 5e-4,  # Increased weight decay for regularization
        "eps": 1e-7  # More stable epsilon for AdamW
    }
}

# --- SCHEDULER ---
USE_LR_SCHEDULER: bool = True
SCHEDULER_CONFIG: dict[str, any] = {
    "name": "CosineAnnealingLR", 
    "params": {
        "eta_min": 5e-6,  # Higher minimum LR for stability
    },
    "monitor": "val_dice"
}
