#!/usr/bin/env python3
"""
Training script for torch-tomo-slab network.

This script demonstrates how to use the torch-tomo-slab library to:
1. Prepare training data from 3D tomograms and masks
2. Train a segmentation model with customizable parameters
3. Save the trained model for later inference

Usage:
    python train_network.py

Make sure your data paths are correctly set in the constants.py file before running.
"""

from pathlib import Path
from torch_tomo_slab import config, constants
from torch_tomo_slab.processing import TrainingDataGenerator
from torch_tomo_slab.trainer import TomoSlabTrainer


def train_and_prep(tomo_dir,
            mask_vol_dir,
            prepared_data_out_dir):

    """Run the complete training pipeline."""
    print("=== Torch-Tomo-Slab Training Pipeline ===")
    print("=== Prepare training inputs ===")
    config.TOMOGRAM_DIR = tomo_dir
    config.MASKS_DIR = mask_vol_dir
    config.PREPARED_DATA_BASE_DIR = prepared_data_out_dir

    # Display current configuration
    print(f"\nData Configuration:")
    print(f"  Tomograms directory: {config.TOMOGRAM_DIR}")
    print(f"  Masks directory: {config.MASKS_DIR}")
    print(f"  Train/Val data will be saved to: {config.PREPARED_DATA_BASE_DIR}")
    
    # Check if input directories exist
    if not config.TOMOGRAM_DIR.exists():
        print(f"\nWarning: Tomogram directory {config.TOMOGRAM_DIR} does not exist!")
        print("Please create this directory and place your .mrc tomogram files there.")
        return
    
    if not config.MASKS_DIR.exists():
        print(f"\nWarning: Mask directory {config.MASKS_DIR} does not exist!")
        print("Please create this directory and place your .mrc mask files there.")
        return
    
    # Step 1: Prepare training data
    print("\n=== Step 1: Preparing Training Data ===")
    print("Converting 3D volumes to 2D training slices...")
    
    generator = TrainingDataGenerator()
    generator.run()
    
    print("✓ Data preparation complete.")
    
    # Step 2: Train the model
    print("\n=== Step 2: Training Model ===")
    print("Starting training process...")
    
    trainer = TomoSlabTrainer()
    trainer.fit()
    
    print("✓ Training complete!")
    print("\nTrained model checkpoints are saved in the 'lightning_logs' directory.")
    print("Look for the best checkpoint (highest validation dice score) to use for inference.")


if __name__ == "__main__":
    tomo_dir = "blah"
    mask_vol_dir = "blah"
    prepared_data_out_dir = "blah"
    train_and_prep(tomo_dir, mask_vol_dir, prepared_data_out_dir)