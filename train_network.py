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


def main():
    """Run the complete training pipeline."""
    print("=== Torch-Tomo-Slab Training Pipeline ===")
    
    # Display current configuration
    print(f"\nData Configuration:")
    print(f"  Tomograms directory: {constants.REFERENCE_TOMOGRAM_DIR}")
    print(f"  Masks directory: {constants.MASK_OUTPUT_DIR}")
    print(f"  Training data will be saved to: {constants.TRAIN_DATA_DIR}")
    print(f"  Validation data will be saved to: {constants.VAL_DATA_DIR}")
    
    print(f"\nTraining Configuration:")
    print(f"  Model: {config.MODEL_CONFIG['arch']} with {config.MODEL_CONFIG['encoder_name']} encoder")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Max epochs: {config.MAX_EPOCHS}")
    print(f"  Loss function: {config.LOSS_CONFIG['name']}")
    
    # Check if input directories exist
    if not constants.REFERENCE_TOMOGRAM_DIR.exists():
        print(f"\nWarning: Tomogram directory {constants.REFERENCE_TOMOGRAM_DIR} does not exist!")
        print("Please create this directory and place your .mrc tomogram files there.")
        return
    
    if not constants.MASK_OUTPUT_DIR.exists():
        print(f"\nWarning: Mask directory {constants.MASK_OUTPUT_DIR} does not exist!")
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


def train_with_custom_config(learning_rate=1e-4, 
                           batch_size=64, 
                           encoder='resnet18',
                           max_epochs=50,
                           loss_function='weighted_bce'):
    """
    Train with custom hyperparameters.
    
    This function demonstrates how to override default configurations
    for experimental purposes.
    
    Parameters
    ----------
    learning_rate : float, default=1e-4
        Learning rate for optimization.
    batch_size : int, default=64
        Training batch size.
    encoder : str, default='resnet18'
        Encoder backbone for the U-Net model.
    max_epochs : int, default=50
        Maximum number of training epochs.
    loss_function : str, default='weighted_bce'
        Loss function to use ('dice', 'bce', 'weighted_bce', etc.).
    """
    print("=== Custom Training Configuration ===")
    
    # Override configuration parameters
    config.LEARNING_RATE = learning_rate
    config.BATCH_SIZE = batch_size
    config.MAX_EPOCHS = max_epochs
    config.MODEL_CONFIG['encoder_name'] = encoder
    config.LOSS_CONFIG['name'] = loss_function
    
    print(f"Custom learning rate: {config.LEARNING_RATE}")
    print(f"Custom batch size: {config.BATCH_SIZE}")
    print(f"Custom encoder: {config.MODEL_CONFIG['encoder_name']}")
    print(f"Custom max epochs: {config.MAX_EPOCHS}")
    print(f"Custom loss function: {config.LOSS_CONFIG['name']}")
    
    # Run training with custom configuration
    trainer = TomoSlabTrainer()
    trainer.fit()
    
    print("✓ Custom training complete!")


if __name__ == "__main__":
    # Run the standard training pipeline
    main()
    
    # Uncomment the line below to run with custom parameters instead:
    # train_with_custom_config(learning_rate=1e-5, batch_size=32, encoder='resnet50')