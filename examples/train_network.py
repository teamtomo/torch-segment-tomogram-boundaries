#!/usr/bin/env python3
"""
Training script for torch-tomo-slab network.

This script demonstrates how to use the torch-tomo-slab library to:
1. Prepare training data from 3D tomograms and masks
2. Train a segmentation model with customizable parameters
3. Save the trained model for later inference

Usage:
    python train_network.py

Make sure your data paths are correctly set in the config.py file before running.
"""
from pathlib import Path

from torch_tomo_slab import config
from torch_tomo_slab.processing import TrainingDataGenerator
from torch_tomo_slab.trainer import train


def train_and_prep(
    tomo_dir: Path,
    mask_vol_dir: Path,
    output_train_dir: Path,
    output_val_dir: Path,
    ckpt_save_dir: Path,
):
    """
    Prepares data and runs the training pipeline.
    """
    train_exists = output_train_dir.exists() and any(output_train_dir.iterdir())
    val_exists = output_val_dir.exists() and any(output_val_dir.iterdir())

    if not (train_exists and val_exists):
        print("\n=== Step 1: Preparing Training Data ===")
        print("Converting 3D volumes to 2D training slices...")

        generator = TrainingDataGenerator(
            volume_dir=tomo_dir,
            mask_dir=mask_vol_dir,
            output_train_dir=output_train_dir,
            output_val_dir=output_val_dir,
        )
        generator.run()
        print("✓ Data preparation complete.")
    else:
        print("✓ Training data already exists, skipping preparation.")

    # Step 2: Train the model
    print("\n=== Step 2: Training Model ===")
    print("Starting training process...")

    train(
        train_data_dir=output_train_dir,
        val_data_dir=output_val_dir,
        ckpt_save_dir=ckpt_save_dir,
    )

    print("✓ Training complete!")
    print(f"\nTrained model checkpoints are saved in the '{ckpt_save_dir}' directory.")
    print("Look for the best checkpoint (highest validation dice score) to use for inference.")


if __name__ == "__main__":
    tomo_dir = Path("/home/pranav/Desktop/pranav/data/training/torch-tomo-slab/data_in/volumes")
    mask_vol_dir = Path("/home/pranav/Desktop/pranav/data/training/torch-tomo-slab/data_in/boundary_mask_volumes")
    output_train_dir = Path("/home/pranav/Desktop/pranav/data/training/torch-tomo-slab/prepared_data/train")
    output_val_dir = Path("/home/pranav/Desktop/pranav/data/training/torch-tomo-slab/prepared_data/val/")
    ckpt_save_dir = Path("/home/pranav/data/training/torch-tomo-slab/weights")
    train_and_prep(
        tomo_dir=tomo_dir,
        mask_vol_dir=mask_vol_dir,
        output_train_dir=output_train_dir,
        output_val_dir=output_val_dir,
        ckpt_save_dir=ckpt_save_dir,
    )
