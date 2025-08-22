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
import os
from typing import Union

from torch_tomo_slab.processing import TrainingDataGenerator
from torch_tomo_slab.trainer import TomoSlabTrainer


def train_and_prep(tomo_dir:Union[str,os.PathLike],
                   mask_vol_dir:Union[str,os.PathLike],
                   output_train_dir:Union[str,os.PathLike],
                   output_val_dir:Union[str,os.PathLike], ckpt_save_dir:Union[str,os.PathLike]):

    # Step 1: Prepare training data
    print("\n=== Step 1: Preparing Training Data ===")
    print("Converting 3D volumes to 2D training slices...")
    
    generator = TrainingDataGenerator(volume_dir=tomo_dir,
                                      mask_dir=mask_vol_dir,
                                      output_train_dir=output_train_dir,
                                      output_val_dir=output_val_dir)
    generator.run()
    
    print("✓ Data preparation complete.")
    
    # Step 2: Train the model
    print("\n=== Step 2: Training Model ===")
    print("Starting training process...")
    
    trainer = TomoSlabTrainer(train_data_dir=output_train_dir,
                              val_data_dir=output_val_dir,
                              ckpt_save_dir=ckpt_save_dir)
    trainer.fit()
    
    print("✓ Training complete!")
    print("\nTrained model checkpoints are saved in the 'lightning_logs' directory.")
    print("Look for the best checkpoint (highest validation dice score) to use for inference.")


if __name__ == "__main__":
    tomo_dir = "/home/pranav/data/training/torch-tomo-slab/data_in/volumes"
    mask_vol_dir = "/home/pranav/data/training/torch-tomo-slab/data_in/boundary_mask_voumes"
    output_train_dir = "/home/pranav/data/training/torch-tomo-slab/prepared_data/train"
    output_val_dir = "/home/pranav/data/training/torch-tomo-slab/prepared_data/val"
    ckpt_save_dir = "/home/pranav/data/training/torch-tomo-slab/"
    train_and_prep(tomo_dir=tomo_dir,
                   mask_vol_dir = mask_vol_dir,
                   output_train_dir = output_train_dir,
                   output_val_dir = output_val_dir,
                   ckpt_save_dir=ckpt_save_dir)