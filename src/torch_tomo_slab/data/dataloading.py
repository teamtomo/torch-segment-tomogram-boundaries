import pytorch_lightning as pl
import torch
import torchio as tio
from pathlib import Path
from typing import Optional, List, Union

from ..config import (
    PATCH_SIZE, 
    OVERLAP, 
    SAMPLES_PER_VOLUME, 
    ALPHA_FOR_DROPPING, 
    NUM_WORKERS, 
    BATCH_SIZE, 
    VALIDATION_PATCH_SAMPLING
)
from .dataset import PTFileDataset
from .sampling import IterablePatchDataset, TorchioPatchSampler
from .transforms import get_transforms


class SegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling 2D segmentation data.
    
    This module orchestrates the creation of training and validation datasets and dataloaders,
    including patch-based sampling.
    """
    def __init__(
        self,
        train_pt_files: List[Path],
        val_pt_files: List[Path],
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        patch_size: int = PATCH_SIZE,
        overlap: int = OVERLAP,
        samples_per_volume: int = SAMPLES_PER_VOLUME,
        alpha_for_dropping: float = ALPHA_FOR_DROPPING,
        val_patch_sampling: bool = VALIDATION_PATCH_SAMPLING,
    ):
        super().__init__()
        # Using save_hyperparameters allows us to access them with self.hparams
        self.save_hyperparameters()

def setup(self, stage: Optional[str] = None):
        """
        This method is called by PyTorch Lightning to set up the datasets.
        """
        # --- Define Label Probabilities for Sampling ---
        foreground_weight = self.hparams.alpha_for_dropping
        background_weight = 1.0 - foreground_weight
        label_probabilities = {0: background_weight, 1: foreground_weight}

        # --- FIX: Explicitly cast hyperparameters to int ---
        # This prevents the "ambiguous truth value" error when using hparams
        # in a distributed environment.
        patch_size_int = int(self.hparams.patch_size)
        overlap_int = int(self.hparams.overlap)

        # --- Training Dataset and Sampler ---
        train_transform = get_transforms(is_training=True)
        train_subjects_dataset = PTFileDataset(self.hparams.train_pt_files, transform=train_transform)

        training_sampler = tio.LabelSampler(
            patch_size=(patch_size_int, patch_size_int), # Use the corrected int value
            label_name='label',
            label_probabilities=label_probabilities,
        )
        
        train_patch_sampler = TorchioPatchSampler(
            subjects_dataset=train_subjects_dataset,
            sampler=training_sampler,
            max_patches_per_volume=self.hparams.samples_per_volume
        )
        
        self.train_dataset = IterablePatchDataset(
            subjects_dataset=train_subjects_dataset,
            patch_sampler=train_patch_sampler,
        )

        # --- Validation Dataset and Sampler ---
        val_transform = get_transforms(is_training=False)
        val_subjects_dataset = PTFileDataset(self.hparams.val_pt_files, transform=val_transform)

        if self.hparams.val_patch_sampling:
            print("Using label-based patch sampling for the validation set.")
            validation_sampler = tio.LabelSampler(
                patch_size=(patch_size_int, patch_size_int), # Use the corrected int value
                label_name='label',
                label_probabilities=label_probabilities,
            )
        else:
            print("Using grid-based patch sampling for the validation set.")
            validation_sampler = tio.GridSampler(
                patch_size=(patch_size_int, patch_size_int), # Use the corrected int value
                patch_overlap=(overlap_int, overlap_int)   # Use the corrected int value
            )

        val_patch_sampler = TorchioPatchSampler(
            subjects_dataset=val_subjects_dataset,
            sampler=validation_sampler,
            max_patches_per_volume=self.hparams.samples_per_volume 
        )

        self.val_dataset = IterablePatchDataset(
            subjects_dataset=val_subjects_dataset, 
            patch_sampler=val_patch_sampler
        )
