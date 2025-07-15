import pytorch_lightning as pl
from typing import List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch_tomo_slab.data.transforms import get_transforms
from torch_tomo_slab.data.dataset import PTFileDataset
from torch_tomo_slab.data.sampling import TorchioPatchSampler, IterablePatchDataset

class SegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule using a torchvision-based pipeline
    with an encapsulated torchio patch sampler and a diverse patch dataset.
    """
    def __init__(
            self,
            train_pt_files: List[Path],
            val_pt_files: List[Path],
            patch_size: Tuple[int, int],
            batch_size: int,
            num_workers: int,
            samples_per_volume: int,
            alpha_for_dropping: float,
            val_patch_sampling: bool,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['alpha_for_dropping'])
        foreground_weight = alpha_for_dropping
        background_weight = 1.0 - foreground_weight
        self.train_label_probabilities = {
            0: background_weight,
            1: foreground_weight
        }


    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transform = get_transforms(is_training=True)
            train_subjects_dataset = PTFileDataset(self.hparams.train_pt_files, transform=train_transform)

            train_patch_sampler = TorchioPatchSampler(
                patch_size=self.hparams.patch_size,
                samples_per_volume=self.hparams.samples_per_volume,
                label_probabilities=self.train_label_probabilities,
            )

            # --- USE THE NEW, DIVERSE DATASET ---
            self.train_dataset = IterablePatchDataset(
                subjects_dataset=train_subjects_dataset,
                patch_sampler=train_patch_sampler,
                shuffle_subjects=True,
            )

            val_transform = get_transforms(is_training=False)
            if self.hparams.val_patch_sampling:
                # Use the same diverse dataset for validation for consistency
                val_subjects_dataset = PTFileDataset(self.hparams.val_pt_files, transform=val_transform)
                val_patch_sampler = TorchioPatchSampler(
                    patch_size=self.hparams.patch_size,
                    samples_per_volume=self.hparams.samples_per_volume // 2, # Sample fewer patches for validation
                )
                self.val_dataset = IterablePatchDataset(
                    subjects_dataset=val_subjects_dataset,
                    patch_sampler=val_patch_sampler,
                    shuffle_subjects=False, # No need to shuffle validation data
                )
            else:
                self.val_dataset = PTFileDataset(self.hparams.val_pt_files, transform=val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
