import pytorch_lightning as pl
from typing import List, Optional, Tuple
from pathlib import Path

from torch.utils.data import DataLoader
from torch_tomo_slab.data.transforms import get_transforms
from torch_tomo_slab.data.dataset import PTFileDataset
from torch_tomo_slab.data.sampling import WeightedPatcheSampler, CustomPatchDataset


class SegmentationDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule with intelligent patch sampling."""

    def __init__(
            self,
            train_pt_files: List[Path],
            val_pt_files: List[Path],
            patch_size: Tuple[int, int],
            overlap: int,
            batch_size: int,
            num_workers: int,
            samples_per_volume: int,
            alpha_for_dropping: float,
            val_patch_sampling: bool,
    ):
        super().__init__()
        self.train_pt_files = train_pt_files
        self.val_pt_files = val_pt_files
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_volume = samples_per_volume
        self.alpha_for_dropping = alpha_for_dropping
        self.val_patch_sampling = val_patch_sampling

    def setup(self, stage: Optional[str] = None):
        # Training dataset with augmentations
        if stage == "fit" or stage is None:
            train_transform = get_transforms(self.patch_size, is_training=True)
            train_subjects_dataset = PTFileDataset(self.train_pt_files, transform=train_transform)

            # Create intelligent patch sampler
            patch_sampler = WeightedPatcheSampler(
                patch_size=self.patch_size,
                samples_per_volume=self.samples_per_volume,
                alpha_for_dropping=self.alpha_for_dropping,
                overlap = self.overlap
            )

            # Create custom patch dataset
            self.train_dataset = CustomPatchDataset(
                subjects_dataset=train_subjects_dataset,
                patch_sampler=patch_sampler,
                shuffle_subjects=True,
            )

            # Validation dataset
            val_transform = get_transforms(self.patch_size, is_training=False)
            if self.val_patch_sampling:
                val_subjects_dataset = PTFileDataset(self.val_pt_files, transform=val_transform)
                val_patch_sampler = WeightedPatcheSampler(
                    patch_size=self.patch_size,
                    samples_per_volume=self.samples_per_volume // 2,  # Fewer patches for validation
                    alpha_for_dropping=0.0,
                    overlap_factor = self.overlap_factor# No dropping for validation
                )
                self.val_dataset = CustomPatchDataset(
                    subjects_dataset=val_subjects_dataset,
                    patch_sampler=val_patch_sampler,
                    shuffle_subjects=False,
                )
            else:
                # Use full images for validation
                self.val_dataset = PTFileDataset(self.val_pt_files, transform=val_transform)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
