# src/torch_tomo_slab/data/dataloading.py

from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torch_tomo_slab.data.dataset import PTFileDataset
from torch_tomo_slab.data.transforms import get_transforms
from torch_tomo_slab.data.sampling import BoundaryAwareSampler


class SegmentationDataModule(pl.LightningDataModule):
    """
    A simplified DataModule that loads full, pre-standardized 2D images
    and applies the robust augmentation pipeline.
    """

    def __init__(
        self,
        train_pt_files: List[Path],
        val_pt_files: List[Path],
        batch_size: int,
        num_workers: int,
        use_boundary_aware_sampling: bool = False,
        use_balanced_crop: bool = False,
        boundary_weight: float = 3.0,
    ):
        super().__init__()
        self.save_hyperparameters("batch_size", "num_workers", "use_boundary_aware_sampling", "use_balanced_crop", "boundary_weight")
        self.train_pt_files = train_pt_files
        self.val_pt_files = val_pt_files
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        This method is called by Lightning to prepare data. It creates the
        datasets and assigns the correct augmentation pipeline to each.
        """
        self.train_dataset = PTFileDataset(
            self.train_pt_files,
            transform=get_transforms(is_training=True, use_balanced_crop=self.hparams.use_balanced_crop)
        )
        self.val_dataset = PTFileDataset(
            self.val_pt_files,
            transform=get_transforms(is_training=False)
        )

    def train_dataloader(self) -> DataLoader:
        """Creates the training DataLoader with optional boundary-aware sampling."""
        # Use boundary-aware sampler if enabled
        if self.hparams.use_boundary_aware_sampling:
            from torch_tomo_slab import constants
            sampler = BoundaryAwareSampler(
                self.train_dataset,
                crop_size=constants.AUGMENTATION_CONFIG['CROP_SIZE'],
                boundary_weight=self.hparams.boundary_weight,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                sampler=sampler,
                # Note: shuffle=False when using custom sampler
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                shuffle=True,
            )

    def val_dataloader(self) -> DataLoader:
        """Creates the standard validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
        )
