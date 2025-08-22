# src/torch_tomo_slab/data/dataloading.py

from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torch_tomo_slab.data.dataset import PTFileDataset
from torch_tomo_slab.data.transforms import get_transforms


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
        use_balanced_crop: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters("batch_size", "num_workers", "use_balanced_crop")
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
        """Creates the standard training DataLoader."""
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
