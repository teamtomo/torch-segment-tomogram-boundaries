import pytorch_lightning as pl
from typing import List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch_tomo_slab.data.transforms import get_transforms
from torch_tomo_slab.data.dataset import PTFileDataset
from torch_tomo_slab.data.sampling import WeightedPatcheSampler, CustomPatchDataset

class SegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule using a torchvision-based pipeline
    with an encapsulated torchio patch sampler.
    """
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
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        # ... (setup logic is correct and remains unchanged)
        if stage == "fit" or stage is None:
            train_transform = get_transforms(is_training=True)
            train_subjects_dataset = PTFileDataset(self.hparams.train_pt_files, transform=train_transform)

            patch_sampler = WeightedPatcheSampler(
                patch_size=self.hparams.patch_size,
                samples_per_volume=self.hparams.samples_per_volume,
                alpha_for_dropping=self.hparams.alpha_for_dropping,
                overlap=self.hparams.overlap
            )

            self.train_dataset = CustomPatchDataset(
                subjects_dataset=train_subjects_dataset,
                patch_sampler=patch_sampler,
                shuffle_subjects=True,
            )

            val_transform = get_transforms(is_training=False)
            if self.hparams.val_patch_sampling:
                val_subjects_dataset = PTFileDataset(self.hparams.val_pt_files, transform=val_transform)
                val_patch_sampler = WeightedPatcheSampler(
                    patch_size=self.hparams.patch_size,
                    samples_per_volume=self.hparams.samples_per_volume // 2,
                    alpha_for_dropping=0.0,
                    overlap=self.hparams.overlap
                )
                self.val_dataset = CustomPatchDataset(
                    subjects_dataset=val_subjects_dataset,
                    patch_sampler=val_patch_sampler,
                    shuffle_subjects=False,
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