from typing import Tuple, List

import numpy as np
import torchio as tio
import torch


class WeightedPatcheSampler:
    """Custom patch sampler that implements intelligent empty patch dropping."""

    def __init__(
            self,
            patch_size: Tuple[int, int],
            samples_per_volume: int,
            alpha_for_dropping: float = 1.0,
            overlap: int = 0,
    ):
        self.patch_size = patch_size + (1,)
        self.samples_per_volume = samples_per_volume
        self.alpha_for_dropping = alpha_for_dropping
        self.overlap = np.array([overlap, overlap, 0], dtype=int)

    def __call__(self, subject: tio.Subject) -> List[tio.Subject]:
        """Generate patches for a subject with intelligent dropping."""
        # ... (code before the loop is fine)
        label_tensor = subject.label.data.squeeze()
        f0 = (label_tensor == 0).sum().item()
        f1 = label_tensor.numel() - f0
        drop_probability = 0.0
        if f0 > 0 and f1 > 0:
            drop_probability = torch.clip(
                1 - self.alpha_for_dropping * (f1 / f0), min=0.0, max=1.0
            ).item()

        grid_sampler = tio.inference.GridSampler(
            subject=subject,
            patch_size=self.patch_size,
            patch_overlap=self.overlap,
        )

        patches = []
        patch_generator = iter(grid_sampler)

        for _ in range(self.samples_per_volume * 2):
            try:
                patch = next(patch_generator)
                patch_label = patch.label.data.squeeze()
                is_empty_patch = patch_label.sum().item() == 0
                if is_empty_patch and torch.rand(1).item() < drop_probability:
                    continue
                patches.append(patch)
                if len(patches) >= self.samples_per_volume:
                    break
            except StopIteration:
                # This block is entered when the GridSampler runs out of patches.
                uniform_sampler = tio.UniformSampler(self.patch_size)
                while len(patches) < self.samples_per_volume:
                    random_patch_generator = uniform_sampler(subject)
                    patch = next(random_patch_generator)

                    patch_label = patch.label.data.squeeze()
                    is_empty_patch = patch_label.sum().item() == 0

                    if is_empty_patch and torch.rand(1).item() < drop_probability:
                        continue

                    patches.append(patch)
                break
        return patches


class CustomPatchDataset(torch.utils.data.IterableDataset):
    """Custom dataset that yields patches with intelligent sampling."""

    def __init__(
            self,
            subjects_dataset: tio.SubjectsDataset,
            patch_sampler: WeightedPatcheSampler,
            shuffle_subjects: bool = True,
    ):
        self.subjects_dataset = subjects_dataset
        self.patch_sampler = patch_sampler
        self.shuffle_subjects = shuffle_subjects

    def __iter__(self):
        # Get subject indices
        subject_indices = list(range(len(self.subjects_dataset)))
        if self.shuffle_subjects:
            torch.manual_seed(torch.initial_seed())  # For reproducibility
            subject_indices = torch.randperm(len(subject_indices)).tolist()

        # Generate patches from each subject
        for subject_idx in subject_indices:
            subject = self.subjects_dataset[subject_idx]
            patches = self.patch_sampler(subject)

            for patch in patches:
                yield patch