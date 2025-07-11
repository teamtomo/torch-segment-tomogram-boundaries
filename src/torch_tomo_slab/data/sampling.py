from typing import Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio


class WeightedPatcheSampler:
    """
    Custom patch sampler that uses torchio's backend for grid sampling
    but operates on standard torch tensors.
    """

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

    def __call__(self, sample: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Generates patches from a single sample (image-label dict) with intelligent dropping.
        """
        image_3d = sample['image']
        label_3d = sample['label']

        if image_3d.ndim != 3:
            raise ValueError(f"Sampler expected a 3D image tensor (C, H, W), but got {image_3d.ndim}D.")

        image_4d = image_3d.unsqueeze(-1)
        label_4d = label_3d.unsqueeze(-1)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_4d),
            label=tio.LabelMap(tensor=label_4d),
        )

        label_tensor_for_stats = subject.label.data.squeeze()
        f0 = (label_tensor_for_stats == 0).sum().item()
        f1 = label_tensor_for_stats.numel() - f0

        drop_probability = 0.0
        if f0 > 0 and f1 > 0:
            # --- THIS IS THE FIX ---
            # The calculation results in a standard Python float.
            # We must use standard Python functions to clamp it, not torch.clip.
            unclamped_prob = 1 - self.alpha_for_dropping * (f1 / f0)
            drop_probability = min(1.0, max(0.0, unclamped_prob))

        grid_sampler = tio.inference.GridSampler(
            subject=subject,
            patch_size=self.patch_size,
            patch_overlap=self.overlap,
        )

        patches = []
        for patch_subject in grid_sampler:
            patch_label = patch_subject.label.data.squeeze()
            is_empty_patch = patch_label.sum().item() == 0
            if is_empty_patch and torch.rand(1).item() < drop_probability:
                continue

            patches.append({
                'image': patch_subject.image.data.squeeze(-1),
                'label': patch_subject.label.data.squeeze(-1)
            })
            if len(patches) >= self.samples_per_volume:
                break

        if len(patches) < self.samples_per_volume:
            uniform_sampler = tio.UniformSampler(self.patch_size)
            for _ in range(self.samples_per_volume - len(patches)):
                random_patch = next(uniform_sampler(subject))
                patch_label = random_patch.label.data.squeeze()
                is_empty_patch = patch_label.sum().item() == 0
                if is_empty_patch and torch.rand(1).item() < drop_probability:
                    continue
                patches.append({
                    'image': random_patch.image.data.squeeze(-1),
                    'label': random_patch.label.data.squeeze(-1)
                })

        return patches


class CustomPatchDataset(torch.utils.data.IterableDataset):
    # ... (This class is correct and remains unchanged) ...
    def __init__(
            self,
            subjects_dataset: Dataset,
            patch_sampler: WeightedPatcheSampler,
            shuffle_subjects: bool = True,
    ):
        self.subjects_dataset = subjects_dataset
        self.patch_sampler = patch_sampler
        self.shuffle_subjects = shuffle_subjects

    def __iter__(self):
        subject_indices = list(range(len(self.subjects_dataset)))
        if self.shuffle_subjects:
            worker_info = torch.utils.data.get_worker_info()
            seed = torch.initial_seed()
            if worker_info is not None:
                seed += worker_info.id
            generator = torch.Generator()
            generator.manual_seed(seed)
            perm = torch.randperm(len(subject_indices), generator=generator).tolist()
            subject_indices = [subject_indices[i] for i in perm]
        for subject_idx in subject_indices:
            subject_sample = self.subjects_dataset[subject_idx]
            patches = self.patch_sampler(subject_sample)
            for patch in patches:
                yield patch