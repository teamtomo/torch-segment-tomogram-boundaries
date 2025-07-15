from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio


class TorchioPatchSampler:
    """
    A robust patch sampler that leverages torchio's sampling strategies.
    This version returns a list of all patches from a given volume.
    """
    def __init__(
        self,
        patch_size: Tuple[int, int],
        samples_per_volume: int,
        label_probabilities: Optional[Dict[int, float]] = None,
    ):
        self.patch_size = patch_size + (1,)  # Add dummy Z dimension
        self.samples_per_volume = samples_per_volume

        if label_probabilities:
            self.sampler = tio.LabelSampler(
                patch_size=self.patch_size,
                label_name='label',
                label_probabilities=label_probabilities,
            )
        else:
            self.sampler = tio.UniformSampler(patch_size=self.patch_size)

        self.padder = tio.CropOrPad(
            target_shape=self.patch_size,
            padding_mode='constant',
        )

    def __call__(self, sample: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Generates a list of random patches from a single sample (image-label dict).
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

        padded_subject = self.padder(subject)

        patches = []
        patch_generator = self.sampler(padded_subject)

        for i, patch_subject in enumerate(patch_generator):
            if i >= self.samples_per_volume:
                break

            patches.append({
                'image': patch_subject.image.data.squeeze(-1),
                'label': patch_subject.label.data.squeeze(-1)
            })
        return patches

class IterablePatchDataset(torch.utils.data.IterableDataset):
    """
    An iterable dataset that ensures each batch contains patches from different
    source images, maximizing batch diversity. It achieves this by generating patches
    in a round-robin fashion from all subjects.
    """
    def __init__(
            self,
            subjects_dataset: Dataset,
            patch_sampler: TorchioPatchSampler,
            shuffle_subjects: bool = True,
    ):
        super().__init__()
        self.subjects_dataset = subjects_dataset
        self.patch_sampler = patch_sampler
        self.num_patches_per_subject = self.patch_sampler.samples_per_volume
        self.shuffle_subjects = shuffle_subjects

    def __len__(self) -> int:
        """
        Returns the total number of patches that will be generated across all subjects.
        """
        return len(self.subjects_dataset) * self.num_patches_per_subject

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Each worker gets a unique slice of the subjects
        all_subject_indices = list(range(len(self.subjects_dataset)))
        subject_indices_for_this_worker = all_subject_indices[worker_id::num_workers]

        # Pre-generate all patches for all subjects this worker is responsible for.
        # This can be memory-intensive but ensures correct behavior.
        patches_from_all_subjects = []
        for subject_idx in subject_indices_for_this_worker:
            subject_sample = self.subjects_dataset[subject_idx]
            patches_from_all_subjects.append(self.patch_sampler(subject_sample))

        # This generator seeds itself based on the worker's seed provided by the DataLoader
        generator = torch.Generator()
        generator.manual_seed(torch.initial_seed())

        # Iterate in a "round-robin" or "pass-through" fashion
        for i in range(self.num_patches_per_subject):
            # Collect the i-th patch from every subject
            pass_patches = [subject_patches[i] for subject_patches in patches_from_all_subjects]

            # Shuffle the order of patches within this pass
            if self.shuffle_subjects:
                perm = torch.randperm(len(pass_patches), generator=generator)
                for p_idx in perm:
                    yield pass_patches[p_idx]
            else:
                for patch in pass_patches:
                    yield patch
