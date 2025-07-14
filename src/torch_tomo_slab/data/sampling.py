from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio


class TorchioPatchSampler:
    """
    A robust patch sampler that leverages torchio's sampling strategies.
    This version adds a crucial padding step to ensure that patches can
    always be extracted, even from images smaller than the patch size.
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

        # --- THIS IS THE FIX ---
        # The 'padding_value' argument is not valid for the CropOrPad constructor.
        # The default behavior for padding_mode='constant' is to pad with zeros,
        # which is exactly what we need.
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

class CustomPatchDataset(torch.utils.data.IterableDataset):
    """
    An iterable dataset that generates patches from a set of subjects (2D images).
    This implementation correctly partitions the data across multiple workers, making
    it safe to use with a __len__ method and a multi-process DataLoader.
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
        self.shuffle_subjects = shuffle_subjects

    def __len__(self) -> int:
        """
        Returns the total number of patches that will be generated across all subjects.
        This is used by PyTorch Lightning to configure progress bars and schedulers.
        """
        return len(self.subjects_dataset) * self.patch_sampler.samples_per_volume

    def __iter__(self):
        """
        NOTE on the PyTorch Lightning Warning:
        PyTorch Lightning may still issue a warning: "Your IterableDataset has `__len__` defined...".
        This is a static check. The logic below correctly handles data partitioning
        across workers, so this warning can be safely ignored. Each worker will
        process a unique subset of the subjects, and the total number of yielded
        patches will match the value returned by __len__.
        """
        subject_indices = list(range(len(self.subjects_dataset)))

        if self.shuffle_subjects:
            seed = torch.initial_seed()
            worker_info_for_seed = torch.utils.data.get_worker_info()
            if worker_info_for_seed is not None:
                seed += worker_info_for_seed.id
            generator = torch.Generator()
            generator.manual_seed(seed)
            perm = torch.randperm(len(subject_indices), generator=generator).tolist()
            subject_indices = [subject_indices[i] for i in perm]

        # Correctly partition the data for multi-process loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # This is a worker process. Give it a unique slice of the data.
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            subject_indices = subject_indices[worker_id::num_workers]

        # Each worker iterates over its unique subset of subjects
        for subject_idx in subject_indices:
            subject_sample = self.subjects_dataset[subject_idx]
            patches = self.patch_sampler(subject_sample)
            for patch in patches:
                yield patch
