# src/torch_tomo_slab/data/dataset.py

from pathlib import Path
from typing import Dict, List, Optional

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset


class PTFileDataset(Dataset):
    """
    A Dataset class for loading pre-prepared 2D slice data from .pt files.
    This version is designed to work seamlessly with the Albumentations pipeline.
    """
    def __init__(self, pt_file_paths: List[Path], transform: Optional[A.Compose] = None):
        """
        Args:
            pt_file_paths: A list of Paths to the .pt files.
            transform: An Albumentations Compose object.
        """
        self.pt_file_paths = pt_file_paths
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.pt_file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads a sample, applies transforms, and returns it as a dictionary of tensors.
        """
        # 1. Load data from the saved .pt file
        pt_path = self.pt_file_paths[idx]
        data = torch.load(pt_path, map_location='cpu')

        # 2. Convert to numpy arrays, the format expected by Albumentations
        # Image: (C, H, W) -> (H, W, C)
        image_np = data['image'].numpy().transpose(1, 2, 0)
        # Label: (1, H, W) -> (H, W)
        label_np = data['label'].numpy().squeeze()

        # 3. Apply the augmentation pipeline if it exists
        if self.transform:
            # The A.Compose pipeline takes a dictionary of numpy arrays
            transformed = self.transform(image=image_np, mask=label_np)
            
            # ToTensorV2 has already converted the image and mask to torch.Tensors
            image_tensor = transformed['image']
            label_tensor = transformed['mask']

            # The label tensor from ToTensorV2 has shape (H,W), add channel dim
            return {
                'image': image_tensor.float(), # Ensure image is float
                'label': label_tensor.unsqueeze(0).long() # Ensure label is long
            }
        else:
            # If no transforms, convert back to tensors manually
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)
            return {'image': image_tensor, 'label': label_tensor}
