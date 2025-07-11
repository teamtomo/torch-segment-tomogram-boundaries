import torch
from typing import List, Optional, Dict
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T


class PTFileDataset(Dataset):
    """
    A standard PyTorch Dataset to load 2D image-label pairs from pre-processed .pt files.
    It ensures the output tensors are 3D (C, H, W) for 2D processing.
    """

    def __init__(self, pt_file_paths: List[Path], transform: Optional[T.Compose] = None):
        super().__init__()
        self.pt_file_paths = pt_file_paths
        self.transform = transform

    def __len__(self):
        return len(self.pt_file_paths)

    def __getitem__(self, idx: int) -> Dict:
        """
        Loads a .pt file, processes the tensors, and applies transforms.
        """
        pt_path = self.pt_file_paths[idx]
        data = torch.load(pt_path, map_location='cpu')

        image_tensor = data['image']
        label_tensor = data['label']

        # --- THIS IS THE FIX ---
        # Defensively remove all trailing singleton dimensions until the tensor is 3D.
        # This makes the dataset robust to how the .pt files were saved.
        while image_tensor.ndim > 3:
            image_tensor = image_tensor.squeeze(-1)
        while label_tensor.ndim > 3:
            label_tensor = label_tensor.squeeze(-1)

        # Ensure tensors are at least 3D (for cases where C=1 was squeezed)
        if image_tensor.ndim == 2:  # H, W -> 1, H, W
            image_tensor = image_tensor.unsqueeze(0)
        if label_tensor.ndim == 2:
            label_tensor = label_tensor.unsqueeze(0)

        sample = {
            'image': image_tensor,  # Should be (C, H, W)
            'label': label_tensor,  # Should be (1, H, W)
            'file_path': str(pt_path),
            **data.get('metadata', {})
        }

        if self.transform:
            sample = self.transform(sample)

        return sample