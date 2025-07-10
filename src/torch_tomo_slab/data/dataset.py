import torchio as tio
import torch
from typing import List, Optional, Tuple
from pathlib import Path

class PTFileDataset(tio.SubjectsDataset):
    """Custom TorchIO dataset that loads .pt files."""

    def __init__(self, pt_file_paths: List[Path], transform: Optional[tio.Transform] = None):
        subjects = []

        for pt_path in pt_file_paths:
            # Load the .pt file
            data = torch.load(pt_path, map_location='cpu')

            # Create TorchIO subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=data['image']),  # (2, H, W, 1)
                label=tio.LabelMap(tensor=data['label']),  # (1, H, W, 1)
                path=str(pt_path),
                **data.get('metadata', {})  # Add metadata as subject attributes
            )
            subjects.append(subject)
        super().__init__(subjects, transform=transform)

