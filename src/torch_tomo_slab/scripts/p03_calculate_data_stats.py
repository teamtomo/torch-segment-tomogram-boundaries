# scripts/compute_dataset_stats.py
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch_tomo_slab.data.dataset import PTFileDataset

def compute_stats(dataset: PTFileDataset):
    """Computes mean and std for a dataset of 2-channel images."""
    num_pixels = 0
    channel_sum = torch.zeros(2)
    channel_sum_sq = torch.zeros(2)

    for sample in tqdm(dataset, desc="Calculating stats"):
        image = sample['image'].to(torch.float32) # C, H, W
        channel_sum += torch.sum(image, dim=[1, 2])
        channel_sum_sq += torch.sum(image**2, dim=[1, 2])
        num_pixels += image.shape[1] * image.shape[2]

    mean = channel_sum / num_pixels
    # Var(X) = E[X^2] - (E[X])^2
    var = (channel_sum_sq / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    return mean, std

if __name__ == "__main__":
    data_dir = Path("/home/pranav/data/training/torch-tomo-slab/prepared_data/labels/") # Adjust path
    all_pt_files = sorted(list(data_dir.glob("*.pt")))
    
    # IMPORTANT: Only use training files to compute stats
    # You would typically split your files first. For simplicity, we use all here.
    # In practice: train_files, _ = train_test_split(...)
    dataset = PTFileDataset(pt_file_paths=all_pt_files)
    
    mean, std = compute_stats(dataset)
    
    print(f"Dataset Mean: {mean.tolist()}")
    print(f"Dataset Std:  {std.tolist()}")
