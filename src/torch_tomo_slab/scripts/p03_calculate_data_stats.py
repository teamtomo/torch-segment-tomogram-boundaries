import torch
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[2]))
from torch_tomo_slab import constants
from torch_tomo_slab.data.dataset import PTFileDataset

def verify_stats(dataset: PTFileDataset):


    num_samples_to_check = min(len(dataset), 500)
    indices = np.random.choice(len(dataset), num_samples_to_check, replace=False)

    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []

    print(f"Verifying stats on {num_samples_to_check} random samples...")
    for i in tqdm(indices):
        sample = dataset[i]
        image = sample['image'].to(torch.float32)


        ch0 = image[0, :, :]
        all_mins.append(ch0.min().item())
        all_maxs.append(ch0.max().item())
        all_means.append(ch0.mean().item())
        all_stds.append(ch0.std().item())

    print("\n--- Normalization Verification Results ---")
    print("These statistics are for the first channel (normalized tomogram slices).")
    print(f"Overall Mean: {np.mean(all_means):.4f} (Expected to be near 0)")
    print(f"Overall Std Dev:  {np.mean(all_stds):.4f}")
    print(f"Overall Min:  {np.min(all_mins):.4f} (Clipping was at -5)")
    print(f"Overall Max:  {np.max(all_maxs):.4f} (Clipping was at +5)")


if __name__ == "__main__":
    train_data_dir = constants.TRAIN_DATA_DIR

    print(f"Loading training data from: {train_data_dir}")

    train_pt_files = sorted(list(train_data_dir.glob("*.pt")))

    if not train_pt_files:
        raise FileNotFoundError(
            f"No training files found in {train_data_dir}. "
            "Please run the p02_data_preparation.py script first."
        )

    dataset = PTFileDataset(pt_file_paths=train_pt_files)

    verify_stats(dataset)
