from pathlib import Path
from typing import Dict, List, Tuple
import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import sys
import argparse

# Add parent src directory to path to allow importing from `torch_tomo_slab`
sys.path.append(str(Path(__file__).resolve().parents[2]))
from torch_tomo_slab import config


def robust_normalization(data: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a 2D torch.Tensor using 5th and 95th percentiles, clipping values.
    This makes the normalization robust to extreme outliers.
    """
    # Move to float for calculations
    data = data.float()
    
    p5 = torch.quantile(data, 0.05)
    p95 = torch.quantile(data, 0.95)
    
    # Check for a flat, zero-variance image
    if p95 - p5 < 1e-6:
        return data - data.mean()

    # Normalize based on the 5-95 percentile range
    data_normalized = (data - torch.median(data)) / (p95 - p5)
    
    # Clip to a fixed range to prevent extreme values from dominating
    return torch.clamp(data_normalized, min=-5, max=5)


def get_device():
    """Dynamically select the best available device."""
    if torch.cuda.is_available():
        print("Using GPU: cuda")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU: mps")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def load_mrc_as_tensor(p: Path) -> Dict:
    """
    Loads an MRC file into a CPU tensor. Normalization is now done later.
    """
    with mrcfile.open(p, permissive=True) as mrc:
        vol_numpy = mrc.data.astype(np.float32)
        angpix = float(mrc.voxel_size.x) if hasattr(mrc.voxel_size, 'x') else 1.0
        
    return {"volume_tensor": torch.from_numpy(vol_numpy), "angpix": angpix}


def local_variance_2d(image: torch.Tensor, kernel_size: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute local mean and variance for a 2D image using torch's convolution.
    Args:
        image: 2D tensor of shape (H, W) that has already been normalized.
        kernel_size: Size of the sliding window
    Returns:
        A tuple of (local_mean, local_variance) 2D tensors.
    """
    if image.dim() != 2:
        raise ValueError("Input image must be a 2D tensor.")

    image_float = image.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

    # Create a uniform kernel for averaging
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size ** 2)
    padding = kernel_size // 2

    # Local mean of the already normalized image is just a blurred version of it
    local_mean = F.conv2d(image_float, kernel, padding=padding).squeeze()

    # Variance = E[X²] - (E[X])²
    # We compute variance on the *normalized* image.
    local_mean_sq = F.conv2d(image_float ** 2, kernel, padding=padding).squeeze()
    local_variance = torch.clamp(local_mean_sq - local_mean ** 2, min=0)

    # The first channel is now the normalized image itself, not its local mean.
    return image, local_variance


def find_data_pairs(vol_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    """Finds matching volume and label files based on their base names."""
    pairs = []
    vol_files = sorted(list(vol_dir.glob("*.mrc")))
    
    label_files_map = {
        p.stem: p for p in label_dir.glob("*.mrc")
    }

    for vol_path in vol_files:
        base_name = vol_path.stem
        if base_name in label_files_map:
            pairs.append((vol_path, label_files_map[base_name]))
        else:
            print(f"Warning: No matching label found for volume '{vol_path.name}'. Skipping.")

    return pairs

def process_and_save_slices(
    data_pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    num_sections: int,
    kernel_size: int,
    device: torch.device,
    set_name: str
):
    """Processes a list of volume pairs and saves their slices to a directory."""
    print(f"\n--- Processing {len(data_pairs)} volumes for the {set_name} set ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for vol_file, label_file in data_pairs:
        print(f"  - Processing {vol_file.name}")

        # --- Load raw data, normalization happens after projection ---
        vol_data_dict = load_mrc_as_tensor(vol_file)
        label_data_dict = load_mrc_as_tensor(label_file)

        volume = vol_data_dict['volume_tensor'].to(device)
        label = label_data_dict['volume_tensor'].to(device)

        for i in range(num_sections):
            axis_to_slice = torch.randint(1, 3, (1,)).item()
            slice_dim_size = volume.shape[axis_to_slice]
            if slice_dim_size < 3:
                print(f"  Warning: Cannot extract 3-slice window from axis {axis_to_slice} with size {slice_dim_size}. Skipping.")
                continue
            slice_idx = torch.randint(1, slice_dim_size - 1, (1,)).item()

            if axis_to_slice == 1:
                view = "zx"
                vol_slice = volume[:, slice_idx - 1:slice_idx + 2, :]
                ortho_img_raw = torch.sum(vol_slice, dim=1)
                ortho_label = label[:, slice_idx, :]
            else: # axis_to_slice == 2
                view = "zy"
                vol_slice = volume[:, :, slice_idx - 1:slice_idx + 2]
                ortho_img_raw = torch.sum(vol_slice, dim=2)
                ortho_label = label[:, :, slice_idx]

            # --- MODIFIED: Normalize the 2D projected image ---
            ortho_img_normalized = robust_normalization(ortho_img_raw)

            # --- MODIFIED: Calculate variance on the normalized image ---
            # The first channel is now the normalized image itself.
            ch1_img, ch2_var = local_variance_2d(ortho_img_normalized, kernel_size=kernel_size)
            two_channel_input = torch.stack([ch1_img, ch2_var], dim=0)

            image_to_save = two_channel_input.cpu()  # Shape: (2, H, W)
            label_to_save = ortho_label.cpu()  # Shape: (1, H, W)

            save_data = {
                'image': image_to_save,
                'label': label_to_save,
                'metadata': {
                    'source_file': vol_file.name,
                    'view_axis': view,
                    'slice_index': slice_idx,
                    'angpix': vol_data_dict['angpix']
                }
            }
            save_path = output_dir / f"{vol_file.stem}_{view}_{slice_idx}.pt"
            torch.save(save_data, save_path)


def main(args):
    """Main execution function to prepare 2D data for training."""
    device = get_device()

    data_dir = Path(args.vol_dir)
    label_dir = Path(args.mask_dir)

    all_data_pairs = find_data_pairs(data_dir, label_dir)
    if not all_data_pairs:
        print(f"Error: No matching volume/label pairs found between '{data_dir}' and '{label_dir}'.")
        return

    train_pairs, val_pairs = train_test_split(
        all_data_pairs,
        test_size=config.VALIDATION_FRACTION,
        random_state=42,
        shuffle=True
    )
    print(f"\nSplitting {len(all_data_pairs)} volumes into:")
    print(f"  - Training set:   {len(train_pairs)} volumes")
    print(f"  - Validation set: {len(val_pairs)} volumes")

    process_and_save_slices(
        data_pairs=train_pairs,
        output_dir=Path(config.TRAIN_DATA_DIR),
        num_sections=args.num_sections,
        kernel_size=args.kernel_size,
        device=device,
        set_name="training"
    )

    process_and_save_slices(
        data_pairs=val_pairs,
        output_dir=Path(config.VAL_DATA_DIR),
        num_sections=args.num_sections,
        kernel_size=args.kernel_size,
        device=device,
        set_name="validation"
    )

    print("\n--- Data preparation finished successfully. ---")
    print("Data was normalized using a robust percentile-based method on 2D projections.")
    print(f"Training data saved to: {config.TRAIN_DATA_DIR}")
    print(f"Validation data saved to: {config.VAL_DATA_DIR}")


def main_cli():
    parser = argparse.ArgumentParser(description="Generate 2D training samples from 3D tomograms and masks.")
    parser.add_argument("--vol_dir", type=str, default=config.REFERENCE_TOMOGRAM_DIR,
                        help="Directory with input volume files (*_vol.mrc).")
    parser.add_argument("--mask_dir", type=str, default=config.MASK_OUTPUT_DIR,
                        help="Directory with input mask/label files (*_label.mrc).")
    parser.add_argument("--num_sections", type=int, default=config.NUM_SECTIONS_PER_VOLUME,
                        help="Number of 2D sections to extract per volume.")
    parser.add_argument("--kernel_size", type=int, default=config.LOCAL_VARIANCE_KERNEL_SIZE,
                        help="Kernel size for local variance calculation.")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    main_cli()
