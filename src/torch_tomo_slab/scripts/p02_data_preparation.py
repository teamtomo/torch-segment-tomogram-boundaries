from pathlib import Path
from typing import Dict, List, Tuple
import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
import sys
import argparse

# Add parent src directory to path to allow importing from `torch_tomo_slab`
sys.path.append(str(Path(__file__).resolve().parents[2]))
from torch_tomo_slab import config


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
    """Loads an MRC file into a CPU tensor."""
    with mrcfile.open(p, permissive=True) as mrc:
        vol_numpy = mrc.data.astype(np.float32)
        # It's safer to check if voxel_size exists and has an 'x' attribute
        angpix = float(mrc.voxel_size.x) if hasattr(mrc.voxel_size, 'x') else 1.0
    return {"volume_tensor": torch.from_numpy(vol_numpy), "angpix": angpix}


def local_variance_2d(image: torch.Tensor, kernel_size: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute local mean and variance for a 2D image using torch's convolution.
    Args:
        image: 2D tensor of shape (H, W)
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

    # Calculate local mean and local mean of the squared image
    local_mean = F.conv2d(image_float, kernel, padding=padding)
    local_mean_sq = F.conv2d(image_float ** 2, kernel, padding=padding)

    # Variance = E[X²] - (E[X])²
    # Clamp at zero to avoid small negative values due to floating point inaccuracies
    local_variance = torch.clamp(local_mean_sq - local_mean ** 2, min=0)

    return local_mean.squeeze(), local_variance.squeeze()


def find_data_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Finds matching volume and label files in the same directory based on their base names."""
    pairs = []

    # Find all volume files (ending with 'vol')
    vol_files = sorted(list(data_dir.glob("*_vol.mrc")))

    # Create a mapping of base names to label files
    label_files = {}
    for label_path in data_dir.glob("*_label.mrc"):
        # Extract base name by removing '_label' suffix
        base_name = label_path.stem.replace('_label', '')
        label_files[base_name] = label_path

    for vol_path in vol_files:
        base_name = vol_path.stem.replace('_vol', '')

        if base_name in label_files:
            pairs.append((vol_path, label_files[base_name]))
        else:
            print(f"Warning: No matching label found for volume '{vol_path.name}'. Skipping.")

    return pairs


def main(args):
    """Main execution function to prepare 2D data for training."""
    device = get_device()

    # Get paths from args (which default to config values)
    data_dir = Path(args.vol_dir)
    output_dir = Path(args.out_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all corresponding volume/label pairs
    data_pairs = find_data_pairs(data_dir)
    if not data_pairs:
        print(f"Error: No matching volume/label pairs found in '{data_dir}'.")
        print("Please ensure your files are named (e.g., 'Tomo_1_vol.mrc' and 'Tomo_1_label.mrc').")
        return

    for vol_file, label_file in data_pairs:
        print(f"\n--- Processing {vol_file.name} ---")

        # Load data to CPU first to conserve GPU memory
        vol_data_dict = load_mrc_as_tensor(vol_file)
        label_data_dict = load_mrc_as_tensor(label_file)

        # Move tensors to the target device for processing
        volume = vol_data_dict['volume_tensor'].to(device)  # Shape: (z, y, x)
        label = label_data_dict['volume_tensor'].to(device)  # Shape: (z, y, x)

        for i in range(args.num_sections):
            # Randomly choose an axis to slice (1 for y-axis, 2 for x-axis)
            axis_to_slice = torch.randint(1, 3, (1,)).item()
            slice_dim_size = volume.shape[axis_to_slice]
            if slice_dim_size < 3:
                print(
                    f"Warning: Cannot extract 3-slice window from axis {axis_to_slice} with size {slice_dim_size}. Skipping this sample.")
                continue
            slice_idx = torch.randint(1, slice_dim_size - 1, (1,)).item()

            if axis_to_slice == 1:  # Slice along y-axis -> creates a z-x plane
                view = "zx"
                vol_slice = volume[:, slice_idx - 1:slice_idx + 2, :]  # Shape: (z, 3, x)
                ortho_img = torch.sum(vol_slice, dim=1)  # Sum across the 3 slices -> (z, x)
                ortho_label = label[:, slice_idx, :]  # Take middle slice for label -> (z, x)
            else:  # axis_to_slice == 2, slice along x-axis -> creates a z-y plane
                view = "zy"
                vol_slice = volume[:, :, slice_idx - 1:slice_idx + 2]  # Shape: (z, y, 3)
                ortho_img = torch.sum(vol_slice, dim=2)  # Sum across the 3 slices -> (z, y)
                ortho_label = label[:, :, slice_idx]  # Take middle slice for label -> (z, y)

            # Compute local variance on the 2D orthogonal image
            ortho_img_mean, ortho_img_var = local_variance_2d(ortho_img, kernel_size=args.kernel_size)

            # Stack mean and variance to create the 2-channel input
            two_channel_input = torch.stack([ortho_img_mean, ortho_img_var], dim=0)  # Shape: (2, H, W)

            # Prepare for saving (add singleton 'depth' dimension for torchio)
            # Tensors are moved to CPU before saving
            image_to_save = two_channel_input.unsqueeze(-1).cpu()  # Shape: (2, H, W, 1)
            label_to_save = ortho_label.unsqueeze(0).unsqueeze(-1).cpu()  # Shape: (1, H, W, 1)

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

            # Use a more descriptive filename
            save_path = output_dir / f"{vol_file.stem}_{view}_{slice_idx}.pt"
            torch.save(save_data, save_path)

    print("\n--- Data preparation finished successfully. ---")

def main_cli():
    parser = argparse.ArgumentParser(description="Generate 2D training samples from 3D tomograms and masks.")
    parser.add_argument("--vol_dir", type=str, default=config.REFERENCE_TOMOGRAM_DIR,
                        help="Directory with input volume files (.mrc).")
    parser.add_argument("--mask_dir", type=str, default=config.MASK_OUTPUT_DIR,
                        help="Directory with input mask/label files (.mrc).")
    parser.add_argument("--out_dir", type=str, default=config.PREPARED_DATA_DIR,
                        help="Directory to save the output .pt files.")
    parser.add_argument("--num_sections", type=int, default=config.NUM_SECTIONS_PER_VOLUME,
                        help="Number of 2D sections to extract per volume.")
    parser.add_argument("--kernel_size", type=int, default=config.LOCAL_VARIANCE_KERNEL_SIZE,
                        help="Kernel size for local variance calculation.")

    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    main_cli()