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


def main(args):
    """Main execution function to prepare 2D data for training."""
    device = get_device()

    data_dir = Path(args.vol_dir)
    label_dir = Path(args.mask_dir)
    output_dir = Path(args.out_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    data_pairs = find_data_pairs(data_dir, label_dir)
    if not data_pairs:
        print(f"Error: No matching volume/label pairs found between '{data_dir}' and '{label_dir}'.")
        print("Please ensure your files are named (e.g., 'Tomo_1_vol.mrc' and 'Tomo_1_label.mrc').")
        return

    for vol_file, label_file in data_pairs:
        print(f"\n--- Processing {vol_file.name} ---")

        vol_data_dict = load_mrc_as_tensor(vol_file)
        label_data_dict = load_mrc_as_tensor(label_file)

        volume = vol_data_dict['volume_tensor'].to(device)
        label = label_data_dict['volume_tensor'].to(device)

        for i in range(args.num_sections):
            axis_to_slice = torch.randint(1, 3, (1,)).item()
            slice_dim_size = volume.shape[axis_to_slice]
            if slice_dim_size < 3:
                print(f"Warning: Cannot extract 3-slice window from axis {axis_to_slice} with size {slice_dim_size}. Skipping.")
                continue
            slice_idx = torch.randint(1, slice_dim_size - 1, (1,)).item()

            if axis_to_slice == 1:
                view = "zx"
                vol_slice = volume[:, slice_idx - 1:slice_idx + 2, :]
                ortho_img = torch.sum(vol_slice, dim=1)
                ortho_label = label[:, slice_idx, :]
            else:
                view = "zy"
                vol_slice = volume[:, :, slice_idx - 1:slice_idx + 2]
                ortho_img = torch.sum(vol_slice, dim=2)
                ortho_label = label[:, :, slice_idx]

            ortho_img_mean, ortho_img_var = local_variance_2d(ortho_img, kernel_size=args.kernel_size)
            two_channel_input = torch.stack([ortho_img_mean, ortho_img_var], dim=0)

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

    print("\n--- Data preparation finished successfully. ---")


def main_cli():
    parser = argparse.ArgumentParser(description="Generate 2D training samples from 3D tomograms and masks.")
    parser.add_argument("--vol_dir", type=str, default=config.REFERENCE_TOMOGRAM_DIR,
                        help="Directory with input volume files (*_vol.mrc).")
    parser.add_argument("--mask_dir", type=str, default=config.MASK_OUTPUT_DIR,
                        help="Directory with input mask/label files (*_label.mrc).")
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
