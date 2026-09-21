"""
Post-processing script for refining boundary masks using plane fitting.

This script demonstrates how to take a binary segmentation mask, extract boundary
points, and fit planes to generate a cleaned, geometrically consistent slab mask.
This is useful for turning a potentially noisy, voxel-based prediction into a
smooth, planar representation of the slab.

Usage:
    python plane_fitting.py <input_mask_path> <output_mask_path> [--downsample_grid_size G]

Example:
    python plane_fitting.py inference_results/my_tomo_raw_mask.mrc inference_results/my_tomo_fitted_mask.mrc

"""
import argparse
import logging
from pathlib import Path

import mrcfile
import numpy as np
from torch_segment_tomogram_boundaries.postprocess import fit_and_generate_mask

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """Main function to run the plane fitting script."""
    parser = argparse.ArgumentParser(description="Fit planes to a binary mask to create a clean slab.")
    parser.add_argument("input_mask", type=Path, help="Path to the input binary mask (.mrc file).")
    parser.add_argument("output_mask", type=Path, help="Path to save the output fitted mask (.mrc file).")
    parser.add_argument(
        "--downsample_grid_size",
        type=int,
        default=8,
        help="Voxel grid size for downsampling point clouds before plane fitting (default: 8).",
    )
    args = parser.parse_args()

    logging.info(f"Loading mask from: {args.input_mask}")
    try:
        with mrcfile.open(args.input_mask, permissive=True) as mrc:
            binary_mask_np = mrc.data.astype(np.uint8)
            voxel_size = mrc.voxel_size.copy()
    except Exception as e:
        logging.error(f"Failed to read input mask file: {e}")
        return

    try:
        final_mask = fit_and_generate_mask(binary_mask_np, args.downsample_grid_size)
    except (ValueError, RuntimeError) as e:
        logging.error(f"Plane fitting failed: {e}. Aborting.")
        return

    logging.info(f"Saving final fitted mask to: {args.output_mask}")
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    mrcfile.write(args.output_mask, final_mask, voxel_size=voxel_size, overwrite=True)

    logging.info("Plane fitting complete.")


if __name__ == "__main__":
    main()
