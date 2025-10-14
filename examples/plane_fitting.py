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
from typing import Dict, List, Tuple

import mrcfile
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def downsample_points(points: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Reduce point cloud density using voxel grid downsampling.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud as (N, 3) array with columns [x, y, z].
    grid_size : int
        Size of the voxel grid for downsampling.

    Returns
    -------
    np.ndarray
        Downsampled points as (M, 3) array where M <= N.
    """
    if points.shape[0] == 0:
        return points
    voxel_indices = np.floor(points / grid_size).astype(np.int32)
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['voxel_x'], df['voxel_y'], df['voxel_z'] = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    return df.groupby(['voxel_x', 'voxel_y', 'voxel_z'])[['x', 'y', 'z']].mean().to_numpy()


def fit_best_plane(points: np.ndarray, angle_res: int = 180, dist_res: int = 200) -> Dict[str, List[float]]:
    """
    Fit the best plane to 3D points using a Hough Transform-like method.

    Parameters
    ----------
    points : np.ndarray
        Input points as (N, 3) array with columns [x, y, z].
    angle_res : int, default=180
        Angular resolution for normal vector discretization.
    dist_res : int, default=200
        Distance resolution for plane distance discretization.

    Returns
    -------
    Dict[str, List[float]]
        Dictionary with 'coefficients' key containing plane equation [a, b, c]
        where z = ax + by + c.

    Raises
    ------
    ValueError
        If insufficient points (<50) or plane is nearly vertical to Z-axis.
    """
    if len(points) < 50:
        raise ValueError(f"Not enough points ({len(points)}) to fit a plane.")

    # Discretize the space of possible plane normals (phi, theta in spherical coords)
    phis = np.linspace(0, np.pi, angle_res)
    thetas = np.linspace(0, np.pi, angle_res)
    phi_grid, theta_grid = np.meshgrid(phis, thetas)

    # Convert spherical to Cartesian coordinates for normals
    nx, ny, nz = np.sin(phi_grid) * np.cos(theta_grid), np.sin(phi_grid) * np.sin(theta_grid), np.cos(phi_grid)
    normals = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)

    # Project points onto each normal to get distances
    dists = np.dot(points, normals.T)
    min_dist, max_dist = dists.min(), dists.max()

    # Create an accumulator array for voting
    accumulator = np.zeros((len(normals), dist_res), dtype=np.uint32)
    dist_bins = np.linspace(min_dist, max_dist, dist_res)

    # Vote for the best (normal, distance) pair
    for i in range(len(normals)):
        hist, _ = np.histogram(dists[:, i], bins=dist_res, range=(min_dist, max_dist))
        accumulator[i, :] = hist

    # Find the peak in the accumulator
    normal_idx, dist_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    best_normal, best_dist = normals[normal_idx], dist_bins[dist_idx]

    # Convert plane parameters to z = ax + by + c form
    nx, ny, nz = best_normal
    if abs(nz) < 1e-6:
        raise ValueError("Detected a plane nearly vertical to the Z-axis. Plane fitting is unstable.")

    # Equation: nx*x + ny*y + nz*z = d  =>  z = (-nx/nz)*x + (-ny/nz)*y + (d/nz)
    return {'coefficients': [-nx / nz, -ny / nz, best_dist / nz]}


def generate_mask_from_planes(planes: Dict[str, Dict[str, List[float]]], volume_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Generate a binary mask volume from top and bottom plane equations.

    Parameters
    ----------
    planes : Dict[str, Dict[str, List[float]]]
        Dictionary with 'top' and 'bottom' keys, each containing plane coefficients.
    volume_shape : Tuple[int, int, int]
        Target volume shape as (depth, height, width).

    Returns
    -------
    np.ndarray
        Binary mask as a 3D array where 1 indicates the region between the planes.
    """
    Nz, Ny, Nx = volume_shape
    coef_b, coef_t = planes['bottom']['coefficients'], planes['top']['coefficients']

    # Create coordinate grids
    yy, xx = np.mgrid[0:Ny, 0:Nx]

    # Calculate z-coordinates for each plane at every (x, y)
    z_bottom = (coef_b[0] * xx + coef_b[1] * yy + coef_b[2])
    z_top = (coef_t[0] * xx + coef_t[1] * yy + coef_t[2])

    # Ensure top plane is always above bottom plane
    min_plane, max_plane = np.minimum(z_bottom, z_top), np.maximum(z_bottom, z_top)

    # Create a z-coordinate grid and check if each voxel is between the planes
    zz = np.arange(Nz)[:, np.newaxis, np.newaxis]
    mask = ((zz >= min_plane) & (zz <= max_plane)).astype(np.int8)
    return mask


def fit_and_generate_mask(mask: np.ndarray, downsample_grid_size: int) -> np.ndarray:
    """
    Extract boundary points, fit planes, and generate a clean slab mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask as a 3D array where non-zero values indicate boundaries.
    downsample_grid_size : int
        Grid size for point cloud downsampling before plane fitting.

    Returns
    -------
    np.ndarray
        Clean binary mask generated from the fitted top and bottom planes.

    Raises
    ------
    ValueError
        If insufficient boundary points (<1000) are found.
    """
    logging.info("Extracting boundary points from the binarized mask...")
    # Erode the mask to find the surface voxels
    surface = mask - binary_erosion(mask)
    coords_zyx = np.argwhere(surface > 0)

    if len(coords_zyx) < 1000:
        raise ValueError(f"Not enough boundary points ({len(coords_zyx)}) found to reliably fit planes.")

    # Convert ZYX coordinates to XYZ for plane fitting
    points_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float32)

    # Split points into top and bottom surfaces based on the median Z value
    z_median = np.median(points_xyz[:, 2])
    top_points = points_xyz[points_xyz[:, 2] >= z_median]
    bottom_points = points_xyz[points_xyz[:, 2] < z_median]

    # Downsample point clouds to make plane fitting faster and more robust
    top_points_ds = downsample_points(top_points, grid_size=downsample_grid_size)
    bottom_points_ds = downsample_points(bottom_points, grid_size=downsample_grid_size)
    logging.info(f"Downsampled top surface from {len(top_points)} to {len(top_points_ds)} points.")
    logging.info(f"Downsampled bottom surface from {len(bottom_points)} to {len(bottom_points_ds)} points.")

    # Fit a plane to each point cloud
    logging.info("Fitting top and bottom planes...")
    plane_top = fit_best_plane(top_points_ds)
    plane_bottom = fit_best_plane(bottom_points_ds)

    # Generate the final mask from the two fitted planes
    logging.info("Generating final mask from fitted planes.")
    return generate_mask_from_planes({'top': plane_top, 'bottom': plane_bottom}, mask.shape)


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
