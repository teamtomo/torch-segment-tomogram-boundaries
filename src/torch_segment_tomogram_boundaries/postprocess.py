"""Post-processing of binary slab masks via plane fitting.

Takes a binary segmentation mask, extracts its surface points, and fits a plane
to the top and bottom surfaces to produce a geometrically clean slab mask.
"""
import logging

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)


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


def fit_slab_planes(mask: np.ndarray, downsample_grid_size: int) -> Dict[str, Dict[str, List[float]]]:
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
    Dict[str, Dict[str, List[float]]]
        ``{'top': {...}, 'bottom': {...}}``, each with 'coefficients' [a, b, c] for
        the plane z = ax + by + c (x, y, z in voxels).

    Raises
    ------
    ValueError
        If insufficient boundary points (<1000) are found.
    """
    log.info("Extracting boundary points from the binarized mask...")
    # Erode the mask to find the surface voxels
    surface = mask - binary_erosion(mask)
    coords_zyx = np.argwhere(surface > 0)

    if len(coords_zyx) < 1000:
        raise ValueError(f"Not enough boundary points ({len(coords_zyx)}) found to reliably fit planes.")

    # Convert ZYX coordinates to XYZ for plane fitting
    points_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float32)

    # Split points into top and bottom surfaces at the median along the slab normal
    # (smallest-variance direction of the surface points), so tilted slabs split
    # correctly. For a flat slab this is equivalent to splitting at the median Z.
    centred = points_xyz - points_xyz.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    if normal[2] < 0:
        normal = -normal
    height = centred @ normal
    is_top = height >= np.median(height)
    top_points = points_xyz[is_top]
    bottom_points = points_xyz[~is_top]

    # Downsample point clouds to make plane fitting faster and more robust
    top_points_ds = downsample_points(top_points, grid_size=downsample_grid_size)
    bottom_points_ds = downsample_points(bottom_points, grid_size=downsample_grid_size)
    log.info(f"Downsampled top surface from {len(top_points)} to {len(top_points_ds)} points.")
    log.info(f"Downsampled bottom surface from {len(bottom_points)} to {len(bottom_points_ds)} points.")

    # Fit a plane to each point cloud
    log.info("Fitting top and bottom planes...")
    plane_top = fit_best_plane(top_points_ds)
    plane_bottom = fit_best_plane(bottom_points_ds)

    return {'top': plane_top, 'bottom': plane_bottom}


def fit_and_generate_mask(mask: np.ndarray, downsample_grid_size: int) -> np.ndarray:
    """Fit top/bottom planes to a binary mask and return the clean slab mask.

    Raises ValueError if too few boundary points are found (see `fit_slab_planes`).
    """
    planes = fit_slab_planes(mask, downsample_grid_size)
    log.info("Generating final mask from fitted planes.")
    return generate_mask_from_planes(planes, mask.shape)


def perpendicular_thickness_map(
    planes: Dict[str, Dict[str, List[float]]], mask: np.ndarray
) -> np.ndarray:
    """Perpendicular top-bottom distance (voxels) for every occupied (Y, X) column.

    For each column in the mask footprint, the point on each fitted plane is
    measured to the *other* plane along that plane's normal, and the two distances
    are averaged so the result does not depend on which plane is used as reference.
    """
    at, bt, ct = planes['top']['coefficients']
    ab, bb, cb = planes['bottom']['coefficients']
    yy, xx = np.nonzero((mask > 0).any(axis=0))
    z_top = at * xx + bt * yy + ct
    z_bot = ab * xx + bb * yy + cb
    dz = np.abs(z_top - z_bot)
    return 0.5 * dz * (1.0 / np.sqrt(at**2 + bt**2 + 1) + 1.0 / np.sqrt(ab**2 + bb**2 + 1))


