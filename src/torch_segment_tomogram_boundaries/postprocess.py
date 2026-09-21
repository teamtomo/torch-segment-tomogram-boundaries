"""Post-processing of binary slab masks via plane fitting.

Takes a binary segmentation mask, extracts its surface points, and fits a plane
to the top and bottom surfaces to produce a geometrically clean slab mask.
"""
import logging

import numpy as np
import pandas as pd
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


def _resolve_device(device):
    """Return a ``torch.device``; ``None`` selects CUDA if available, else CPU.

    MPS is not auto-selected: its scatter/nonzero kernels were measured slower than
    the CPU path for this workload. Pass ``device="mps"`` to force it.
    """
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _surface_coords_zyx(mask: np.ndarray, device) -> np.ndarray:
    """(N, 3) ZYX coordinates of surface voxels (mask minus its 6-connected erosion).

    Equivalent to ``argwhere(mask - scipy.ndimage.binary_erosion(mask))`` (voxels on
    the volume border count as surface), but runs on ``device``.
    """
    import torch

    m = torch.from_numpy(np.ascontiguousarray(mask > 0)).to(device)
    padded = torch.zeros(tuple(n + 2 for n in m.shape), dtype=torch.bool, device=device)
    padded[1:-1, 1:-1, 1:-1] = m
    eroded = (
        m
        & padded[:-2, 1:-1, 1:-1] & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1] & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2] & padded[1:-1, 1:-1, 2:]
    )
    return torch.nonzero(m & ~eroded).cpu().numpy()


def _lstsq_plane(points: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares z = ax + by + c through (N, 3) XYZ points."""
    centre = points.mean(axis=0)
    p = points - centre
    design = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    a, b, c0 = np.linalg.lstsq(design, p[:, 2], rcond=None)[0]
    return float(a), float(b), float(c0 + centre[2] - a * centre[0] - b * centre[1])


def fit_best_plane(
    points: np.ndarray,
    angle_res: int = 60,
    dist_res: int = 200,
    device=None,
    refine_iters: int = 5,
) -> Dict[str, List[float]]:
    """
    Fit the best plane to 3D points: Hough voting, then least-squares refinement.

    Every candidate normal (an ``angle_res`` x ``angle_res`` grid) votes with the
    binned signed distances of all points; voting is batched and runs on ``device``.
    The winning plane is then refined by least squares on its inliers, so the result
    is not limited by the angular/distance resolution of the accumulator.

    Parameters
    ----------
    points : np.ndarray
        Input points as (N, 3) array with columns [x, y, z].
    angle_res : int, default=60
        Angular resolution for normal vector discretization.
    dist_res : int, default=200
        Distance resolution for plane distance discretization.
    device : torch.device or str, optional
        Device for the voting step. Defaults to CUDA when available, else CPU.
    refine_iters : int, default=5
        Maximum inlier re-fitting iterations (0 disables refinement).

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
    import torch

    if len(points) < 50:
        raise ValueError(f"Not enough points ({len(points)}) to fit a plane.")

    device = _resolve_device(device)
    pts64 = np.asarray(points, dtype=np.float64)
    centre = pts64.mean(axis=0)
    pc = pts64 - centre
    # Centred float32 keeps precision and runs on every backend (MPS has no float64).
    p = torch.from_numpy(pc.astype(np.float32)).to(device)

    # Discretize the space of possible plane normals (phi, theta in spherical coords)
    angles = torch.linspace(0, np.pi, angle_res, device=device)
    phi, theta = torch.meshgrid(angles, angles, indexing="xy")
    normals = torch.stack(
        [
            (torch.sin(phi) * torch.cos(theta)).ravel(),
            (torch.sin(phi) * torch.sin(theta)).ravel(),
            torch.cos(phi).ravel(),
        ],
        dim=1,
    )

    # Signed distances lie in [-r, r]; use one shared range so bins are comparable.
    radius = float(p.norm(dim=1).max()) + 1e-6
    bin_width = 2 * radius / dist_res
    n_pts = p.shape[0]
    chunk = max(1, (1 << 26) // n_pts)  # bound the (chunk x N) distance matrix to ~256 MB

    best_votes, best_idx, best_bin = -1.0, 0, 0
    for start in range(0, len(normals), chunk):
        n_chunk = normals[start:start + chunk]
        bins = ((p @ n_chunk.T + radius) / bin_width).long().clamp_(0, dist_res - 1)
        votes = torch.zeros((len(n_chunk), dist_res), dtype=torch.float32, device=device)
        votes.scatter_add_(1, bins.T.contiguous(), torch.ones_like(bins.T, dtype=torch.float32))
        peak, flat = votes.view(-1).max(0)
        if float(peak) > best_votes:
            best_votes = float(peak)
            best_idx, best_bin = start + int(flat) // dist_res, int(flat) % dist_res

    nx, ny, nz = (float(v) for v in normals[best_idx])
    if abs(nz) < 1e-6:
        raise ValueError("Detected a plane nearly vertical to the Z-axis. Plane fitting is unstable.")
    offset = (best_bin + 0.5) * bin_width - radius  # in centred coordinates

    # Equation: nx*x + ny*y + nz*z = d  =>  z = (-nx/nz)*x + (-ny/nz)*y + (d/nz)
    a, b, c = -nx / nz, -ny / nz, offset / nz  # centred coordinates
    for _ in range(refine_iters):
        resid = np.abs(pc[:, 2] - (a * pc[:, 0] + b * pc[:, 1] + c))
        inliers = resid / np.sqrt(a * a + b * b + 1) <= bin_width
        if inliers.sum() < 3:
            break
        a_new, b_new, c_new = _lstsq_plane(pc[inliers])
        converged = max(abs(a_new - a), abs(b_new - b)) < 1e-6 and abs(c_new - c) < 1e-4
        a, b, c = a_new, b_new, c_new
        if converged:
            break

    # Back from centred to original coordinates.
    return {'coefficients': [a, b, c + centre[2] - a * centre[0] - b * centre[1]]}


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


def fit_slab_planes(mask: np.ndarray, downsample_grid_size: int, device=None) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract boundary points, fit planes, and generate a clean slab mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask as a 3D array where non-zero values indicate boundaries.
    downsample_grid_size : int
        Grid size for point cloud downsampling before plane fitting.
    device : torch.device or str, optional
        Device for surface extraction and plane fitting. Defaults to CUDA when
        available, else CPU.

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
    device = _resolve_device(device)
    coords_zyx = _surface_coords_zyx(mask, device)

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
    plane_top = fit_best_plane(top_points_ds, device=device)
    plane_bottom = fit_best_plane(bottom_points_ds, device=device)

    return {'top': plane_top, 'bottom': plane_bottom}


def fit_and_generate_mask(mask: np.ndarray, downsample_grid_size: int, device=None) -> np.ndarray:
    """Fit top/bottom planes to a binary mask and return the clean slab mask.

    Raises ValueError if too few boundary points are found (see `fit_slab_planes`).
    """
    planes = fit_slab_planes(mask, downsample_grid_size, device)
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


