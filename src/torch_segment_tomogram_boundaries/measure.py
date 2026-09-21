"""Measure slab thickness (perpendicular top-bottom distance) from binary masks."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from torch_segment_tomogram_boundaries.postprocess import (
    fit_slab_planes,
    perpendicular_thickness_map,
)

FIELDS = [
    "name", "voxel_size_A", "mean_vox", "median_vox", "std_vox", "min_vox", "max_vox",
    "mean_nm", "median_nm", "std_nm", "coverage",
]


def measure_thickness(
    mask: np.ndarray,
    voxel_size: Optional[float] = None,
    planes: Optional[Dict[str, Dict[str, List[float]]]] = None,
    downsample_grid_size: int = 8,
) -> dict:
    """Measure slab thickness as the perpendicular distance between top and bottom.

    Planes are fitted to the top and bottom surfaces of the (Z, Y, X) binary mask
    (see `postprocess.fit_slab_planes`), unless already-fitted ``planes`` are passed
    in, in which case no fitting is done. For every (Y, X) column in the mask
    footprint the distance between the two planes, measured along the plane normals,
    is computed. Values are in voxels and, when ``voxel_size`` (Angstrom) is
    positive, in nm. Voxels are assumed isotropic.

    ``coverage`` is the fraction of columns containing any mask. If the planes
    cannot be fitted (empty or tiny mask) all statistics are NaN.
    """
    nan = float("nan")
    keys = ("mean", "median", "std", "min", "max")
    coverage = float((mask > 0).any(axis=0).mean())
    try:
        if planes is None:
            planes = fit_slab_planes((mask > 0).astype(np.uint8), downsample_grid_size)
        d = perpendicular_thickness_map(planes, mask)
        stats = dict(zip(keys, (d.mean(), np.median(d), d.std(), d.min(), d.max())))
    except ValueError:
        stats = dict.fromkeys(keys, nan)

    has_size = voxel_size is not None and voxel_size > 0
    nm = (voxel_size / 10.0) if has_size else None  # Angstrom -> nm per voxel
    return {
        "voxel_size_A": float(voxel_size) if has_size else nan,
        "mean_vox": stats["mean"], "median_vox": stats["median"], "std_vox": stats["std"],
        "min_vox": stats["min"], "max_vox": stats["max"],
        "mean_nm": stats["mean"] * nm if nm else nan,
        "median_nm": stats["median"] * nm if nm else nan,
        "std_nm": stats["std"] * nm if nm else nan,
        "coverage": coverage,
    }


def write_thickness_csv(rows: list[dict], path: Path) -> None:
    """Write measurement rows (each with a ``name`` key) to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in row.items()})
