import numpy as np
import pytest
import torch
from scipy.ndimage import binary_erosion

from torch_segment_tomogram_boundaries import postprocess as pp


def _plane_points(n=5000, coeffs=(0.02, -0.01, 40.0), noise=1.0, extent=512, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, extent, (n, 2))
    a, b, c = coeffs
    z = a * xy[:, 0] + b * xy[:, 1] + c + rng.normal(0, noise, n)
    return np.column_stack([xy, z]).astype(np.float32)


def test_fit_best_plane_recovers_coefficients():
    coeffs = (0.02, -0.01, 40.0)
    fit = pp.fit_best_plane(_plane_points(coeffs=coeffs), device="cpu")["coefficients"]
    assert fit == pytest.approx(coeffs, abs=0.05)


def test_fit_best_plane_ignores_outliers():
    pts = _plane_points(n=4000)
    rng = np.random.default_rng(1)
    outliers = rng.uniform([0, 0, 0], [512, 512, 120], (400, 3)).astype(np.float32)
    fit = pp.fit_best_plane(np.vstack([pts, outliers]), device="cpu")["coefficients"]
    assert fit == pytest.approx((0.02, -0.01, 40.0), abs=0.1)


def test_fit_best_plane_needs_enough_points():
    with pytest.raises(ValueError, match="Not enough points"):
        pp.fit_best_plane(_plane_points(n=10), device="cpu")


@pytest.mark.parametrize("border", [False, True])
def test_surface_coords_match_scipy_erosion(border):
    rng = np.random.default_rng(0)
    mask = np.zeros((20, 30, 40), dtype=np.uint8)
    mask[(slice(0, 20) if border else slice(4, 15)), 5:25, 6:35] = 1
    mask ^= (rng.random(mask.shape) < 0.01).astype(np.uint8)
    expected = np.argwhere((mask - binary_erosion(mask)) > 0)
    got = pp._surface_coords_zyx(mask, torch.device("cpu"))
    assert np.array_equal(got, expected)


def test_fit_slab_planes_tilted_slab_cpu():
    shape, thickness, slope = (96, 160, 160), 20.0, 0.1
    z, _, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    centre = 48 + slope * x
    mask = (np.abs(z - centre) <= thickness / 2).astype(np.uint8)
    planes = pp.fit_slab_planes(mask, downsample_grid_size=4, device="cpu")
    for name, offset in (("top", 48 + thickness / 2), ("bottom", 48 - thickness / 2)):
        a, b, c = planes[name]["coefficients"]
        assert (a, b) == pytest.approx((slope, 0.0), abs=0.01)
        assert c == pytest.approx(offset, abs=1.0)
