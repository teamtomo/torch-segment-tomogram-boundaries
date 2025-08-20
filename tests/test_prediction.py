# tests/test_prediction.py
import numpy as np
import torch
from torch_tomo_slab.predict import TomoSlabPredictor, fit_best_plane
import pytest


def test_predictor_initialization(trained_checkpoint):
    """Test loading the model from a real checkpoint file."""
    assert trained_checkpoint.exists()
    predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint))
    assert predictor.model is not None
    assert isinstance(predictor.model, torch.nn.Module)
    assert predictor.target_shape_3d is not None


def test_predict_from_numpy_array(trained_checkpoint):
    """Test prediction when the input is a NumPy array."""
    predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint))
    # Input shape must match what the model was trained on to some extent
    input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

    # Use a small slab size for speed
    result_mask = predictor.predict(input_tomo, slab_size=3)

    assert isinstance(result_mask, np.ndarray)
    assert result_mask.shape == input_tomo.shape
    assert result_mask.dtype == np.int8
    # The mask should be binary
    assert np.all(np.isin(result_mask, [0, 1]))


def test_predict_from_file(trained_checkpoint, dummy_mrc_files, tmp_path):
    """Test prediction from an input file, writing to an output file."""
    predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint))
    output_path = tmp_path / "output_mask.mrc"

    predictor.predict(
        input_tomogram=dummy_mrc_files["vol_path"],
        output_path=output_path,
        slab_size=3
    )

    assert output_path.exists()


def test_fit_best_plane_logic():
    """Unit test for the plane fitting algorithm."""
    # Create points that lie perfectly on the plane z = 0.1x + 0.2y + 5
    rng = np.random.default_rng(42)
    x = rng.uniform(-10, 10, size=100)
    y = rng.uniform(-10, 10, size=100)
    z = 0.1 * x + 0.2 * y + 5
    points = np.stack([x, y, z], axis=1)

    # Add some noise
    points += rng.normal(0, 0.01, size=points.shape)

    plane = fit_best_plane(points)
    coeffs = plane['coefficients']

    # Check if the fitted coefficients are close to the true ones
    assert np.isclose(coeffs[0], 0.1, atol=1e-2)
    assert np.isclose(coeffs[1], 0.2, atol=1e-2)
    assert np.isclose(coeffs[2], 5.0, atol=1e-1)

    # Test failure on insufficient points
    with pytest.raises(ValueError, match="Not enough points"):
        fit_best_plane(points[:10])