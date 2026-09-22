# tests/test_prediction.py
import warnings

import mrcfile
import numpy as np
import pytest
import torch

from torch_segment_tomogram_boundaries.predict import TomoSlabPredictor


def test_predictor_initialization(trained_checkpoint):
    """Test loading the model from a real checkpoint file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert trained_checkpoint.exists()
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint), compile_model=False)
        assert predictor.model is not None
        assert isinstance(predictor.model, torch.nn.Module)
        assert predictor.target_shape_3d is not None


def test_predict_from_numpy_array(trained_checkpoint):
    """Test prediction when the input is a NumPy array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint), compile_model=False)
        # Input shape must match what the model was trained on to some extent
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        # Use a small slab size for speed
        result_mask = predictor.predict_binary(input_tomo, slab_size=3)

        assert isinstance(result_mask, np.ndarray)
        assert result_mask.shape == input_tomo.shape
        assert result_mask.dtype == np.uint8
        # The mask should be binary
        assert np.all(np.isin(result_mask, [0, 1]))


def test_predict_from_file(trained_checkpoint, dummy_mrc_files, tmp_path):
    """Test prediction from an input file, writing to an output file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint), compile_model=False)
        output_path = tmp_path / "output_mask.mrc"
        input_path = dummy_mrc_files["vol_path"]

        with mrcfile.open(input_path, permissive=True) as mrc:
            input_shape = mrc.data.shape

        mask = predictor.predict_binary(
            input_tomogram=input_path,
            slab_size=3
        )
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(mask)

        assert output_path.exists()
        with mrcfile.open(output_path) as mrc:
            assert mrc.data.shape == input_shape
            # MRC has no uint8 mode; mrcfile widens it on write.
            assert mrc.data.dtype in (np.uint8, np.uint16)
            assert np.all(np.isin(mrc.data, [0, 1]))


def test_predict_probabilities_output(trained_checkpoint):
    """Test the output of the predict_probabilities method."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint), compile_model=False)
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        prob_map = predictor.predict_probabilities(input_tomo, slab_size=3)

        assert isinstance(prob_map, np.ndarray)
        assert prob_map.shape == input_tomo.shape
        assert prob_map.dtype == np.float32
        assert prob_map.min() >= 0.0
        assert prob_map.max() <= 1.0


def test_predict_with_no_slab_blending(trained_checkpoint):
    """Test prediction with slab_size=1 (no blending)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint), compile_model=False)
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        # slab_size=1 disables blending
        result_mask = predictor.predict_binary(input_tomo, slab_size=1)

        assert result_mask.shape == input_tomo.shape
        assert np.all(np.isin(result_mask, [0, 1]))


def test_predict_with_smoothing(trained_checkpoint):
    """Test prediction with Gaussian smoothing enabled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint), compile_model=False)
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        # Use a non-zero sigma to enable smoothing
        prob_map = predictor.predict_probabilities(input_tomo, slab_size=3, smoothing_sigma=1.5)

        assert prob_map.shape == input_tomo.shape
        assert prob_map.dtype == np.float32
        # A simple check: a smoothed output should not be identical to a non-smoothed one
        prob_map_no_smooth = predictor.predict_probabilities(input_tomo, slab_size=3, smoothing_sigma=None)
        assert not np.allclose(prob_map, prob_map_no_smooth, atol=1e-5)


def test_predict_with_compile_disabled(trained_checkpoint):
    """Test that prediction works with torch.compile disabled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Initialize predictor with compile_model=False
        predictor = TomoSlabPredictor(
            model_checkpoint_path=str(trained_checkpoint),
            compile_model=False
        )
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        result_mask = predictor.predict_binary(input_tomo, slab_size=3)

        assert result_mask.shape == input_tomo.shape
        assert np.all(np.isin(result_mask, [0, 1]))

def test_predictor_explicit_device(trained_checkpoint):
    """An explicit device is honoured and the default falls back to CUDA/CPU."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(str(trained_checkpoint), compile_model=False, device="cpu")
        assert predictor.device == torch.device("cpu")
        assert next(predictor.model.parameters()).device.type == "cpu"

        default = TomoSlabPredictor(str(trained_checkpoint), compile_model=False)
        assert default.device.type == ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("slab_size", [1, 3])
@pytest.mark.parametrize("smoothing_sigma", [None, 1.5])
def test_parallel_axes_matches_serial(trained_checkpoint, slab_size, smoothing_sigma):
    """parallel_axes=True must give the same result as the default serial path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(str(trained_checkpoint), compile_model=False)
        input_tomo = np.random.RandomState(0).rand(32, 64, 64).astype(np.float32)

        serial = predictor.predict_probabilities(
            input_tomo, slab_size=slab_size, smoothing_sigma=smoothing_sigma,
            parallel_axes=False,
        )
        parallel = predictor.predict_probabilities(
            input_tomo, slab_size=slab_size, smoothing_sigma=smoothing_sigma,
            parallel_axes=True,
        )

        np.testing.assert_allclose(serial, parallel, atol=1e-5)


def test_parallel_axes_propagates_exception(trained_checkpoint, monkeypatch):
    """An exception raised in one axis' worker thread must propagate to the caller."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(str(trained_checkpoint), compile_model=False)
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        original = TomoSlabPredictor._predict_single_axis_with_slab_blending

        def flaky(self, volume_3d, axis, slab_size, batch_size, tqdm_position=None):
            if axis == "YZ":
                raise RuntimeError("boom")
            return original(
                self, volume_3d, axis, slab_size, batch_size, tqdm_position=tqdm_position
            )

        monkeypatch.setattr(
            TomoSlabPredictor, "_predict_single_axis_with_slab_blending", flaky
        )

        with pytest.raises(RuntimeError, match="boom"):
            predictor.predict_probabilities(input_tomo, slab_size=3, parallel_axes=True)


def test_parallel_axes_rejected_with_compiled_model(trained_checkpoint):
    """parallel_axes=True must be rejected when the model is compiled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(str(trained_checkpoint), compile_model=False)
        # Simulate a successfully compiled model without depending on torch.compile
        # actually succeeding in this environment.
        predictor._compiled = True

        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)
        with pytest.raises(ValueError, match="parallel_axes"):
            predictor.predict_probabilities(input_tomo, slab_size=3, parallel_axes=True)
