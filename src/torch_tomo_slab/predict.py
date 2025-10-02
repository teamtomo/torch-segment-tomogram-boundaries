"""Tomographic slab prediction and inference pipeline.

This module provides functionality for running inference on trained models
to generate boundary masks from tomographic volumes. It includes utilities
for plane fitting, point cloud processing, and mask generation.
"""
import gc
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from torch_tomo_slab.processing import TrainingDataGenerator

import mrcfile
import numpy as np
import pandas as pd
from torch_tomo_slab.models import create_unet
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch_tomo_slab import config
from torch_tomo_slab.losses import get_loss_function
from torch_tomo_slab.pl_model import SegmentationModel

from torch_tomo_slab.utils import threeD, twoD
from torch_tomo_slab.utils.twoD import robust_normalization

# Configure logging for prediction pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from torch_tomo_slab.utils.common import get_device

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
    if points.shape[0] == 0: return points
    voxel_indices = np.floor(points / grid_size).astype(np.int32)
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['voxel_x'], df['voxel_y'], df['voxel_z'] = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    return df.groupby(['voxel_x', 'voxel_y', 'voxel_z'])[['x', 'y', 'z']].mean().to_numpy()

def fit_best_plane(points: np.ndarray, angle_res: int = 180, dist_res: int = 200) -> Dict[str, List[float]]:
    """
    Fit the best plane to 3D points using Hough Transform.

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
    if len(points) < 50: raise ValueError(f"Not enough points ({len(points)}) to fit a plane.")
    phis = np.linspace(0, np.pi, angle_res); thetas = np.linspace(0, np.pi, angle_res)
    phi_grid, theta_grid = np.meshgrid(phis, thetas)
    nx, ny, nz = np.sin(phi_grid) * np.cos(theta_grid), np.sin(phi_grid) * np.sin(theta_grid), np.cos(phi_grid)
    normals = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    dists = np.dot(points, normals.T)
    min_dist, max_dist = dists.min(), dists.max()
    accumulator = np.zeros((len(normals), dist_res), dtype=np.uint32)
    dist_bins = np.linspace(min_dist, max_dist, dist_res)
    for i in range(len(normals)):
        hist, _ = np.histogram(dists[:, i], bins=dist_res, range=(min_dist, max_dist))
        accumulator[i, :] = hist
    normal_idx, dist_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    best_normal, best_dist = normals[normal_idx], dist_bins[dist_idx]
    nx, ny, nz = best_normal
    if abs(nz) < 1e-6: raise ValueError("Detected a plane nearly vertical to the Z-axis.")
    return {'coefficients': [-nx/nz, -ny/nz, best_dist/nz]}

def fit_and_generate_mask(mask: np.ndarray, downsample_grid_size: int) -> np.ndarray:
    """
    Extract boundary points, fit planes, and generate clean slab mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask as 3D array where non-zero values indicate boundaries.
    downsample_grid_size : int
        Grid size for point cloud downsampling before plane fitting.

    Returns
    -------
    np.ndarray
        Clean binary mask generated from fitted top/bottom planes.

    Raises
    ------
    ValueError
        If insufficient boundary points (<1000) are found.
    """
    logging.info("Extracting boundary points from the binarized mask...")
    surface = mask - binary_erosion(mask)
    coords_zyx = np.argwhere(surface > 0)
    if len(coords_zyx) < 1000: raise ValueError(f"Not enough boundary points ({len(coords_zyx)}) found.")
    points_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float32)
    z_median = np.median(points_xyz[:, 2])
    top_points, bottom_points = points_xyz[points_xyz[:, 2] >= z_median], points_xyz[points_xyz[:, 2] < z_median]
    top_points_ds = downsample_points(top_points, grid_size=downsample_grid_size)
    bottom_points_ds = downsample_points(bottom_points, grid_size=downsample_grid_size)
    logging.info(f"Downsampled top surface from {len(top_points)} to {len(top_points_ds)} points.")
    logging.info(f"Downsampled bottom surface from {len(bottom_points)} to {len(bottom_points_ds)} points.")
    plane_top = fit_best_plane(top_points_ds)
    plane_bottom = fit_best_plane(bottom_points_ds)
    return generate_mask_from_planes({'top': plane_top, 'bottom': plane_bottom}, mask.shape)

def generate_mask_from_planes(planes: Dict[str, Dict[str, List[float]]], volume_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Generate binary mask volume from top and bottom plane equations.

    Parameters
    ----------
    planes : Dict[str, Dict[str, List[float]]]
        Dictionary with 'top' and 'bottom' keys, each containing plane coefficients.
    volume_shape : Tuple[int, int, int]
        Target volume shape as (depth, height, width).

    Returns
    -------
    np.ndarray
        Binary mask as 3D array where 1 indicates region between planes.
    """
    Nz, Ny, Nx = volume_shape
    coef_b, coef_t = planes['bottom']['coefficients'], planes['top']['coefficients']
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    z_bottom, z_top = (coef_b[0]*xx + coef_b[1]*yy + coef_b[2]), (coef_t[0]*xx + coef_t[1]*yy + coef_t[2])
    min_plane, max_plane = np.minimum(z_bottom, z_top), np.maximum(z_bottom, z_top)
    zz = np.arange(Nz)[:, np.newaxis, np.newaxis]
    return ((zz >= min_plane) & (zz <= max_plane)).astype(np.int8)

class TomoSlabPredictor:
    """
    Predict tomographic slab boundaries using trained deep learning models.

    This class provides a complete inference pipeline for generating clean
    boundary masks from 3D tomographic volumes. It loads trained PyTorch Lightning
    models and applies them to new data with post-processing to generate
    geometrically consistent slab boundaries.

    The prediction workflow includes:
    1. Model loading and setup from checkpoint
    2. Volume preprocessing and normalization
    3. Multi-axis inference with slab blending
    4. Probability map averaging and smoothing
    5. Plane fitting to boundary points
    6. Final mask generation from fitted planes

    Attributes
    ----------
    device : torch.device
        PyTorch device for inference (GPU/CPU).
    model : SegmentationModel
        Loaded PyTorch Lightning model for inference.
    target_shape_3d : tuple
        Target 3D shape for volume preprocessing.
    """

    def __init__(self, model_checkpoint_path: Union[str, Path], compile_model: bool = True) -> None:
        """
        Initialize predictor with trained model checkpoint.

        Parameters
        ----------
        model_checkpoint_path : str or Path
            Path to the trained PyTorch Lightning model checkpoint (.ckpt file).
        compile_model : bool, default=True
            Whether to compile the model using torch.compile for faster inference.
            Requires PyTorch 2.0+ and may increase initial load time.

        Raises
        ------
        ValueError
            If target_shape is not found in model checkpoint hyperparameters.
        """
        self.device = get_device()
        logging.info(f"Loading model from checkpoint: {model_checkpoint_path}")
        base_model = create_unet(**config.MODEL_CONFIG)

        loss_fn = get_loss_function(config.LOSS_CONFIG)
        self.model = SegmentationModel.load_from_checkpoint(
            model_checkpoint_path, map_location=self.device,
            model=base_model, loss_function=loss_fn
        )
        self.model.eval().to(self.device)

        # Compile model for faster inference if requested and supported
        if compile_model:
            self._compile_model()
        # Load target shape from the model's saved hyperparameters
        self.target_shape_3d = self.model.hparams.get('target_shape')
        if not self.target_shape_3d:
            raise ValueError("`target_shape` not found in model checkpoint. Please retrain with the updated trainer.")
        logging.info(f"Using target shape from checkpoint for resizing: {self.target_shape_3d}")

    def _compile_model(self) -> None:
        """
        Compile model using torch.compile for faster inference.

        Uses TorchDynamo and TorchInductor to optimize the model.
        Falls back gracefully if compilation is not supported.
        """
        try:
            # Check if torch.compile is available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                logging.info("Compiling model for faster inference...")

                # Try different compilation modes and backends
                compile_configs = [
                    # Most aggressive optimization
                    {'mode': 'max-autotune', 'fullgraph': False, 'dynamic': False},
                    # Moderate optimization with better compatibility
                    {'mode': 'reduce-overhead', 'fullgraph': False, 'dynamic': False},
                    # Safe default optimization
                    {'mode': 'default', 'fullgraph': False, 'dynamic': True},
                ]

                # Try different backends in order of preference
                backends_to_try = ['inductor', 'aot_eager', 'eager']

                for config in compile_configs:
                    for backend in backends_to_try:
                        try:
                            config['backend'] = backend
                            self.model = torch.compile(self.model, **config)
                            logging.info(f"Model compiled successfully using '{backend}' backend with mode '{config['mode']}'")
                            return
                        except Exception as e:
                            logging.warning(f"Failed to compile with '{backend}' backend and mode '{config['mode']}': {e}")
                            continue

                logging.warning("All compilation backends failed. Using uncompiled model.")
            else:
                logging.warning("torch.compile not available. Requires PyTorch 2.0+. Using uncompiled model.")

        except Exception as e:
            logging.warning(f"Model compilation failed: {e}. Using uncompiled model.")

    def warm_up_model(self, batch_size: int = 16) -> None:
        """
        Warm up compiled model with dummy inputs to trigger optimizations.

        Parameters
        ----------
        batch_size : int, default=16
            Batch size for warm-up inference.
        """
        if hasattr(self.model, '_orig_mod'):  # Check if model is compiled
            try:
                logging.info("Warming up compiled model...")

                # Create dummy input matching the single-channel model input shape
                if hasattr(self.model, 'hparams') and self.target_shape_3d:
                    D, H, W = self.target_shape_3d
                    dummy_input = torch.randn(batch_size, 1, D, W, device=self.device)

                    # Run a few warm-up iterations
                    for _ in range(3):
                        _ = self.model(dummy_input)

                    logging.info("Model warm-up completed")
                else:
                    logging.warning("Could not determine input shape for warm-up")

            except Exception as e:
                logging.warning(f"Model warm-up failed: {e}")
        else:
            logging.info("Model not compiled, skipping warm-up")

    @torch.no_grad()
    def predict(self,
                input_tomogram: Union[str, Path, np.ndarray],
                output_path: Optional[Path] = None,
                save_raw_mask_path: Optional[Path] = None,
                save_orthogonal_views_path: Optional[Path] = None,
                slab_size: int = 15,
                batch_size: int = 16,
                binarize_threshold: float = 0.5,
                smoothing_sigma: Optional[float] = None,
                downsample_grid_size: int = 8,
                warm_up: bool = False) -> np.ndarray:
        """
        Execute full prediction pipeline on tomographic volume.

        This method performs the complete inference workflow:
        1. Load and preprocess input tomogram
        2. Predict boundary probabilities using trained model
        3. Apply slab blending for temporal consistency
        4. Apply optional 3D Gaussian smoothing
        5. Binarize probability maps
        6. Fit planes to boundary points
        7. Generate final slab mask from fitted planes

        Parameters
        ----------
        input_tomogram : Path or np.ndarray
            Input tomogram as MRC file path or pre-loaded 3D numpy array.
        output_path : Path, optional
            Path to save output mask as MRC file. If None, only returns array.
        save_raw_mask_path : Path, optional
            Path to save the raw binarized mask before plane fitting. If None, not saved.
        slab_size : int, default=15
            Size of slab for temporal blending (must be odd number). If 1, no blending.
        batch_size : int, default=16
            Batch size for processing 2D slices during inference.
        binarize_threshold : float, default=0.5
            Threshold for converting probability maps to binary masks.
        smoothing_sigma : float, optional
            Standard deviation for 3D Gaussian smoothing filter. If None, no smoothing.
        downsample_grid_size : int, default=8
            Voxel grid size for downsampling point clouds before plane fitting.
        warm_up : bool, default=False
            Whether to warm up the compiled model before inference for optimal performance.

        Returns
        -------
        np.ndarray
            Final binary slab mask as 3D numpy array with same shape as input.

        Raises
        ------
        ValueError
            If plane fitting fails due to insufficient boundary points.
        RuntimeError
            If RANSAC plane fitting encounters numerical issues.
        """
        if isinstance(input_tomogram, Path) or isinstance(input_tomogram, str):
            with mrcfile.open(input_tomogram, permissive=True) as mrc:
                original_data_np = mrc.data.astype(np.float32)
                voxel_size = mrc.voxel_size.copy()
        else:
            original_data_np = input_tomogram.astype(np.float32)
            voxel_size = None # Voxel size is unknown if input is an array

        original_shape = original_data_np.shape
        logging.info(f"Input tomogram shape: {original_shape}")

        # Warm up compiled model if requested
        if warm_up:
            self.warm_up_model(batch_size)

        resized_volume = threeD.resize_and_pad_3d(torch.from_numpy(original_data_np),target_shape=self.target_shape_3d, mode='image').to(self.device)

        # Predict along XZ and YZ axes using memory-efficient slab blending
        pred_xz = self._predict_single_axis_with_slab_blending_optimized(resized_volume, 'XZ', slab_size, batch_size)
        pred_yz_permuted = self._predict_single_axis_with_slab_blending_optimized(
            resized_volume.permute(0, 2, 1), 'YZ', slab_size, batch_size
        )
        pred_yz = pred_yz_permuted.permute(0, 2, 1)

        logging.info("Averaging final predictions from both axes.")
        prob_map_tensor = (pred_xz + pred_yz) / 2.0

        if smoothing_sigma and smoothing_sigma > 0:
            logging.info(f"Applying 3D Gaussian smoothing with sigma={smoothing_sigma}...")
            prob_map_tensor = threeD.gpu_gaussian_blur_3d(prob_map_tensor, smoothing_sigma, self.device)

        if save_orthogonal_views_path:
            logging.info(f"Saving orthogonal views to {save_orthogonal_views_path}")
            save_orthogonal_views_path.mkdir(parents=True, exist_ok=True)
            prob_map_np_before_resize = prob_map_tensor.cpu().numpy()
            
            # XY view
            xy_slice = prob_map_np_before_resize[prob_map_np_before_resize.shape[0] // 2, :, :]
            plt.imsave(save_orthogonal_views_path / "xy_view.png", xy_slice, cmap='gray')

            # XZ view
            xz_slice = prob_map_np_before_resize[:, prob_map_np_before_resize.shape[1] // 2, :]
            plt.imsave(save_orthogonal_views_path / "xz_view.png", xz_slice, cmap='gray')

            # YZ view
            yz_slice = prob_map_np_before_resize[:, :, prob_map_np_before_resize.shape[2] // 2]
            plt.imsave(save_orthogonal_views_path / "yz_view.png", yz_slice, cmap='gray')

        logging.info(f"Resizing prediction back to original shape {original_shape}...")
        prob_map_np = F.interpolate(prob_map_tensor.unsqueeze(0).unsqueeze(0), size=original_shape, mode='trilinear', align_corners=False).squeeze().cpu().numpy()

        binary_mask_np = (prob_map_np > binarize_threshold).astype(np.uint8)

        try:
            final_mask = fit_and_generate_mask(binary_mask_np, downsample_grid_size)
        except (ValueError, RuntimeError) as e:
            logging.error(f"Plane fitting failed: {e}. Returning the raw binarized mask instead.")
            final_mask = binary_mask_np

        if output_path:
            logging.info(f"Saving final mask to {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mrcfile.write(output_path, final_mask, voxel_size=voxel_size, overwrite=True)

        # Save the raw probability map (network output) if a path is provided
        if save_raw_mask_path:
            logging.info(f"Saving raw probability map to {save_raw_mask_path}")
            save_raw_mask_path.parent.mkdir(parents=True, exist_ok=True)
            mrcfile.write(save_raw_mask_path, prob_map_np, voxel_size=voxel_size, overwrite=True)

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info("Prediction complete.")
        return final_mask

    @torch.no_grad()
    def _predict_single_axis(self, volume_3d: torch.Tensor, axis: str, batch_size: int) -> torch.Tensor:
        """
        Predict boundary probabilities along single axis without slab blending.

        Parameters
        ----------
        volume_3d : torch.Tensor
            Input volume tensor of shape (D, H, W).
        axis : str
            Axis name for logging ('XZ' or 'YZ').
        batch_size : int
            Batch size for model inference.

        Returns
        -------
        torch.Tensor
            Predicted probability volume of same shape as input.
        """
        logging.info(f"Predicting along {axis} axis...")
        D, H, W = volume_3d.shape
        predictions = []

        # Process slices in batches for efficiency
        for i in tqdm(range(0, H, batch_size), desc=f"Processing {axis} axis", leave=False, ncols=80):
            end_idx = min(i + batch_size, H)
            batch_slices = []

            for j in range(i, end_idx):
                slice_2d = volume_3d[:, j, :]  # Shape: (D, W)

                # Apply robust normalization and keep the single-channel input
                slice_norm = robust_normalization(slice_2d).unsqueeze(0)
                batch_slices.append(slice_norm)

            # Stack slices into batch
            batch_tensor = torch.stack(batch_slices, dim=0)  # Shape: (batch_size, 1, D, W)

            # Process batch through model
            pred_batch = self.model(batch_tensor)
            pred_batch = torch.sigmoid(pred_batch).squeeze(1)  # Remove channel dim: (batch_size, D, W)

            predictions.extend([pred_batch[k] for k in range(pred_batch.shape[0])])

        return torch.stack(predictions, dim=1)  # Shape: (D, H, W)


    @torch.no_grad()
    def _predict_raw_slab(self, slab_3d: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Predict raw probabilities for a 3D slab using batched processing.

        Parameters
        ----------
        slab_3d : torch.Tensor
            Input 3D slab tensor of shape (D, H, W).
        batch_size : int
            Batch size for model inference.
        proc : TrainingDataGenerator
            Data processor instance (unused, kept for compatibility).

        Returns
        -------
        torch.Tensor
            Predicted probability slab of same shape as input.
        """
        D, H, W = slab_3d.shape
        predictions = []

        # Process slices in batches
        for i in range(0, H, batch_size):
            end_idx = min(i + batch_size, H)
            batch_slices = []

            for j in range(i, end_idx):
                slice_2d = slab_3d[:, j, :]  # Shape: (D, W)

                # Apply robust normalization and keep the single-channel input
                slice_norm = robust_normalization(slice_2d).unsqueeze(0)
                batch_slices.append(slice_norm)

            # Stack slices into batch
            batch_tensor = torch.stack(batch_slices, dim=0)  # Shape: (batch_size, 1, D, W)

            # Process batch through model
            pred_batch = self.model(batch_tensor)
            pred_batch = torch.sigmoid(pred_batch).squeeze(1)  # Remove channel dim: (batch_size, D, W)

            predictions.extend([pred_batch[k] for k in range(pred_batch.shape[0])])

        return torch.stack(predictions, dim=1)  # Shape: (D, H, W)

    @torch.no_grad()
    def _predict_single_axis_with_slab_blending_optimized(self, volume_3d: torch.Tensor, axis: str, slab_size: int, batch_size: int) -> torch.Tensor:
        """
        Optimized version of slab blending prediction combining both steps.
        This method provides better memory efficiency for large volumes.

        Parameters
        ----------
        volume_3d : torch.Tensor
            Input volume tensor of shape (D, H, W).
        axis : str
            Axis name for logging ('XZ' or 'YZ').
        slab_size : int
            Size of temporal slab for blending adjacent slices.
        batch_size : int
            Batch size for model inference.
        proc : TrainingDataGenerator
            Data processor instance (unused, kept for compatibility).

        Returns
        -------
        torch.Tensor
            Predicted probability volume of same shape as input.
        """
        if slab_size <= 1:
            return self._predict_single_axis(volume_3d, axis, batch_size)

        logging.info(f"Predicting with slab blending along {axis} axis (slab_size={slab_size})...")
        num_slices = volume_3d.shape[1]
        final_slices = []
        hann_window = torch.hann_window(slab_size, periodic=False, device=self.device)

        for i in tqdm(range(num_slices), desc=f"Slab Blending ({axis} axis)", leave=False, ncols=80):
            half_slab = slab_size // 2
            start = max(0, i - half_slab)
            end = min(num_slices, i + half_slab + 1)

            # Amount of padding required on each side to keep slab length static
            expected_start = i - half_slab
            expected_end = i + half_slab + 1
            pad_left = max(0, -expected_start)
            pad_right = max(0, expected_end - num_slices)

            slab_3d = volume_3d[:, start:end, :]
            if pad_left > 0 or pad_right > 0:
                # Replicate border slices so the model always receives slab_size slices.
                first_slice = slab_3d[:, :1, :]
                last_slice = slab_3d[:, -1:, :]
                if pad_left > 0:
                    left_pad = first_slice.repeat(1, pad_left, 1)
                    slab_3d = torch.cat((left_pad, slab_3d), dim=1)
                if pad_right > 0:
                    right_pad = last_slice.repeat(1, pad_right, 1)
                    slab_3d = torch.cat((slab_3d, right_pad), dim=1)

            # Extract and predict slab
            predicted_slab = self._predict_raw_slab(slab_3d, batch_size)

            # Calculate window indices and apply blending
            # Start from the full Hann window and zero padded regions so they do not contribute
            current_window = hann_window.clone()
            if pad_left > 0:
                current_window[:pad_left] = 0
            if pad_right > 0:
                current_window[-pad_right:] = 0

            # Apply weighted averaging
            if current_window.sum() > 0:
                # predicted_slab has shape (D, slab_len, W); we need to blend across the slab_len axis
                weights = current_window.view(1, -1, 1)
                weighted_sum = (predicted_slab * weights).sum(dim=1)
                final_slice = weighted_sum / weights.sum()
            else:
                # Fallback: predict single slice with the same single-channel pathway
                slice_2d = volume_3d[:, i, :]
                slice_input = robust_normalization(slice_2d).unsqueeze(0).unsqueeze(0)
                final_slice = torch.sigmoid(self.model(slice_input)).squeeze(0).squeeze(0)

            final_slices.append(final_slice)

        return torch.stack(final_slices, dim=1)

def predict(
    input_tomogram: Union[str, Path, np.ndarray],
    model_checkpoint_path: Union[str, Path],
    output_path: Optional[Path] = None,
    **predict_kwargs,
) -> np.ndarray:
    """
    High-level function to predict a slab mask from a tomogram.

    This function provides a simple interface for inference, handling the
    instantiation of the predictor, model loading, and execution of the
    prediction pipeline.

    Parameters
    ----------
    input_tomogram : Union[str, Path, np.ndarray]
        Input tomogram, can be a path to an MRC file or a pre-loaded numpy array.
    model_checkpoint_path : Union[str, Path]
        Path to the trained PyTorch Lightning model checkpoint (.ckpt file).
    output_path : Optional[Path], optional
        Path to save the final output mask as an MRC file. If None, the mask
        is returned but not saved to disk. By default None.
    **predict_kwargs :
        Additional keyword arguments to be passed to the predictor's predict method.
        This allows for overriding settings like 'slab_size', 'batch_size', etc.

    Returns
    -------
    np.ndarray
        The final binary slab mask as a 3D numpy array.
    """
    # Initialize the predictor with the given model checkpoint
    predictor = TomoSlabPredictor(model_checkpoint_path=model_checkpoint_path)

    # Run the prediction pipeline
    final_mask = predictor.predict(
        input_tomogram=input_tomogram,
        output_path=output_path,
        **predict_kwargs,
    )

    return final_mask
