# src/torch_tomo_slab/predict.py
import logging
from pathlib import Path
import gc
import math

from typing import Union, Optional
import mrcfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_erosion
from tqdm import tqdm

from .pl_model import SegmentationModel
from .processing import TrainingDataGenerator
from . import constants, config
from .losses import get_loss_function

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device() -> torch.device:
    """Gets the appropriate torch device."""
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def downsample_points(points: np.ndarray, grid_size: int) -> np.ndarray:
    """Reduces point cloud density using voxel grid downsampling."""
    if points.shape[0] == 0: return points
    voxel_indices = np.floor(points / grid_size).astype(np.int32)
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['voxel_x'], df['voxel_y'], df['voxel_z'] = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    return df.groupby(['voxel_x', 'voxel_y', 'voxel_z'])[['x', 'y', 'z']].mean().to_numpy()

def fit_best_plane(points: np.ndarray, angle_res: int = 180, dist_res: int = 200) -> dict:
    """Fits the best plane to a set of 3D points using a Hough Transform."""
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
    """Extracts points, fits planes, and generates a clean mask."""
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

def generate_mask_from_planes(planes: dict, volume_shape: tuple) -> np.ndarray:
    """Generates a binary mask volume from two plane equations."""
    Nz, Ny, Nx = volume_shape
    coef_b, coef_t = planes['bottom']['coefficients'], planes['top']['coefficients']
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    z_bottom, z_top = (coef_b[0]*xx + coef_b[1]*yy + coef_b[2]), (coef_t[0]*xx + coef_t[1]*yy + coef_t[2])
    min_plane, max_plane = np.minimum(z_bottom, z_top), np.maximum(z_bottom, z_top)
    zz = np.arange(Nz)[:, np.newaxis, np.newaxis]
    return ((zz >= min_plane) & (zz <= max_plane)).astype(np.int8)

class TomoSlabPredictor:
    """
    Handles loading a trained model and performing inference on 3D tomograms to generate slab masks.
    """
    def __init__(self, model_checkpoint_path: str):
        self.device = get_device()
        logging.info(f"Loading model from checkpoint: {model_checkpoint_path}")
        base_model = smp.create_model(
            arch=constants.MODEL_ARCH, encoder_name=constants.MODEL_ENCODER, encoder_weights=None,
            encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16], decoder_attention_type='scse',
            classes=1, in_channels=2, activation=None
        )
        loss_fn = get_loss_function(config.LOSS_CONFIG)
        self.model = SegmentationModel.load_from_checkpoint(
            model_checkpoint_path, map_location=self.device,
            model=base_model, loss_function=loss_fn
        )
        self.model.eval().to(self.device)
        # Load target shape from the model's saved hyperparameters
        self.target_shape_3d = self.model.hparams.get('target_shape')
        if not self.target_shape_3d:
            raise ValueError("`target_shape` not found in model checkpoint. Please retrain with the updated trainer.")
        logging.info(f"Using target shape from checkpoint for resizing: {self.target_shape_3d}")

    @torch.no_grad()
    def predict(self,
                input_tomogram: Union[Path, np.ndarray],
                output_path: Optional[Path] = None,
                slab_size: int = 15,
                batch_size: int = 16,
                binarize_threshold: float = 0.5,
                smoothing_sigma: Optional[float] = None,
                downsample_grid_size: int = 8) -> np.ndarray:
        """
        Runs the full prediction pipeline on a tomogram.
        Args:
            input_tomogram: Path to the input .mrc file or a pre-loaded NumPy array.
            output_path: Optional path to save the final mask. If None, the mask is only returned.
            slab_size: Size of the slab for blending (must be odd).
            batch_size: Batch size for processing 2D slices.
            binarize_threshold: Threshold for binarizing the probability map.
            smoothing_sigma: Sigma for an optional 3D Gaussian smoothing filter.
            downsample_grid_size: Grid size for point cloud downsampling before plane fitting.
        Returns:
            The final, plane-fitted mask as a NumPy array.
        """
        if isinstance(input_tomogram, Path):
            with mrcfile.open(input_tomogram, permissive=True) as mrc:
                original_data_np = mrc.data.astype(np.float32)
                voxel_size = mrc.voxel_size.copy()
        else:
            original_data_np = input_tomogram.astype(np.float32)
            voxel_size = None # Voxel size is unknown if input is an array

        original_shape = original_data_np.shape
        logging.info(f"Input tomogram shape: {original_shape}")

        processor = TrainingDataGenerator() # Use its methods for processing
        resized_volume = processor._resize_and_pad_3d(torch.from_numpy(original_data_np), mode='image').to(self.device)

        pred_xz = self._predict_single_axis_with_slab_blending(resized_volume, 'XZ', slab_size, batch_size, processor)
        pred_yz = self._predict_single_axis_with_slab_blending(resized_volume.permute(0, 2, 1), 'YZ', slab_size, batch_size, processor).permute(0, 2, 1)

        logging.info("Averaging final predictions from both axes.")
        prob_map_tensor = (pred_xz + pred_yz) / 2.0

        if smoothing_sigma and smoothing_sigma > 0:
            logging.info(f"Applying 3D Gaussian smoothing with sigma={smoothing_sigma}...")
            prob_map_tensor = self._gpu_gaussian_blur_3d(prob_map_tensor, sigma=smoothing_sigma)

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

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info("Prediction complete.")
        return final_mask

    def _predict_single_axis_with_slab_blending(self, volume_3d: torch.Tensor, axis: str, slab_size: int, batch_size: int, proc: TrainingDataGenerator) -> torch.Tensor:
        num_slices = volume_3d.shape[1]
        final_slices, hann_window = [], torch.hann_window(slab_size, periodic=False, device=self.device)
        for i in tqdm(range(num_slices), desc=f"Slab Blending ({axis} axis)", leave=False, ncols=80):
            half_slab = slab_size // 2
            start, end = max(0, i - half_slab), min(num_slices, i + half_slab + 1)
            predicted_slab = self._predict_raw_slab(volume_3d[:, start:end, :], batch_size, proc)
            win_start, win_end = max(0, half_slab - i), min(slab_size, half_slab + (num_slices - i))
            current_window = hann_window[win_start:win_end]
            final_slice = torch.einsum('dwh,d->wh', predicted_slab, current_window) / current_window.sum()
            final_slices.append(final_slice)
        return torch.stack(final_slices, dim=1)

    def _predict_raw_slab(self, raw_slab: torch.Tensor, batch_size: int, proc: TrainingDataGenerator) -> torch.Tensor:
        views = raw_slab.permute(1, 0, 2)
        all_preds = []
        for i in range(0, len(views), batch_size):
            batch_views = views[i:i+batch_size]
            input_channels = []
            for v in batch_views:
                norm_v = proc._robust_normalization(v)
                var_v = proc._local_variance_2d(norm_v)
                input_channels.append(torch.stack([norm_v, var_v], dim=0))
            preds = torch.sigmoid(self.model(torch.stack(input_channels))).detach()
            all_preds.append(preds)
        return torch.cat(all_preds).squeeze(1)

    def _gpu_gaussian_blur_3d(self, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        kernel_size = 2 * math.ceil(3.5 * sigma) + 1
        coords = torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype) - kernel_size // 2
        g = coords.pow(2); kernel_1d = torch.exp(-g / (2 * sigma**2))
        kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel_3d /= kernel_3d.sum()
        kernel_5d = kernel_3d.unsqueeze(0).unsqueeze(0);
        conv = nn.Conv3d(1, 1, kernel_size, padding='same', bias=False).to(tensor.device)
        conv.weight.data = kernel_5d
        return conv(tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)