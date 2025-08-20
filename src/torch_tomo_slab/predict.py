# src/torch_tomo_slab/scripts/predict.py

# --- Standard Library Imports ---
import argparse
import logging
from pathlib import Path
import gc
import math
import sys

# --- Third-Party Imports ---
import mrcfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_erosion
from tqdm import tqdm

# --- Add project root to path for local imports ---
sys.path.append(str(Path(__file__).resolve().parents[2]))

# --- Local Project Imports ---
from torch_tomo_slab.pl_model import SegmentationModel
from torch_tomo_slab.scripts.p02_data_preparation import (
    resize_and_pad_3d,
    robust_normalization,
    local_variance_2d,
    find_data_pairs
)
from torch_tomo_slab import constants, config
from torch_tomo_slab.train import get_loss_function

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def gpu_gaussian_blur_3d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = 2 * math.ceil(3.5 * sigma) + 1
    coords = torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype) - kernel_size // 2
    g = coords.pow(2); kernel_1d = torch.exp(-g / (2 * sigma**2))
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    kernel_3d /= kernel_3d.sum()
    kernel_5d = kernel_3d.unsqueeze(0).unsqueeze(0); tensor_5d = tensor.unsqueeze(0).unsqueeze(0)
    conv = nn.Conv3d(1, 1, kernel_size, padding='same', bias=False).to(tensor.device)
    conv.weight.data = kernel_5d
    return conv(tensor_5d).squeeze(0).squeeze(0)

# --- NEW: Voxel Grid Downsampling for Performance ---
def downsample_points(points: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Reduces the number of points in a point cloud using voxel grid downsampling.
    """
    if points.shape[0] == 0: return points
    
    # Assign each point to a coarse voxel grid
    voxel_indices = np.floor(points / grid_size).astype(np.int32)
    
    # Use pandas for a highly efficient groupby-mean operation
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['voxel_x'] = voxel_indices[:, 0]
    df['voxel_y'] = voxel_indices[:, 1]
    df['voxel_z'] = voxel_indices[:, 2]
    
    # Calculate the centroid of points within each voxel
    downsampled_df = df.groupby(['voxel_x', 'voxel_y', 'voxel_z']).mean()
    
    return downsampled_df[['x', 'y', 'z']].to_numpy()

# --- Fast, Vectorized Hough Transform for Plane Fitting ---

def fit_best_plane(points: np.ndarray, angle_res: int = 180, dist_res: int = 200) -> dict:
    if len(points) < 50: raise ValueError(f"Not enough points ({len(points)}) to fit a plane.")
    phis = np.linspace(0, np.pi, angle_res); thetas = np.linspace(0, np.pi, angle_res)
    phi_grid, theta_grid = np.meshgrid(phis, thetas)
    nx, ny, nz = np.sin(phi_grid) * np.cos(theta_grid), np.sin(phi_grid) * np.sin(theta_grid), np.cos(phi_grid)
    normals = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    dists = np.dot(points, normals.T)
    min_dist, max_dist = dists.min(), dists.max()
    accumulator = np.zeros((len(normals), dist_res), dtype=np.uint32)
    dist_bins = np.linspace(min_dist, max_dist, dist_res)

    for i in tqdm(range(len(normals)), desc="Hough Voting", leave=False, ncols=80):
        hist, _ = np.histogram(dists[:, i], bins=dist_res, range=(min_dist, max_dist))
        accumulator[i, :] = hist

    normal_idx, dist_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    best_normal, best_dist = normals[normal_idx], dist_bins[dist_idx]
    nx, ny, nz = best_normal
    if abs(nz) < 1e-6: raise ValueError("Detected a plane nearly vertical to the Z-axis.")
    a, b, d_intercept = -nx / nz, -ny / nz, best_dist / nz
    return {'coefficients': [a, b, d_intercept]}

def fit_and_generate_mask(mask: np.ndarray, downsample_grid_size: int):
    logging.info("Extracting boundary points from the binarized mask...")
    surface = mask - binary_erosion(mask)
    coords_zyx = np.argwhere(surface > 0)
    if len(coords_zyx) < 1000: raise ValueError(f"Not enough boundary points ({len(coords_zyx)}) found.")
    
    points_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float32)
    z_median = np.median(points_xyz[:, 2])
    top_points, bottom_points = points_xyz[points_xyz[:, 2] >= z_median], points_xyz[points_xyz[:, 2] < z_median]

    # --- NEW: Downsample the points before fitting ---
    top_points_ds = downsample_points(top_points, grid_size=downsample_grid_size)
    bottom_points_ds = downsample_points(bottom_points, grid_size=downsample_grid_size)
    logging.info(f"Downsampled top surface from {len(top_points)} to {len(top_points_ds)} points.")
    logging.info(f"Downsampled bottom surface from {len(bottom_points)} to {len(bottom_points_ds)} points.")
    
    plane_top = fit_best_plane(top_points_ds)
    plane_bottom = fit_best_plane(bottom_points_ds)

    planes = {'top': plane_top, 'bottom': plane_bottom}
    logging.info("Generating final, clean mask from fitted planes...")
    return generate_mask_from_planes(planes, mask.shape)

def generate_mask_from_planes(planes: dict, volume_shape: tuple) -> np.ndarray:
    Nz, Ny, Nx = volume_shape
    coef_b, coef_t = planes['bottom']['coefficients'], planes['top']['coefficients']
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    z_bottom, z_top = (coef_b[0]*xx + coef_b[1]*yy + coef_b[2]), (coef_t[0]*xx + coef_t[1]*yy + coef_t[2])
    min_plane, max_plane = np.minimum(z_bottom, z_top), np.maximum(z_bottom, z_top)
    zz = np.arange(Nz)[:, np.newaxis, np.newaxis]
    return ((zz >= min_plane) & (zz <= max_plane)).astype(np.int8)

# --- Main Predictor Class ---

class TomogramPredictor:
    def __init__(self, model_checkpoint_path: str):
        self.device = get_device()
        logging.info(f"Loading model from checkpoint: {model_checkpoint_path}")
        base_model = smp.create_model(arch=constants.MODEL_ARCH, encoder_name=constants.MODEL_ENCODER, encoder_weights=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16], decoder_attention_type='scse', classes=1, in_channels=2, activation=None)
        loss_fn = get_loss_function(config.LOSS_CONFIG)
        self.model = SegmentationModel.load_from_checkpoint(model_checkpoint_path, map_location=self.device, model=base_model, loss_function=loss_fn)
        self.model.eval().to(self.device)
        self.target_shape_3d = self.model.hparams.get('target_shape')
        if not self.target_shape_3d:
            logging.warning("`target_shape` not found. Determining from training data...")
            all_shapes = [mrcfile.open(p[0], permissive=True).data.shape for p in find_data_pairs(Path(constants.REFERENCE_TOMOGRAM_DIR), Path(constants.MASK_OUTPUT_DIR))]
            self.target_shape_3d = tuple(int(d + (d % 2)) for d in np.max(np.array(all_shapes), axis=0))
        logging.info(f"Using target shape for resizing: {self.target_shape_3d}")

    @torch.no_grad()
    def predict(self, input_mrc_path: Path, output_mrc_path: Path, diagnostics_path: Path, slab_size: int, batch_size: int, binarize_threshold: float, smoothing_sigma: float, downsample_grid_size: int):
        with mrcfile.open(input_mrc_path, permissive=True) as mrc:
            original_data_np = mrc.data.astype(np.float32)
            original_shape, voxel_size = original_data_np.shape, mrc.voxel_size.copy()
        logging.info(f"Input tomogram shape: {original_shape}")
        
        resized_volume = resize_and_pad_3d(torch.from_numpy(original_data_np), self.target_shape_3d, mode='image').to(self.device)
        pred_xz = self._predict_single_axis_with_slab_blending(resized_volume, 'XZ', slab_size, batch_size)
        pred_yz = self._predict_single_axis_with_slab_blending(resized_volume.permute(0, 2, 1), 'YZ', slab_size, batch_size).permute(0, 2, 1)

        logging.info("Averaging final predictions from both axes.")
        prob_map_tensor = (pred_xz + pred_yz) / 2.0
        
        if smoothing_sigma and smoothing_sigma > 0:
            logging.info(f"Applying GPU-based 3D Gaussian smoothing with sigma={smoothing_sigma}...")
            prob_map_tensor = gpu_gaussian_blur_3d(prob_map_tensor, sigma=smoothing_sigma)
            
        logging.info(f"Resizing prediction back to original shape {original_shape}...")
        prob_map_np = F.interpolate(prob_map_tensor.unsqueeze(0).unsqueeze(0), size=original_shape, mode='trilinear', align_corners=False).squeeze().cpu().numpy()

        logging.info(f"Binarizing probability map with threshold {binarize_threshold}...")
        binary_mask_np = (prob_map_np > binarize_threshold).astype(np.uint8)
        
        if diagnostics_path:
            logging.info(f"Saving diagnostic (raw binarized) mask to {diagnostics_path}")
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            mrcfile.write(diagnostics_path, binary_mask_np, voxel_size=voxel_size, overwrite=True)
        
        try:
            final_mask = fit_and_generate_mask(binary_mask_np, downsample_grid_size)
        except (ValueError, RuntimeError) as e:
            logging.error(f"Plane fitting failed: {e}. Saving the raw binarized mask as the final output instead.")
            final_mask = binary_mask_np

        logging.info(f"Saving final mask to {output_mrc_path}")
        output_mrc_path.parent.mkdir(parents=True, exist_ok=True)
        mrcfile.write(output_mrc_path, final_mask, voxel_size=voxel_size, overwrite=True)
        
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info("Prediction complete.")

    def _predict_single_axis_with_slab_blending(self, volume_3d: torch.Tensor, axis: str, slab_size: int, batch_size: int) -> torch.Tensor:
        num_slices = volume_3d.shape[1]
        final_slices, hann_window = [], torch.hann_window(slab_size, periodic=False, device=self.device)
        for i in tqdm(range(num_slices), desc=f"Slab Blending ({axis} axis)", leave=False, ncols=80):
            half_slab = slab_size // 2
            start, end = max(0, i - half_slab), min(num_slices, i + half_slab + 1)
            predicted_slab = self._predict_raw_slab(volume_3d[:, start:end, :], batch_size)
            win_start, win_end = max(0, half_slab - i), min(slab_size, half_slab + (num_slices - i))
            current_window = hann_window[win_start:win_end]
            final_slice = torch.einsum('dwh,d->wh', predicted_slab, current_window) / current_window.sum()
            final_slices.append(final_slice)
        return torch.stack(final_slices, dim=1)

    def _predict_raw_slab(self, raw_slab: torch.Tensor, batch_size: int) -> torch.Tensor:
        views = raw_slab.permute(1, 0, 2)
        all_preds = []
        for i in range(0, len(views), batch_size):
            input_channels = [torch.stack([robust_normalization(v), local_variance_2d(robust_normalization(v), constants.LOCAL_VARIANCE_KERNEL_SIZE)], dim=0) for v in views[i:i+batch_size]]
            preds = torch.sigmoid(self.model(torch.stack(input_channels))).detach()
            all_preds.append(preds)
        return torch.cat(all_preds).squeeze(1)

def main():
    parser = argparse.ArgumentParser(description="Predict a slab mask by fitting planes to a deep learning segmentation.")
    parser.add_argument("--input_tomogram", type=Path, required=True, help="Path to the input tomogram (.mrc) file.")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to the trained model checkpoint (.ckpt) file.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the final, plane-fitted mask (.mrc) file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing 2D slices.")
    parser.add_argument("--slab_size", type=int, default=15, help="Size of the slab for blending (must be odd).")
    parser.add_argument("--smoothing_sigma", type=float, default=None, help="Optional: Sigma for GPU Gaussian filter. Try 1.0.")
    parser.add_argument("--binarize_threshold", type=float, default=0.5, help="Threshold for binarization. Default: 0.5.")
    parser.add_argument("--diagnostics", action="store_true", help="Save the intermediate binarized mask.")
    # --- NEW ARGUMENT FOR PERFORMANCE ---
    parser.add_argument("--downsample_grid_size", type=int, default=8, help="Grid size for point cloud downsampling before plane fitting. Larger values are faster but less precise. Default: 8.")

    args = parser.parse_args()
    if args.slab_size % 2 == 0: raise ValueError("Slab size must be an odd number.")
    diagnostics_path = args.output_path.with_name(f"{args.output_path.stem}_diagnostic{args.output_path.suffix}") if args.diagnostics else None
    predictor = TomogramPredictor(model_checkpoint_path=str(args.checkpoint_path))
    predictor.predict(input_mrc_path=args.input_tomogram, output_mrc_path=args.output_path, diagnostics_path=diagnostics_path, slab_size=args.slab_size, batch_size=args.batch_size, smoothing_sigma=args.smoothing_sigma, binarize_threshold=args.binarize_threshold, downsample_grid_size=args.downsample_grid_size)

if __name__ == "__main__":
    main()
