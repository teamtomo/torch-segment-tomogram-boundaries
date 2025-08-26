"""
Inference script for torch-tomo-slab network - Orthogonal Views Analysis.

This script extracts two orthogonal views from a tomogram and runs inference
on individual slices to isolate whether jagged appearance comes from blending
or the inference itself.

Usage:
    python predict_network_orthogonal.py
"""
import logging
import numpy as np
import torch
import mrcfile
from pathlib import Path
from torch_tomo_slab.predict import TomoSlabPredictor
from torch_tomo_slab.utils import threeD
from torch_tomo_slab.utils.twoD import robust_normalization, local_variance_2d

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_orthogonal_views(volume: np.ndarray, center_fraction: float = 0.5):
    """
    Extract two orthogonal views from the center of the volume.
    
    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume of shape (D, H, W).
    center_fraction : float
        Fraction of volume center to extract view from (0.5 = middle).
    
    Returns
    -------
    tuple
        (xz_view, yz_view) - Two orthogonal 2D slices.
    """
    D, H, W = volume.shape
    
    # Extract XZ view (slice along Y-axis at center)
    y_center = int(H * center_fraction)
    xz_view = volume[:, y_center, :]  # Shape: (D, W)
    
    # Extract YZ view (slice along X-axis at center)  
    x_center = int(W * center_fraction)
    yz_view = volume[:, :, x_center]  # Shape: (D, H)
    
    logging.info(f"Extracted XZ view at Y={y_center}: shape {xz_view.shape}")
    logging.info(f"Extracted YZ view at X={x_center}: shape {yz_view.shape}")
    
    return xz_view, yz_view

def predict_single_slice(predictor: TomoSlabPredictor, slice_2d: np.ndarray, axis_name: str):
    """
    Run inference on a single 2D slice.
    
    Parameters
    ----------
    predictor : TomoSlabPredictor
        Initialized predictor instance.
    slice_2d : np.ndarray
        Input 2D slice.
    axis_name : str
        Name for logging (e.g., 'XZ', 'YZ').
    
    Returns
    -------
    np.ndarray
        Predicted probability map for the slice.
    """
    logging.info(f"Running inference on {axis_name} slice...")
    
    # Convert to tensor and move to device
    slice_tensor = torch.from_numpy(slice_2d.astype(np.float32)).to(predictor.device)
    
    # Apply preprocessing (same as training pipeline)
    slice_norm = robust_normalization(slice_tensor)
    slice_var = local_variance_2d(slice_norm)
    
    # Create 2-channel input
    two_channel_input = torch.stack([slice_norm, slice_var], dim=0)  # Shape: (2, D, W)
    batch_input = two_channel_input.unsqueeze(0)  # Add batch dimension: (1, 2, D, W)
    
    # Run inference
    with torch.no_grad():
        pred = predictor.model(batch_input)
        pred_prob = torch.sigmoid(pred).squeeze(0).squeeze(0)  # Remove batch and channel dims
    
    return pred_prob.cpu().numpy()

def analyze_orthogonal_views(model_checkpoint: Path, input_tomogram: Path, output_dir: Path):
    """
    Extract orthogonal views and run inference to analyze jagged artifacts.
    
    Parameters
    ----------
    model_checkpoint : Path
        Path to trained model checkpoint.
    input_tomogram : Path
        Path to input tomogram.
    output_dir : Path
        Directory to save analysis results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading tomogram from {input_tomogram}")
    
    # Load tomogram
    with mrcfile.open(input_tomogram, permissive=True) as mrc:
        original_data = mrc.data.astype(np.float32)
        voxel_size = mrc.voxel_size.copy()
    
    logging.info(f"Original tomogram shape: {original_data.shape}")
    
    # Initialize predictor
    logging.info(f"Loading model from {model_checkpoint}")
    predictor = TomoSlabPredictor(model_checkpoint, compile_model=False)  # Disable compilation for analysis
    
    # Resize volume to model's expected shape
    target_shape = predictor.target_shape_3d
    logging.info(f"Resizing to target shape: {target_shape}")
    
    resized_volume_tensor = threeD.resize_and_pad_3d(
        torch.from_numpy(original_data), 
        target_shape=target_shape, 
        mode='image'
    )
    resized_volume = resized_volume_tensor.numpy()
    
    # Extract orthogonal views
    xz_view, yz_view = extract_orthogonal_views(resized_volume)
    
    # Save original views
    mrcfile.write(output_dir / "xz_original.mrc", xz_view, voxel_size=voxel_size, overwrite=True)
    mrcfile.write(output_dir / "yz_original.mrc", yz_view, voxel_size=voxel_size, overwrite=True)
    
    # Run inference on each view
    xz_pred = predict_single_slice(predictor, xz_view, "XZ")
    yz_pred = predict_single_slice(predictor, yz_view, "YZ")
    
    # Save predicted probability maps
    mrcfile.write(output_dir / "xz_prediction.mrc", xz_pred, voxel_size=voxel_size, overwrite=True)
    mrcfile.write(output_dir / "yz_prediction.mrc", yz_pred, voxel_size=voxel_size, overwrite=True)
    
    # Save binarized versions for comparison
    threshold = 0.5
    xz_binary = (xz_pred > threshold).astype(np.uint8)
    yz_binary = (yz_pred > threshold).astype(np.uint8)
    
    mrcfile.write(output_dir / "xz_binary.mrc", xz_binary, voxel_size=voxel_size, overwrite=True)
    mrcfile.write(output_dir / "yz_binary.mrc", yz_binary, voxel_size=voxel_size, overwrite=True)
    
    # Analysis summary
    logging.info("=== Analysis Complete ===")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"XZ view prediction range: [{xz_pred.min():.4f}, {xz_pred.max():.4f}]")
    logging.info(f"YZ view prediction range: [{yz_pred.min():.4f}, {yz_pred.max():.4f}]")
    logging.info(f"XZ binary pixels: {np.sum(xz_binary)} / {xz_binary.size} ({100*np.sum(xz_binary)/xz_binary.size:.1f}%)")
    logging.info(f"YZ binary pixels: {np.sum(yz_binary)} / {yz_binary.size} ({100*np.sum(yz_binary)/yz_binary.size:.1f}%)")
    
    return {
        'xz_pred': xz_pred,
        'yz_pred': yz_pred,
        'xz_binary': xz_binary,
        'yz_binary': yz_binary
    }

def compare_with_full_inference(model_checkpoint: Path, input_tomogram: Path, output_dir: Path):
    """
    Run both orthogonal view analysis and full inference for comparison.
    
    Parameters
    ----------
    model_checkpoint : Path
        Path to trained model checkpoint.
    input_tomogram : Path
        Path to input tomogram.
    output_dir : Path
        Directory to save comparison results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=== Running Orthogonal View Analysis ===")
    orthogonal_results = analyze_orthogonal_views(model_checkpoint, input_tomogram, output_dir / "orthogonal")
    
    logging.info("=== Running Full Inference (No Blending) ===")
    predictor = TomoSlabPredictor(model_checkpoint, compile_model=False)
    
    # Run full inference without blending (slab_size=1)
    full_result = predictor.predict(
        input_tomogram=input_tomogram,
        output_path=output_dir / "full_no_blending_mask.mrc",
        save_raw_mask_path=output_dir / "full_no_blending_raw.mrc",
        slab_size=1,  # Disable blending
        batch_size=16
    )
    
    logging.info("=== Running Full Inference (With Blending) ===")
    # Run full inference with blending
    full_result_blended = predictor.predict(
        input_tomogram=input_tomogram,
        output_path=output_dir / "full_with_blending_mask.mrc",
        save_raw_mask_path=output_dir / "full_with_blending_raw.mrc",
        slab_size=15,  # Enable blending
        batch_size=16
    )
    
    logging.info("=== Comparison Complete ===")
    logging.info("Files generated:")
    logging.info(f"  Orthogonal views: {output_dir / 'orthogonal'}")
    logging.info(f"  No blending: full_no_blending_*.mrc")
    logging.info(f"  With blending: full_with_blending_*.mrc")
    
    return orthogonal_results

if __name__ == "__main__":
    # Configuration
    model_checkpoint = Path("/home/pranav/Desktop/pranav/data/training/torch-tomo-slab/lightning_logs/Unet-resnet18/loss-dice_weighted_bce/version_0/checkpoints/best-epoch=27-val_dice=0.9119.ckpt")
    input_tomogram = Path("/home/pranav/Desktop/pranav/data/training/warp_apo/warp_tiltseries/reconstruction/TS_32_10.00Apx.mrc")
    output_dir = Path("orthogonal_analysis")
    
    # Run comprehensive comparison
    compare_with_full_inference(
        model_checkpoint=model_checkpoint,
        input_tomogram=input_tomogram,
        output_dir=output_dir
    )