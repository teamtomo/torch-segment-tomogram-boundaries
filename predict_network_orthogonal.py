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
    
    logging.info("=== Running Full Inference (No Blending) ===")
    predictor = TomoSlabPredictor(model_checkpoint, compile_model=False)
    
    # Run full inference without blending (slab_size=1)
    full_result = predictor.predict(
        input_tomogram=input_tomogram,
        output_path=output_dir / "full_no_blending_mask.mrc",
        save_raw_mask_path=output_dir / "full_no_blending_raw.mrc",
        save_orthogonal_views_path=output_dir / "full_no_blending_orthogonal_views",
        slab_size=1,  # Disable blending
        batch_size=16
    )
    
    logging.info("=== Running Full Inference (With Blending) ===")
    # Run full inference with blending
    full_result_blended = predictor.predict(
        input_tomogram=input_tomogram,
        output_path=output_dir / "full_with_blending_mask.mrc",
        save_raw_mask_path=output_dir / "full_with_blending_raw.mrc",
        save_orthogonal_views_path=output_dir / "full_with_blending_orthogonal_views",
        slab_size=15,  # Enable blending
        batch_size=16
    )
    
    logging.info("=== Comparison Complete ===")
    logging.info("Files generated:")
    logging.info(f"  No blending: full_no_blending_*.mrc and orthogonal views in {output_dir / 'full_no_blending_orthogonal_views'}")
    logging.info(f"  With blending: full_with_blending_*.mrc and orthogonal views in {output_dir / 'full_with_blending_orthogonal_views'}")

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