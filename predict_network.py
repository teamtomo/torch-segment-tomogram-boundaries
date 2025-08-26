"""
Inference script for torch-tomo-slab network.

This script demonstrates how to use the torch-tomo-slab library to:
1. Load a trained segmentation model
2. Run inference on a 3D tomogram
3. Save the resulting slab mask

Usage:
    python predict_network.py
"""
from pathlib import Path
from torch_tomo_slab.predict import TomoSlabPredictor

def predict_mask(model_checkpoint: Path, input_tomogram: Path, output_path: Path, save_raw_mask_path: Path = None):
    """
    Runs the prediction pipeline.

    Args:
        model_checkpoint (Path): Path to the trained model checkpoint.
        input_tomogram (Path): Path to the input tomogram.
        output_path (Path): Path to save the final mask.
        save_raw_mask_path (Path, optional): Path to save the raw mask. Defaults to None.
    """
    print("\n=== Running Inference ===")

    predictor = TomoSlabPredictor(model_checkpoint)
    predictor.predict(
        input_tomogram=input_tomogram,
        output_path=output_path,
        save_raw_mask_path=save_raw_mask_path
    )
    print(f"✓ Inference complete. Mask saved to {output_path}")

if __name__ == "__main__":
    # Example usage:
    model_checkpoint = Path("/home/pranav/Desktop/pranav/data/training/torch-tomo-slab/lightning_logs/Unet-resnet18/loss-dice_weighted_bce/version_0/checkpoints/best-epoch=27-val_dice=0.9119.ckpt")
    input_tomogram = Path("/home/pranav/Desktop/pranav/data/training/warp_apo/warp_tiltseries/reconstruction/TS_32_10.00Apx.mrc")
    output_path = Path("tomo_mask/TS_32_8.00Apx.mrc")
    raw_mask_path = Path("tomo_mask/TS_32_8.00Apx_diagnostic.mrc") # Optional

    predict_mask(
        model_checkpoint=model_checkpoint,
        input_tomogram=input_tomogram,
        output_path=output_path,
        save_raw_mask_path=raw_mask_path
    )