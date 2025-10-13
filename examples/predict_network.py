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

from torch_tomo_slab import config
from torch_tomo_slab.predict import predict


def predict_mask(
    model_checkpoint: Path,
    input_tomogram: Path,
    output_path: Path,
    save_raw_mask_path: Path = None,
):
    """
    Runs the prediction pipeline.

    Args:
        model_checkpoint (Path): Path to the trained model checkpoint.
        input_tomogram (Path): Path to the input tomogram.
        output_path (Path): Path to save the final mask.
        save_raw_mask_path (Path, optional): Path to save the raw mask. Defaults to None.
    """
    print("\n=== Running Inference ===")

    predict(
        model_checkpoint_path=model_checkpoint,
        input_tomogram=input_tomogram,
        output_path=output_path,
        save_raw_mask_path=save_raw_mask_path,
    )
    print(f"✓ Inference complete. Mask saved to {output_path}")


if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: You must replace this with the actual path to your trained model checkpoint.
    # You can find your trained models under the directory specified by CKPT_SAVE_PATH in config.py.
    # Example path structure:
    # /path/to/checkpoints/Unet-vgg11--loss-weighted_bce_weighted_huber_with_gradient/version_0/checkpoints/best-epoch=....ckpt
    model_checkpoint = Path("path/to/your/best-model.ckpt")

    # NOTE: Replace with the path to the tomogram you want to process.
    # An example file 'example_tomogram.mrc' is expected in the TOMOGRAM_DIR.
    input_tomogram = config.TOMOGRAM_DIR / "example_tomogram.mrc"

    # Define output paths
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{input_tomogram.stem}_mask.mrc"
    raw_mask_path = output_dir / f"{input_tomogram.stem}_raw_mask.mrc"  # Optional

    # --- Execution ---
    print(f"INFO: Using model checkpoint: {model_checkpoint}")
    print(f"INFO: Processing tomogram: {input_tomogram}")

    if "path/to/your" in str(model_checkpoint) or not model_checkpoint.exists():
        print(f"ERROR: Model checkpoint not found or placeholder path is not updated.")
        print("Please update the 'model_checkpoint' variable in this script with the correct path to your .ckpt file.")
elif not input_tomogram.exists():
        print(f"ERROR: Input tomogram not found at '{input_tomogram}'")
        print(f"Please place a tomogram at that location or update the 'input_tomogram' variable.")
    else:
        predict_mask(
            model_checkpoint=model_checkpoint,
            input_tomogram=input_tomogram,
            output_path=output_path,
            save_raw_mask_path=raw_mask_path,
        )
