"""
Inference script for torch-tomo-slab network.

This script demonstrates how to use the torch-tomo-slab library to:
1. Load a trained segmentation model.
2. Run inference on a 3D tomogram to get a binary mask.
3. Save the resulting mask.

This script uses the simple functional API. For running inference on multiple
tomograms, it is more efficient to instantiate the `TomoSlabPredictor` class
directly to avoid reloading the model for each prediction.

For advanced post-processing, such as fitting planes to the binary mask, please
see the `plane_fitting.py` example script.

Usage:
    python predict_network.py
"""
from pathlib import Path

import mrcfile
import numpy as np

from torch_segment_tomogram_boundaries import config
from torch_segment_tomogram_boundaries.predict import predict_binary, predict_probabilities


def run_prediction(
    model_checkpoint: Path,
    input_tomogram: Path,
    output_binary_path: Path,
    output_probs_path: Path = None,
):
    """
    Runs the prediction pipeline and saves the output masks.

    Args:
        model_checkpoint (Path): Path to the trained model checkpoint.
        input_tomogram (Path): Path to the input tomogram.
        output_binary_path (Path): Path to save the final binary mask.
        output_probs_path (Path, optional): Path to save the raw probability map.
                                            Defaults to None.
    """
    print("\n=== Running Inference ===")

    # 1. Generate and save the binary mask using the functional API
    print("1. Generating binary mask...")
    binary_mask = predict_binary(
        model_checkpoint_path=model_checkpoint,
        input_tomogram=input_tomogram,
        binarize_threshold=0.5, # This is the default value
        # Optional: Adjust parameters for performance/quality trade-off
        # slab_size=15,
        # batch_size=16,
    )
    print(f"Saving binary mask to {output_binary_path}...")
    mrcfile.write(output_binary_path, binary_mask.astype(np.float32), overwrite=True)

    # 2. (Optional) Generate and save the probability map
    if output_probs_path:
        print("2. Generating probability map...")
        prob_map = predict_probabilities(
            model_checkpoint_path=model_checkpoint,
            input_tomogram=input_tomogram,
            # slab_size=15,
            # batch_size=16,
            # smoothing_sigma=1.0,
        )
        print(f"Saving probability map to {output_probs_path}...")
        mrcfile.write(output_probs_path, prob_map, overwrite=True)

    print(f"\n✓ Inference complete.")
    print(f"- Binary mask saved to: {output_binary_path}")
    if output_probs_path:
        print(f"- Probability map saved to: {output_probs_path}")


if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: You must replace this with the actual path to your trained model checkpoint.
    # You can find your trained models under the directory specified by CKPT_SAVE_PATH in config.py.
    # Example path structure:
    # /path/to/checkpoints/unet-monai-loss-weighted_bce/version_0/checkpoints/best-epoch=....ckpt
    model_checkpoint = Path("path/to/your/best-model.ckpt")

    # NOTE: Replace with the path to the tomogram you want to process.
    # An example file 'example_tomogram.mrc' is expected in the TOMOGRAM_DIR.
    input_tomogram = config.TOMOGRAM_DIR / "example_tomogram.mrc"

    # Define output paths
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    output_binary_mask_path = output_dir / f"{input_tomogram.stem}_binary_mask.mrc"
    output_probs_path = output_dir / f"{input_tomogram.stem}_probabilities.mrc"  # Optional

    # --- Execution ---
    print(f"INFO: Using model checkpoint: {model_checkpoint}")
    print(f"INFO: Processing tomogram: {input_tomogram}")

    if "path/to/your" in str(model_checkpoint) or not model_checkpoint.exists():
        print(f"\nERROR: Model checkpoint not found or placeholder path is not updated.")
        print("Please update the 'model_checkpoint' variable in this script with the correct path to your .ckpt file.")
    elif not input_tomogram.exists():
        print(f"\nERROR: Input tomogram not found at '{input_tomogram}'")
        print(f"Please place a tomogram at that location or update the 'input_tomogram' variable.")
    else:
        run_prediction(
            model_checkpoint=model_checkpoint,
            input_tomogram=input_tomogram,
            output_binary_path=output_binary_mask_path,
            output_probs_path=output_probs_path, # Set to None to skip saving
        )