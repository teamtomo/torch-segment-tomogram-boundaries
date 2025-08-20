# torch-tomo-slab

[![License](https://img.shields.io/pypi/l/torch-tomo-slab.svg?color=green)](https://github.com/shahpnmlab/torch-tomo-slab/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-tomo-slab.svg?color=green)](https://pypi.org/project/torch-tomo-slab)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-tomo-slab.svg?color=green)](https://python.org)
[![CI](https://github.com/shahpnmlab/torch-tomo-slab/actions/workflows/ci.yml/badge.svg)](https://github.com/shahpnmlab/torch-tomo-slab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/shahpnmlab/torch-tomo-slab/branch/main/graph/badge.svg)](https://codecov.io/gh/shahpnmlab/torch-tomo-slab)

A simple Unet application to detect boundaries of tomographic volumes

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork shahpnmlab/torch-tomo-slab --clone
# or just
# gh repo clone shahpnmlab/torch-tomo-slab
cd torch-tomo-slab
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```

## API Usage

The library is designed around three core classes: `TrainingDataGenerator`, `TomoSlabTrainer`, and `TomoSlabPredictor`. The following demonstrates the standard workflow.

### Step 1: Prepare Training Data

First, you need to convert your 3D tomogram volumes and corresponding label masks into 2D slices suitable for training.

-   Place your tomograms (`.mrc` files) in the directory specified by `constants.REFERENCE_TOMOGRAM_DIR`.
-   Place your masks (`.mrc` files) in the directory specified by `constants.MASK_OUTPUT_DIR`.

Then, run the `TrainingDataGenerator`.

```python
from torch_tomo_slab.processing import TrainingDataGenerator
from torch_tomo_slab import constants

# This class reads its path configuration from the `constants.py` file.
# Ensure the paths in that file are correct for your system.
print(f"Reading tomograms from: {constants.REFERENCE_TOMOGRAM_DIR}")
print(f"Reading masks from: {constants.MASK_OUTPUT_DIR}")

# Initialize the generator
generator = TrainingDataGenerator()

# Run the data preparation pipeline
# This will process all volumes and save 2D .pt files to the
# directories specified by constants.TRAIN_DATA_DIR and constants.VAL_DATA_DIR.
generator.run()

print("Data preparation complete.")
```

### Step 2: Train a Segmentation Model

Once the data is prepared, you can train the model using the `TomoSlabTrainer`.

```python
from torch_tomo_slab.trainer import TomoSlabTrainer

# The trainer automatically loads its configuration from `config.py`.
# You can modify that file directly or override the settings (see note below).

# 1. Initialize the trainer
trainer = TomoSlabTrainer()

# 2. Start the training process
# The trainer will handle dataloading, model setup, and the training loop.
trainer.fit()

print("Training finished.")
```

### Step 3: Predict on a New Tomogram

After training, use the best model checkpoint to predict a slab mask for a new, unseen tomogram.

```python
from pathlib import Path
from torch_tomo_slab.predict import TomoSlabPredictor

# 1. Specify the path to your trained model checkpoint
checkpoint_path = "lightning_logs/Unet-resnet18/loss-weighted_bce/checkpoints/best-epoch=25-val_dice=0.98.ckpt"

# 2. Specify the path to the input tomogram you want to predict
input_tomo_path = Path("/path/to/your/new_tomogram.mrc")

# 3. Define where to save the output mask
output_mask_path = Path("/path/to/your/output/predicted_mask.mrc")

# 4. Initialize the predictor with the trained model
predictor = TomoSlabPredictor(model_checkpoint_path=checkpoint_path)

# 5. Run the full prediction pipeline
# This loads the tomogram, runs inference, fits planes, and saves the final mask.
final_mask = predictor.predict(
    input_tomogram=input_tomo_path,
    output_path=output_mask_path,
    binarize_threshold=0.5,
    downsample_grid_size=8
)

print(f"Prediction complete. Mask saved to {output_mask_path}")

```

### A Note on Overriding Default Configurations

The default hyperparameters (e.g., learning rate, batch size, model architecture) are stored in `src/torch_tomo_slab/config.py`, and file paths are in `src/torch_tomo_slab/constants.py`.

For quick experiments or when using this library as part of a larger application, it is best to **programmatically override** these settings in your script *before* you create instances of the library's classes.

Here is an example of a wrapper function that demonstrates this best practice:

```python
from torch_tomo_slab import config, constants
from torch_tomo_slab.trainer import TomoSlabTrainer
from pathlib import Path

def run_custom_experiment(lr, batch_size, encoder, data_dir):
    """
    A wrapper to run training with custom, non-default hyperparameters.
    """
    print("--- Applying custom configuration ---")

    # Overwrite config values before initializing the trainer
    config.LEARNING_RATE = lr
    config.BATCH_SIZE = batch_size
    config.MODEL_CONFIG['encoder_name'] = encoder
    
    # Overwrite constants (e.g., data paths)
    prepared_data_path = Path(data_dir)
    constants.TRAIN_DATA_DIR = prepared_data_path / "train"
    constants.VAL_DATA_DIR = prepared_data_path / "val"

    print(f"Set learning rate -> {config.LEARNING_RATE}")
    print(f"Set model encoder -> {config.MODEL_CONFIG['encoder_name']}")
    print(f"Set training data path -> {constants.TRAIN_DATA_DIR}")

    # Now, initialize and run the trainer
    # It will use the new values you just set.
    trainer = TomoSlabTrainer()
    trainer.fit()

# Example usage:
# run_custom_experiment(
#     lr=1e-5,
#     batch_size=16,
#     encoder='resnet50',
#     data_dir='/path/to/another/dataset'
# )
```