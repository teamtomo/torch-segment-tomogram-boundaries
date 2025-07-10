import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
    TQDMProgressBar
)
# Add src directory to path to allow top-level imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch_tomo_slab import config
from torch_tomo_slab.pl_model import SegmentationModel
from torch_tomo_slab.data.dataloading import SegmentationDataModule


def run_training():
    """Main function to configure and run the training pipeline."""

    prepared_data_dir = Path(config.PREPARED_DATA_DIR)
    if not prepared_data_dir.exists() or not any(prepared_data_dir.iterdir()):
        print(f"Error: Prepared data directory is empty or does not exist: {prepared_data_dir}")
        print("Please run 'p02_data_preparation.py' script first.")
        sys.exit(1)
    all_pt_files = list(prepared_data_dir.glob("*.pt"))
    if len(all_pt_files) < 2:
        print(f"Error: Not enough data files ({len(all_pt_files)}) found for a train/validation split.")
        sys.exit(1)
    train_files, val_files = train_test_split(
        all_pt_files, test_size=config.VALIDATION_FRACTION, shuffle=True, random_state=42
    )
    print(f"Found {len(all_pt_files)} total samples.")
    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")

    datamodule = SegmentationDataModule(
        train_pt_files=train_files, val_pt_files=val_files, patch_size=(config.PATCH_SIZE, config.PATCH_SIZE),
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, samples_per_volume=config.SAMPLES_PER_VOLUME,
        alpha_for_dropping=config.ALPHA_FOR_DROPPING, val_patch_sampling=config.VALIDATION_PATCH_SAMPLING,
        overlap=config.OVERLAP
    )
    model = smp.create_model(
        arch=config.MODEL_ARCH, encoder_name=config.MODEL_ENCODER, classes=1, in_channels=2,
    )
    pl_model = SegmentationModel(
        model=model, learning_rate=config.LEARNING_RATE, loss_function=config.LOSS_FUNCTION,
    )

    # --- NEW: CONFIGURE AND INITIALIZE CALLBACKS ---
    print("\n--- Configuring Callbacks ---")

    # 1. Model Checkpointing: Save the best model based on validation loss
    checkpointer = ModelCheckpoint(
        monitor=config.MONITOR_METRIC,
        mode="min",  # "min" for loss, "max" for accuracy/dice
        filename=f"{config.MODEL_ARCH}-{config.MODEL_ENCODER}-best-{{epoch}}-{{{config.MONITOR_METRIC}:.4f}}",
        save_top_k=config.CHECKPOINT_SAVE_TOP_K,
        verbose=True,
    )

    # 2. Early Stopping: Stop training if validation loss doesn't improve
    early_stopper = EarlyStopping(
        monitor=config.MONITOR_METRIC,
        mode="min",
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        verbose=True,
    )

    # 3. Learning Rate Monitoring: Log the learning rate for inspection in TensorBoard
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks = [progress_bar, checkpointer, early_stopper, lr_monitor]

    # 5. Stochastic Weight Averaging (SWA): Optional callback for better generalization
    if config.USE_SWA:
        swa_start_epoch = int(config.MAX_EPOCHS * config.SWA_START_EPOCH_FRACTION)
        print(f"Enabling Stochastic Weight Averaging (SWA) starting at epoch {swa_start_epoch}.")
        swa = StochasticWeightAveraging(
            swa_lrs=config.SWA_LEARNING_RATE,
            swa_epoch_start=swa_start_epoch
        )
        callbacks.append(swa)

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        check_val_every_n_epoch=config.CHECK_VAL_EVERY_N_EPOCH,
        callbacks=callbacks
    )

    # 6. Start training
    print("\n--- Starting Training ---")
    trainer.fit(pl_model, datamodule=datamodule)
    print("--- Training Finished ---")

if __name__ == "__main__":
    run_training()