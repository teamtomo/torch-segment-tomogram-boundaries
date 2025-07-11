import sys
from pathlib import Path
import torch
import torch.nn as nn
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
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch_tomo_slab import config
from torch_tomo_slab.pl_model import SegmentationModel
from torch_tomo_slab.data.dataloading import SegmentationDataModule


class CombinedLoss(nn.Module):
    # ... (class definition is unchanged)
    def __init__(self, loss1: nn.Module, loss2: nn.Module, weight1: float = 0.5, weight2: float = 0.5):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2
        loss1_name = self.loss1.__class__.__name__
        loss2_name = self.loss2.__class__.__name__
        self.name = f"{self.weight1}*{loss1_name}_{self.weight2}*{loss2_name}"
    def forward(self, pred, target):
        loss_val1 = self.loss1(pred, target)
        loss_val2 = self.loss2(pred, target)
        return self.weight1 * loss_val1 + self.weight2 * loss_val2

def get_loss_function(name: str, weights: tuple = (0.5, 0.5)):
    # ... (function definition is unchanged)
    name = name.lower()
    if name == "diceloss":
        return smp.losses.DiceLoss(mode='binary', from_logits=True)
    elif name == "bcewithlogitsloss":
        return nn.BCEWithLogitsLoss()
    elif name == "dice+bce":
        print(f"Creating CombinedLoss with Dice/BCE weights: {weights}")
        dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        bce_loss = nn.BCEWithLogitsLoss()
        return CombinedLoss(dice_loss, bce_loss, weight1=weights[0], weight2=weights[1])
    else:
        raise ValueError(f"Unknown loss function: {name}")


def run_training():
    """Main function to configure and run the training pipeline."""
    # --- 1. DATA SETUP ---
    prepared_data_dir = Path(config.PREPARED_DATA_DIR)
    all_pt_files = sorted(list(prepared_data_dir.glob("*.pt")))
    # ... (error checking for empty dir remains the same)

    train_files, val_files = train_test_split(
        all_pt_files, test_size=config.VALIDATION_FRACTION, shuffle=True, random_state=42
    )
    print(f"Found {len(all_pt_files)} total 2D sections.")
    print(f"Training sections: {len(train_files)}, Validation sections: {len(val_files)}")

    datamodule = SegmentationDataModule(
        train_pt_files=train_files, val_pt_files=val_files, patch_size=(config.PATCH_SIZE, config.PATCH_SIZE),
        overlap=config.OVERLAP, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        samples_per_volume=config.SAMPLES_PER_VOLUME, alpha_for_dropping=config.ALPHA_FOR_DROPPING,
        val_patch_sampling=config.VALIDATION_PATCH_SAMPLING
    )

    # --- 2. MODEL AND LOSS FUNCTION SETUP ---
    model = smp.create_model(
        arch=config.MODEL_ARCH, encoder_name=config.MODEL_ENCODER, classes=1, in_channels=2
    )
    loss_fn = get_loss_function(config.LOSS_FUNCTION, config.LOSS_WEIGHTS)

    # --- THIS IS THE FIX ---
    # Inject the number of training samples directly into the model.
    pl_model = SegmentationModel(
        model=model,
        loss_function=loss_fn,
        num_train_samples=len(train_files),  # <-- INJECT DEPENDENCY HERE
        learning_rate=config.LEARNING_RATE,
    )

    # --- 3. CALLBACKS & TRAINER CONFIGURATION ---
    # ... (this section is unchanged)
    print("\n--- Configuring Callbacks ---")
    checkpointer = ModelCheckpoint(
        monitor=config.MONITOR_METRIC, mode="min",
        filename=f"{config.MODEL_ARCH}-{config.MODEL_ENCODER}-best-{{epoch}}-{{{config.MONITOR_METRIC}:.4f}}",
        save_top_k=config.CHECKPOINT_SAVE_TOP_K, verbose=True,
    )
    early_stopper = EarlyStopping(
        monitor=config.MONITOR_METRIC, mode="min", patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks = [progress_bar, checkpointer, early_stopper, lr_monitor]

    if config.USE_SWA:
        swa_start_epoch = int(config.MAX_EPOCHS * config.SWA_START_EPOCH_FRACTION)
        print(f"Enabling Stochastic Weight Averaging (SWA) starting at epoch {swa_start_epoch}.")
        swa = StochasticWeightAveraging(swa_lrs=config.SWA_LEARNING_RATE, swa_epoch_start=swa_start_epoch)
        callbacks.append(swa)

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS, accelerator=config.ACCELERATOR, devices=config.DEVICES,
        precision=config.PRECISION, log_every_n_steps=config.LOG_EVERY_N_STEPS,
        check_val_every_n_epoch=config.CHECK_VAL_EVERY_N_EPOCH, callbacks=callbacks
    )

    # --- 4. START TRAINING ---
    # ... (this section is unchanged)
    print("\n--- Starting Training ---")
    trainer.fit(pl_model, datamodule=datamodule)
    print("--- Training Finished ---")
    print(f"Best model checkpoint saved at: {checkpointer.best_model_path}")
    if config.USE_SWA:
        swa_model_path = Path(trainer.logger.log_dir) / "swa_model.pth"
        torch.save(pl_model.model.state_dict(), swa_model_path)
        print(f"SWA-averaged model weights saved to: {swa_model_path}")

if __name__ == "__main__":
    run_training()