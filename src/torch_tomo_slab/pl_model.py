import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_tomo_slab import config

class SegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for 2-channel segmentation.
    This module is agnostic to the specific loss function used.
    """

    def __init__(
            self,
            model: nn.Module,
            loss_function: nn.Module,
            num_train_samples: int,
            learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_function'])
        self.model = model
        self.criterion = loss_function
        loss_name = getattr(self.criterion, 'name', self.criterion.__class__.__name__)
        self.hparams.loss_function_name = loss_name
        print(f"Using loss function: {loss_name}")

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage: str):
        """Shared logic for training and validation steps."""
        image = batch['image']
        label = batch['label']

        # Get the batch size for logging
        batch_size = image.size(0)

        if image.dim() == 5:
            image = image.squeeze(-1)
        if label.dim() == 5:
            label = label.squeeze(-1)

        pred_logits = self(image)
        loss = self.criterion(pred_logits, label.to(torch.float32))

        # --- THIS IS THE FIX ---
        # Explicitly provide the batch_size to self.log to remove the warning
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=(stage=='train'), on_epoch=True, batch_size=batch_size)

        if stage == 'val':
            pred_probs = torch.sigmoid(pred_logits)
            pred_binary = (pred_probs > 0.5).float()
            dice = self.dice_coefficient(pred_binary, label)
            self.log('val_dice', dice, prog_bar=True, on_epoch=True, batch_size=batch_size)

        return loss

    # ... (the rest of the file is unchanged) ...
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def dice_coefficient(self, pred, target, smooth=1e-5):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = torch.mean((2. * intersection + smooth) / (union + smooth))
        return dice

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if not config.USE_LR_SCHEDULER:
            return optimizer
        total_patches_per_epoch = self.hparams.num_train_samples * config.SAMPLES_PER_VOLUME
        steps_per_epoch = np.ceil(total_patches_per_epoch / config.BATCH_SIZE)
        total_steps = int(steps_per_epoch * self.trainer.max_epochs)
        print(f"Scheduler configured with total steps: {total_steps}")
        if total_steps <= 0:
            raise ValueError(f"Calculated total_steps is not positive: {total_steps}.")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps,
            pct_start=config.SCHEDULER_WARMUP_EPOCHS / self.trainer.max_epochs,
            anneal_strategy='cos',
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": { "scheduler": scheduler, "interval": "step" },
        }