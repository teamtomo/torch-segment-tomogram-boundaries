import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import OneCycleLR
from torch_tomo_slab import config
import torch.nn as nn


class CombinedLoss(nn.Module):
    """
    A loss function that is a weighted sum of two individual losses.
    """

    def __init__(self, loss1, loss2, weight1=0.5, weight2=0.5):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2
        loss1_name = self.loss1.__class__.__name__
        loss2_name = self.loss2.__class__.__name__

        self.name = f"{self.weight1}*{loss1_name} + {self.weight2}*{loss2_name}"

    def forward(self, pred, target):
        loss_val1 = self.loss1(pred, target)
        loss_val2 = self.loss2(pred, target)
        return self.weight1 * loss_val1 + self.weight2 * loss_val2

class SegmentationModel(pl.LightningModule):
    """PyTorch Lightning module for 2-channel segmentation."""

    def __init__(
            self,
            model: torch.nn.Module,
            learning_rate: float = 1e-3,
            loss_function: str = 'dice',
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model'])

        self.total_steps = 0
        if loss_function == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
        elif loss_function == 'bce':
            self.criterion = smp.losses.SoftBCEWithLogitsLoss()
        elif loss_function == 'dice+bce':
            dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
            bce_loss = nn.BCEWithLogitsLoss()  # Using the standard torch loss
            self.criterion = CombinedLoss(dice_loss, bce_loss, weight1=0.5, weight2=0.5)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}. "
                             f"Available options: 'dice', 'bce', 'dice+bce'")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # TorchIO batch structure
        image = batch['image'][tio.DATA]  # Shape: (B, 2, H, W, 1)
        label = batch['label'][tio.DATA]  # Shape: (B, 1, H, W, 1)

        # Remove the singleton dimension
        image = image.squeeze(-1)  # (B, 2, H, W)
        label = label.squeeze(-1)  # (B, 1, H, W)

        # Forward pass
        pred = self(image)  # (B, 1, H, W)
        loss = self.criterion(pred, label.to(torch.float32))

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image'][tio.DATA].squeeze(-1)
        label = batch['label'][tio.DATA].squeeze(-1)

        pred = self(image)
        loss = self.criterion(pred, label.to(torch.float32))

        # Calculate metrics
        # Apply sigmoid to logits to get probabilities for metric calculation
        pred_probs = torch.sigmoid(pred)
        pred_binary = (pred_probs > 0.5).float()
        dice = self.dice_coefficient(pred_binary, label.to(torch.float32))

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)

    def dice_coefficient(self, pred, target, smooth=1e-5):
        """
        Metric calculation. This is separate from the loss function.
        It expects a binarized prediction.
        """
        # Ensure intersection and union are calculated on the same device
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2 * intersection + smooth) / (union + smooth)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if not config.USE_LR_SCHEDULER:
            return optimizer

        datamodule = self.trainer.datamodule
        num_train_images = len(datamodule.train_pt_files)
        total_patches_per_epoch = num_train_images * config.SAMPLES_PER_VOLUME
        steps_per_epoch = np.ceil(total_patches_per_epoch / config.BATCH_SIZE)
        total_steps = int(steps_per_epoch * config.MAX_EPOCHS)

        print(f"Manually calculated total steps for OneCycleLR: {total_steps}")
        if total_steps <= 0:
            raise ValueError(
                f"Calculated total_steps is not positive: {total_steps}. "
                "Check your training data and config."
            )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=config.SCHEDULER_WARMUP_EPOCHS / config.MAX_EPOCHS,
            anneal_strategy='cos',
            final_div_factor=self.learning_rate / config.SCHEDULER_MIN_LR,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }