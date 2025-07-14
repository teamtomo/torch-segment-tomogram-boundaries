import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch_tomo_slab import config

class SegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for 2-channel segmentation.
    This version robustly configures the LR scheduler by using the `setup` hook
    to determine the number of training steps per epoch.
    """

    def __init__(
            self,
            model: nn.Module,
            loss_function: nn.Module,
            learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_function'])
        self.model = model
        self.criterion = loss_function
        # --- NEW: A placeholder for steps_per_epoch ---
        self.steps_per_epoch = None

    def setup(self, stage: str):
        """
        --- THIS IS THE FIX ---
        This hook is called after the datamodule has been prepared.
        We use it to calculate the number of steps per epoch and store it.
        """
        if stage == 'fit':
            # Calculate the number of training batches
            train_loader = self.trainer.datamodule.train_dataloader()
            self.steps_per_epoch = len(train_loader)
            print(f"Detected {self.steps_per_epoch} steps per epoch for the scheduler.")


    def forward(self, x):
        return self.model(x)

    # _common_step, training_step, validation_step, _log_validation_images, dice_coefficient...
    # ... all remain exactly the same as the previous version ...
    def _common_step(self, batch, batch_idx, stage: str):
        image = batch['image']
        label = batch['label']
        batch_size = image.size(0)
        if image.dim() == 5: image = image.squeeze(-1)
        if label.dim() == 5: label = label.squeeze(-1)
        pred_logits = self(image)
        loss = self.criterion(pred_logits, label.to(torch.float32))
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=(stage=='train'), on_epoch=True, batch_size=batch_size, sync_dist=True)
        return loss, pred_logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_logits = self._common_step(batch, batch_idx, "val")
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > 0.5).float()
        dice = self.dice_coefficient(pred_binary, batch['label'])
        self.log('val_dice', dice, prog_bar=True, on_epoch=True, batch_size=batch['image'].size(0), sync_dist=True)
        if batch_idx == 0 and self.trainer.is_global_zero:
            self._log_validation_images(batch, pred_probs)
        return loss

    def _log_validation_images(self, batch: dict, pred_probs: torch.Tensor):
        if self.logger is None or not hasattr(self.logger.experiment, 'add_image'): return
        image, label = batch['image'], batch['label']
        num_images_to_log = min(8, image.size(0))
        grid_params = {"padding": 2, "pad_value": 1.0, "nrow": 4}
        input_grid = torchvision.utils.make_grid(image[:num_images_to_log, 0:1, :, :], **grid_params)
        self.logger.experiment.add_image("Validation/Input (Channel 1)", input_grid, self.current_epoch)
        label_grid = torchvision.utils.make_grid(label[:num_images_to_log].to(torch.float32), **grid_params)
        self.logger.experiment.add_image("Validation/Ground Truth", label_grid, self.current_epoch)
        pred_grid = torchvision.utils.make_grid(pred_probs[:num_images_to_log].to(torch.float32), **grid_params)
        self.logger.experiment.add_image("Validation/Prediction", pred_grid, self.current_epoch)

    def dice_coefficient(self, pred, target, smooth=1e-5):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = torch.mean((2. * intersection + smooth) / (union + smooth))
        return dice


    def configure_optimizers(self):
        """
        Now correctly uses the steps_per_epoch calculated in the `setup` hook.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        if not config.USE_LR_SCHEDULER:
            return optimizer

        # --- THIS IS THE CORRECTED PART ---
        if self.steps_per_epoch is None:
            raise RuntimeError("steps_per_epoch not set. This should be set in the `setup` hook.")

        total_steps = self.steps_per_epoch * self.trainer.max_epochs
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps,  # Now we pass the explicitly calculated value
            pct_start=config.SCHEDULER_WARMUP_EPOCHS / self.trainer.max_epochs,
            anneal_strategy='cos',
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": { "scheduler": scheduler, "interval": "step" },
        }
