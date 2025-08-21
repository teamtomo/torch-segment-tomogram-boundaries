"""Advanced PyTorch Lightning callbacks for intelligent training management.

This module provides sophisticated callback implementations that go beyond
standard early stopping to implement adaptive training strategies like
dynamic SWA triggering and EMA-based performance monitoring.
"""
import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import StochasticWeightAveraging

log = logging.getLogger(__name__)

class DynamicTrainingManager(pl.Callback):
    """
    Dynamic training management callback with adaptive SWA and early stopping.
    
    This callback implements an intelligent training management strategy that:
    1. Monitors validation metrics using exponential moving average (EMA) to reduce noise
    2. Automatically triggers Stochastic Weight Averaging (SWA) when performance plateaus
    3. Applies early stopping after SWA has been activated if no improvement occurs
    
    The callback uses a two-phase approach: first allowing the model to train normally,
    then switching to SWA when improvement stagnates, and finally stopping early if
    SWA doesn't help.
    
    Parameters
    ----------
    monitor : str, default='val_dice'
        Metric name to monitor for training decisions.
    mode : str, default='max'
        Whether to maximize ('max') or minimize ('min') the monitored metric.
    ema_alpha : float, default=0.3
        Smoothing factor for exponential moving average (higher = more responsive).
    trigger_swa_patience : int, default=3
        Number of epochs without improvement before triggering SWA.
    early_stop_patience : int, default=10
        Number of epochs without improvement after SWA before early stopping.
    min_delta : float, default=1e-4
        Minimum change required to be considered an improvement.
        
    Attributes
    ----------
    ema_score : float or None
        Current exponential moving average of the monitored metric.
    swa_triggered : bool
        Whether SWA has been activated.
    swa_wait_count : int
        Counter for epochs since last improvement (before SWA).
    stop_wait_count : int
        Counter for epochs since last improvement (after SWA).
    """

    def __init__(
        self,
        monitor: str = "val_dice",
        mode: str = "max",
        ema_alpha: float = 0.3,
        trigger_swa_patience: int = 3,
        early_stop_patience: int = 10,
        min_delta: float = 1e-4,
    ) -> None:
        """
        Initialize dynamic training manager.
        
        Parameters are documented in the class docstring.
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.ema_alpha = ema_alpha
        self.trigger_swa_patience = trigger_swa_patience
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta

        self.ema_score = None
        self.swa_triggered = False
        self.swa_wait_count = 0
        self.stop_wait_count = 0

        if self.mode == "max":
            self.monitor_op = torch.gt
            self.best_score = -torch.inf
        else:
            self.monitor_op = torch.lt
            self.best_score = torch.inf

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Callback executed after each validation epoch to manage training dynamics.
        
        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The model being trained.
        """
        if trainer.sanity_checking:
            return

        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        # Initialize EMA on the first validation epoch
        if self.ema_score is None:
            self.ema_score = current_score
            self.best_score = current_score
            return

        # Update EMA score
        self.ema_score = self.ema_alpha * current_score + (1 - self.ema_alpha) * self.ema_score

        # --- SWA Trigger Logic ---
        if not self.swa_triggered:
            # Check if the current score is close to its EMA, indicating a plateau
            if abs(current_score - self.ema_score) < self.min_delta:
                self.swa_wait_count += 1
            else:
                self.swa_wait_count = 0 # Reset if we see a significant jump

            if self.swa_wait_count >= self.trigger_swa_patience:
                self.trigger_swa(trainer, pl_module.current_epoch)

        # --- Early Stopping Logic (only active after SWA has started) ---
        else:
            # Check for improvement over the best score seen so far
            if self.monitor_op(current_score, self.best_score + self.min_delta):
                self.best_score = current_score
                self.stop_wait_count = 0
            else:
                self.stop_wait_count += 1

            if self.stop_wait_count >= self.early_stop_patience:
                self.trigger_early_stopping(trainer)

    def trigger_swa(self, trainer: pl.Trainer, current_epoch: int) -> None:
        """
        Activate Stochastic Weight Averaging at the current epoch.
        
        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        current_epoch : int
            Current training epoch number.
        """
        swa_callback = None
        for cb in trainer.callbacks:
            if isinstance(cb, StochasticWeightAveraging):
                swa_callback = cb
                break

        if swa_callback:
            # Manually activate the SWA callback
            swa_callback._swa_epoch_start = current_epoch
            self.swa_triggered = True
            log.info(f"\n[DynamicTrainingManager] Validation metric plateaued. Triggering SWA at epoch {current_epoch}.")
            # Reset best_score and patience to give SWA a fair chance to improve the model
            self.best_score = self.ema_score
            self.stop_wait_count = 0

    def trigger_early_stopping(self, trainer: pl.Trainer) -> None:
        """
        Stop training early due to lack of improvement after SWA activation.
        
        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        """
        trainer.should_stop = True
        log.info(f"\n[DynamicTrainingManager] Stopping training early at epoch {trainer.current_epoch} as monitored metric '{self.monitor}' has not improved for {self.early_stop_patience} epochs.")
