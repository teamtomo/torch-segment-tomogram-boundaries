# src/torch_tomo_slab/trainer.py
import os
from pathlib import Path
from typing import List

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging, \
    TQDMProgressBar

from . import config, constants
from .pl_model import SegmentationModel
from .data.dataloading import SegmentationDataModule
from .losses import get_loss_function
from .callbacks import DynamicTrainingManager


class TomoSlabTrainer:
    """
    Encapsulates the entire training pipeline for the tomogram slab segmentation model.
    This class handles data loading, model instantiation, and the PyTorch Lightning training process.
    """

    def __init__(self,
                 model_arch: str = config.MODEL_CONFIG['arch'],
                 model_encoder: str = config.MODEL_CONFIG['encoder_name'],
                 encoder_weights: str = config.MODEL_CONFIG['encoder_weights'],
                 encoder_depth: int = config.MODEL_CONFIG['encoder_depth'],
                 decoder_channels: List = config.MODEL_CONFIG['decoder_channels'],
                 decoder_attention_type: str = config.MODEL_CONFIG['decoder_attention_type'],
                 classes: int = config.MODEL_CONFIG['classes'],
                 in_channels: int = config.MODEL_CONFIG['in_channels'],
                 loss_config: dict = config.LOSS_CONFIG,
                 learning_rate: float = config.LEARNING_RATE):
        """
        Initializes the trainer with model and loss configurations.
        Args:
            model_arch: The architecture of the segmentation model (e.g., 'Unet').
            model_encoder: The encoder backbone for the model (e.g., 'resnet18').
            loss_config: A dictionary defining the loss function and its parameters.
            learning_rate: The initial learning rate for the optimizer.
        """
        self.model_arch = model_arch
        self.model_encoder = model_encoder
        self.encoder_weights = encoder_weights
        self.encoder_depth = encoder_depth
        self.decoder_channels = decoder_channels
        self.decoder_attention_type = decoder_attention_type
        self.classes = classes
        self.in_channels = in_channels
        self.loss_config = loss_config
        self.learning_rate = learning_rate
        self.global_rank = int(os.environ.get("GLOBAL_RANK", 0))

    def _setup_datamodule(self) -> SegmentationDataModule:
        train_files = sorted(list(constants.TRAIN_DATA_DIR.glob("*.pt")))
        val_files = sorted(list(constants.VAL_DATA_DIR.glob("*.pt")))
        if not train_files or not val_files:
            raise FileNotFoundError("Training or validation data not found. Run the TrainingDataGenerator first.")
        return SegmentationDataModule(
            train_pt_files=train_files,
            val_pt_files=val_files,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )

    def _setup_model(self) -> SegmentationModel:
        base_model = smp.create_model(
            arch=self.model_arch,
            encoder_name=self.model_encoder,
            encoder_weights=self.encoder_weights,
            encoder_depth=self.encoder_depth,
            decoder_channels=self.decoder_channels,
            decoder_attention_type=self.decoder_attention_type,
            classes=self.classes, in_channels=self.in_channels,
            activation=None
        )
        loss_fn = get_loss_function(self.loss_config)
        if self.global_rank == 0:
            loss_name = getattr(loss_fn, 'name', loss_fn.__class__.__name__)
            print(f"Using Model: {self.model_arch}-{self.model_encoder} (ImageNet pre-trained)")
            print(f"Using Loss Function: {loss_name}")
        return SegmentationModel(
            model=base_model,
            loss_function=loss_fn,
            learning_rate=self.learning_rate,
            target_shape=constants.TARGET_VOLUME_SHAPE,
        )

    def _setup_callbacks(self) -> list:
        checkpointer = ModelCheckpoint(
            monitor=config.MONITOR_METRIC, mode="max",
            filename=f"best-{{epoch}}-{{{config.MONITOR_METRIC}:.4f}}",
            save_top_k=config.CHECKPOINT_SAVE_TOP_K, verbose=(self.global_rank == 0),
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        progress_bar = TQDMProgressBar(refresh_rate=10)
        callbacks = [progress_bar, checkpointer, lr_monitor]

        if config.USE_DYNAMIC_MANAGER:
            if self.global_rank == 0: print("Using DynamicTrainingManager for adaptive SWA and Early Stopping.")
            callbacks.append(DynamicTrainingManager(
                monitor=config.MONITOR_METRIC, mode="max", ema_alpha=config.EMA_ALPHA,
                trigger_swa_patience=config.SWA_TRIGGER_PATIENCE,
                early_stop_patience=config.EARLY_STOP_PATIENCE, min_delta=config.EARLY_STOP_MIN_DELTA
            ))
        else:
            if self.global_rank == 0: print("Using standard EarlyStopping callback.")
            callbacks.append(EarlyStopping(
                monitor=config.MONITOR_METRIC, mode="max", patience=config.STANDARD_EARLY_STOPPING_PATIENCE,
                min_delta=config.EARLY_STOP_MIN_DELTA, verbose=(self.global_rank == 0),
            ))
        if config.USE_SWA:
            if self.global_rank == 0: print("Stochastic Weight Averaging (SWA) is enabled.")
            swa_start = config.MAX_EPOCHS + 1 if config.USE_DYNAMIC_MANAGER else config.STANDARD_SWA_START_FRACTION
            callbacks.append(StochasticWeightAveraging(swa_lrs=config.SWA_LEARNING_RATE, swa_epoch_start=swa_start))
        return callbacks

    def fit(self):
        """
        Sets up all components and starts the training process.
        """
        if self.global_rank == 0:
            print("--- Running on main process (rank 0). Verbose output enabled. ---")
        torch.set_float32_matmul_precision('high')

        datamodule = self._setup_datamodule()
        pl_model = self._setup_model()

        if self.global_rank == 0: print("\n--- Configuring Logger and Callbacks ---")
        experiment_name = f"{self.model_arch}-{self.model_encoder}"
        experiment_details = f"loss-{self.loss_config['name'].replace('+', '_')}"
        logger = TensorBoardLogger(
            save_dir="lightning_logs",
            name=f"{experiment_name}/{experiment_details}"
        )

        callbacks = self._setup_callbacks()

        trainer = pl.Trainer(
            max_epochs=config.MAX_EPOCHS, accelerator=config.ACCELERATOR, devices=config.DEVICES,
            precision=config.PRECISION, log_every_n_steps=config.LOG_EVERY_N_STEPS,
            check_val_every_n_epoch=config.CHECK_VAL_EVERY_N_EPOCH,
            logger=logger,
            callbacks=callbacks,
            strategy='ddp_find_unused_parameters_true'
        )

        if self.global_rank == 0:
            print("\n--- Starting Training ---")
            print(f"To view logs, run: tensorboard --logdir={logger.save_dir}")

        trainer.fit(pl_model, datamodule=datamodule)

        if trainer.is_global_zero:
            print("--- Training Finished ---")
            if hasattr(trainer.checkpoint_callback, 'best_model_path'):
                print(f"Best model checkpoint saved at: {trainer.checkpoint_callback.best_model_path}")