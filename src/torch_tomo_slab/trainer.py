"""PyTorch Lightning training pipeline for tomographic boundary segmentation.

This module provides the main training interface that orchestrates data loading,
model setup, callback configuration, and training execution using PyTorch Lightning.
"""
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from torch_tomo_slab import config, constants
from torch_tomo_slab.callbacks import DynamicTrainingManager
from torch_tomo_slab.data.dataloading import SegmentationDataModule
from torch_tomo_slab.losses import get_loss_function
from torch_tomo_slab.pl_model import SegmentationModel


class TomoSlabTrainer:
    """
    Complete training pipeline for tomographic boundary segmentation models.
    
    This class provides a high-level interface for training deep learning models
    on tomographic data. It encapsulates the entire training workflow including
    data preparation, model configuration, callback setup, and training execution
    using PyTorch Lightning.
    
    The trainer supports:
    - Configurable model architectures from segmentation-models-pytorch
    - Multiple loss functions with automatic selection
    - Advanced training features (SWA, dynamic management, early stopping)
    - Multi-GPU training and mixed precision
    - Comprehensive logging and checkpointing
    
    Attributes
    ----------
    model_arch : str
        Architecture name (e.g., 'Unet', 'UnetPlusPlus').
    model_encoder : str  
        Encoder backbone name (e.g., 'resnet18', 'efficientnet-b0').
    encoder_weights : str or None
        Pre-trained weights ('imagenet', None).
    encoder_depth : int
        Encoder depth (typically 5 for ResNet).
    decoder_channels : List[int]
        Decoder channel configuration.
    decoder_attention_type : str
        Attention mechanism type ('scse', 'cbam', None).
    classes : int
        Number of output classes (1 for binary segmentation).
    in_channels : int
        Input channels (2 for normalized + variance).
    loss_config : Dict[str, Any]
        Loss function configuration.
    learning_rate : float
        Initial learning rate.
    global_rank : int
        Process rank for distributed training.
    """

    def __init__(self,
                 model_arch: str = config.MODEL_CONFIG['arch'],
                 model_encoder: str = config.MODEL_CONFIG['encoder_name'],
                 encoder_weights: Optional[str] = config.MODEL_CONFIG['encoder_weights'],
                 encoder_depth: int = config.MODEL_CONFIG['encoder_depth'],
                 decoder_channels: List[int] = config.MODEL_CONFIG['decoder_channels'],
                 decoder_attention_type: str = config.MODEL_CONFIG['decoder_attention_type'],
                 classes: int = config.MODEL_CONFIG['classes'],
                 in_channels: int = config.MODEL_CONFIG['in_channels'],
                 loss_config: Dict[str, Any] = config.LOSS_CONFIG,
                 learning_rate: float = config.LEARNING_RATE,
                 train_data_dir: Path = config.TRAIN_DATA_DIR,
                 val_data_dir: Path = config.VAL_DATA_DIR,
                 ckpt_save_dir: Path = config.CKPT_SAVE_PATH) -> None:
        """
        Initialize the TomoSlabTrainer with model and training configurations.

        Parameters
        ----------
        model_arch : str
            Architecture name for the segmentation model (e.g., 'Unet', 'UnetPlusPlus').
        model_encoder : str
            Encoder backbone name (e.g., 'resnet18', 'resnet34', 'efficientnet-b0').
        encoder_weights : str or None
            Pre-trained weights for encoder ('imagenet', None).
        encoder_depth : int
            Depth of the encoder (typically 5 for ResNet architectures).
        decoder_channels : List[int]
            Number of channels in each decoder block, from deepest to shallowest.
        decoder_attention_type : str
            Type of attention mechanism in decoder ('scse', 'cbam', None).
        classes : int
            Number of output classes (1 for binary segmentation).
        in_channels : int
            Number of input channels (2 for normalized + variance images).
        loss_config : Dict[str, Any]
            Loss function configuration with 'name' and optional 'weights'.
        learning_rate : float
            Initial learning rate for the optimizer.
        train_data_dir : Path
            Directory containing training data files.
        val_data_dir : Path
            Directory containing validation data files.
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
        self.train_data_dir = Path(train_data_dir)
        self.val_data_dir = Path(val_data_dir)
        self.ckpt_save_dir = Path(ckpt_save_dir)
        self.global_rank = int(os.environ.get("GLOBAL_RANK", 0))

    def _setup_datamodule(self) -> SegmentationDataModule:
        """
        Set up the PyTorch Lightning DataModule for training and validation.

        Returns
        -------
        SegmentationDataModule
            Configured data module with train/val datasets.

        Raises
        ------
        FileNotFoundError
            If training or validation data files are not found.
        """
        train_files = sorted(list(self.train_data_dir.glob("*.pt")))
        val_files = sorted(list(self.val_data_dir.glob("*.pt")))
        if not train_files or not val_files:
            raise FileNotFoundError("Training or validation data not found. Run the TrainingDataGenerator first.")
        return SegmentationDataModule(
            train_pt_files=train_files,
            val_pt_files=val_files,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )

    def _setup_model(self) -> SegmentationModel:
        """
        Create and configure the PyTorch Lightning model wrapper.

        Returns
        -------
        SegmentationModel
            Configured Lightning module with model, loss function, and hyperparameters.
        """
        base_model = smp.create_model(
            arch=self.model_arch,
            encoder_name=self.model_encoder,
            encoder_weights=self.encoder_weights,
            encoder_depth=self.encoder_depth,
            decoder_channels=self.decoder_channels,
            decoder_attention_type=self.decoder_attention_type,
            classes=self.classes, in_channels=self.in_channels,
            activation=None,
            aux_params={'classes':1,
                        'dropout':config.MODEL_CONFIG['dropout']}
        )
        loss_fn = get_loss_function(self.loss_config)
        if self.global_rank == 0:
            loss_name = getattr(loss_fn, 'name', loss_fn.__class__.__name__)
            print(f"Using Model: {self.model_arch}-{self.model_encoder} (ImageNet pre-trained)")
            print(f"Using Loss Function: {loss_name}")
            print("Using enhanced augmentations without cropping for consistent train/val pipeline")
        return SegmentationModel(
            model=base_model,
            loss_function=loss_fn, learning_rate=self.learning_rate,
            target_shape=constants.TARGET_VOLUME_SHAPE,
        )

    def _setup_callbacks(self) -> List[pl.Callback]:
        """
        Configure PyTorch Lightning callbacks for training.

        Returns
        -------
        List[pl.Callback]
            List of configured callbacks including checkpointing, early stopping,
            learning rate monitoring, and optional SWA/dynamic training management.
        """
        checkpointer = ModelCheckpoint(
            monitor=constants.MONITOR_METRIC, mode="max",
            filename=f"best-{{epoch}}-{{{constants.MONITOR_METRIC}:.4f}}",
            save_top_k=config.CHECKPOINT_SAVE_TOP_K, verbose=(self.global_rank == 0),
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        progress_bar = TQDMProgressBar(refresh_rate=10)
        callbacks = [progress_bar, checkpointer, lr_monitor]

        if config.USE_DYNAMIC_MANAGER:
            if self.global_rank == 0: print("Using DynamicTrainingManager for adaptive SWA and Early Stopping.")
            callbacks.append(DynamicTrainingManager(
                monitor=constants.MONITOR_METRIC, mode="max", ema_alpha=config.EMA_ALPHA,
                trigger_swa_patience=config.SWA_TRIGGER_PATIENCE,
                early_stop_patience=config.EARLY_STOP_PATIENCE, min_delta=config.EARLY_STOP_MIN_DELTA
            ))
        else:
            if self.global_rank == 0: print("Using standard EarlyStopping callback.")
            callbacks.append(EarlyStopping(
                monitor=constants.MONITOR_METRIC, mode="max", patience=config.STANDARD_EARLY_STOPPING_PATIENCE,
                min_delta=config.EARLY_STOP_MIN_DELTA, verbose=(self.global_rank == 0),
            ))
        if constants.USE_SWA:
            if self.global_rank == 0: print("Stochastic Weight Averaging (SWA) is enabled.")
            swa_start = config.MAX_EPOCHS + 1 if config.USE_DYNAMIC_MANAGER else config.STANDARD_SWA_START_FRACTION
            callbacks.append(StochasticWeightAveraging(swa_lrs=config.SWA_LEARNING_RATE, swa_epoch_start=swa_start))
        return callbacks

    def fit(self, extra_callbacks: Optional[List[pl.Callback]] = None, extra_trainer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Execute the complete training pipeline.

        This method orchestrates the entire training process including:
        - Data module setup and validation
        - Model instantiation with loss function
        - Callback and logger configuration
        - PyTorch Lightning trainer setup
        - Model training execution

        The trained model checkpoints will be saved to 'lightning_logs/' directory.
        """
        if self.global_rank == 0:
            print("--- Running on main process (rank 0). Verbose output enabled. ---")
        torch.set_float32_matmul_precision('high')

        datamodule = self._setup_datamodule()
        pl_model = self._setup_model()

        if self.global_rank == 0: print("--- Configuring Logger and Callbacks ---")
        experiment_name = f"{self.model_arch}-{self.model_encoder}"
        experiment_details = f"loss-{self.loss_config['name'].replace('+', '_')}"
        logger = TensorBoardLogger(
            save_dir=self.ckpt_save_dir,
            name=f"{experiment_name}--{experiment_details}"
        )

        callbacks = self._setup_callbacks()
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        trainer_config = {
            "max_epochs": config.MAX_EPOCHS,
            "accelerator": config.ACCELERATOR,
            "devices": config.DEVICES,
            "precision": config.PRECISION,
            "log_every_n_steps": constants.LOG_EVERY_N_STEPS,
            "check_val_every_n_epoch": constants.CHECK_VAL_EVERY_N_EPOCH,
            "logger": logger,
            "callbacks": callbacks,
            "strategy": "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto",
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "enable_model_summary": True,
        }
        if extra_trainer_kwargs:
            trainer_config.update(extra_trainer_kwargs)

        self.trainer = pl.Trainer(**trainer_config)

        if self.global_rank == 0:
            print("--- Starting Training ---")
            print(f"To view logs, run: tensorboard --logdir={logger.save_dir}")

        self.trainer.fit(pl_model, datamodule=datamodule)

        if self.trainer.is_global_zero:
            print("--- Training Finished ---")
            if hasattr(self.trainer.checkpoint_callback, 'best_model_path') and self.trainer.checkpoint_callback is not None:
                print(f"Best model checkpoint saved at: {self.trainer.checkpoint_callback.best_model_path}")


def train(
    train_data_dir: Path,
    val_data_dir: Path,
    ckpt_save_dir: Path = config.CKPT_SAVE_PATH,
    learning_rate: float = config.LEARNING_RATE,
    max_epochs: int = config.MAX_EPOCHS,
    **trainer_kwargs,
):
    """
    High-level function to train a tomogram segmentation model.

    This function provides a simple interface for training, handling the setup
    of the model, data, and trainer based on the provided parameters and
    project configurations.

    Parameters
    ----------
    train_data_dir : Path
        Directory containing training data files (*.pt).
    val_data_dir : Path
        Directory containing validation data files (*.pt).
    ckpt_save_dir : Path, optional
        Directory to save model checkpoints and logs, by default config.CKPT_SAVE_PATH.
    learning_rate : float, optional
        Initial learning rate for the optimizer, by default config.LEARNING_RATE.
    max_epochs : int, optional
        Maximum number of training epochs, by default config.MAX_EPOCHS.
    **trainer_kwargs :
        Additional keyword arguments to be passed to the PyTorch Lightning Trainer.
        This allows for overriding settings like 'accelerator', 'devices', etc.
    """
    trainer = TomoSlabTrainer(
        learning_rate=learning_rate,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        ckpt_save_dir=ckpt_save_dir,
    )

    # Combine max_epochs with other trainer arguments
    all_trainer_kwargs = {"max_epochs": max_epochs, **trainer_kwargs}
    trainer.fit(extra_trainer_kwargs=all_trainer_kwargs)