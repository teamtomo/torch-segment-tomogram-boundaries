import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging, TQDMProgressBar

sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch_tomo_slab import config, constants
from torch_tomo_slab.pl_model import SegmentationModel
from torch_tomo_slab.data.dataloading import SegmentationDataModule
from torch_tomo_slab.losses import BoundaryLoss, WeightedBCELoss, CombinedLoss, SMPLossWrapper
from torch_tomo_slab.callbacks import DynamicTrainingManager

def get_loss_function(loss_config: dict):
    name = loss_config['name'].lower()
    params = loss_config.get('params', {})
    from_logits = True # All our model outputs are logits

    loss_lib = {
        'bce': nn.BCEWithLogitsLoss(),
        'weighted_bce': WeightedBCELoss(from_logits=from_logits),
        'boundary': BoundaryLoss(from_logits=from_logits),

        # SMP losses wrapped to be compatible
        'dice': SMPLossWrapper(smp.losses.DiceLoss(mode='binary'), from_logits=from_logits),
        'focal': SMPLossWrapper(smp.losses.FocalLoss(mode='binary', **params.get('focal', {})), from_logits=from_logits),
        'lovasz': SMPLossWrapper(smp.losses.LovaszLoss(mode='binary'), from_logits=from_logits),
        'tversky': SMPLossWrapper(smp.losses.TverskyLoss(mode='binary', **params.get('tversky', {})), from_logits=from_logits),
    }

    if '+' not in name:
        if name in loss_lib:
            return loss_lib[name]
        raise ValueError(f"Unknown loss function: {name}")

    # Handle combined losses
    loss_components_names = name.split('+')
    loss_weights = loss_config.get('weights')
    if not loss_weights or len(loss_components_names) != len(loss_weights):
        raise ValueError(f"Loss weights must be provided for combined loss '{name}' and match the number of components.")

    losses_to_combine = {}
    for comp_name, weight in zip(loss_components_names, loss_weights):
        if comp_name in loss_lib:
            losses_to_combine[comp_name] = (loss_lib[comp_name], weight)
        else:
            raise ValueError(f"Unknown component '{comp_name}' in combined loss.")
            
    return CombinedLoss(losses_to_combine, from_logits=from_logits)

def run_training():
    global_rank = int(os.environ.get("GLOBAL_RANK", 0))
    if global_rank == 0:
        print("--- Running on main process (rank 0). Verbose output enabled. ---")

    torch.set_float32_matmul_precision('high')

    train_files = sorted(list(constants.TRAIN_DATA_DIR.glob("*.pt")))
    val_files = sorted(list(constants.VAL_DATA_DIR.glob("*.pt")))
    if not train_files or not val_files:
        raise FileNotFoundError("Training or validation data not found. Did you run the p02 script?")

    datamodule = SegmentationDataModule(
        train_pt_files=train_files,
        val_pt_files=val_files,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    model = smp.create_model(
        arch=constants.MODEL_ARCH,
        encoder_name=constants.MODEL_ENCODER,
        encoder_weights="imagenet",
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_attention_type='scse',
        classes=1, in_channels=2,
        activation=None
    )

    loss_fn = get_loss_function(config.LOSS_CONFIG)

    if global_rank == 0:
        loss_name = getattr(loss_fn, 'name', loss_fn.__class__.__name__)
        print(f"Using Model: {constants.MODEL_ARCH}-{constants.MODEL_ENCODER} (ImageNet pre-trained)")
        print(f"Using Loss Function: {loss_name}")

    pl_model = SegmentationModel(
        model=model,
        loss_function=loss_fn,
        learning_rate=config.LEARNING_RATE,
    )

    if global_rank == 0:
        print("\n--- Configuring Logger and Callbacks ---")
    experiment_name = f"{constants.MODEL_ARCH}-{constants.MODEL_ENCODER}"
    experiment_details = f"loss-{config.LOSS_CONFIG['name'].replace('+', '_')}"
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{experiment_name}/{experiment_details}"
    )
    
    checkpointer = ModelCheckpoint(
        monitor=config.MONITOR_METRIC, mode="max",
        filename=f"best-{{epoch}}-{{{config.MONITOR_METRIC}:.4f}}",
        save_top_k=config.CHECKPOINT_SAVE_TOP_K, verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks = [progress_bar, checkpointer, lr_monitor]

    if config.USE_DYNAMIC_MANAGER:
        if global_rank == 0:
            print("Using DynamicTrainingManager for adaptive SWA and Early Stopping.")
        dynamic_manager = DynamicTrainingManager(
            monitor=config.MONITOR_METRIC,
            mode="max",
            ema_alpha=config.EMA_ALPHA,
            trigger_swa_patience=config.SWA_TRIGGER_PATIENCE,
            early_stop_patience=config.EARLY_STOP_PATIENCE,
            min_delta=config.EARLY_STOP_MIN_DELTA
        )
        callbacks.append(dynamic_manager)
    else:
        if global_rank == 0:
            print("Using standard EarlyStopping callback.")
        early_stopper = EarlyStopping(
            monitor=config.MONITOR_METRIC, mode="max", patience=config.STANDARD_EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOP_MIN_DELTA, verbose=True,
        )
        callbacks.append(early_stopper)
        
    if config.USE_SWA:
        if global_rank == 0:
            print(f"Stochastic Weight Averaging (SWA) is enabled.")
        # If using dynamic manager, SWA starts when triggered. Otherwise, it uses a fixed fraction.
        swa_start = config.MAX_EPOCHS + 1 if config.USE_DYNAMIC_MANAGER else config.STANDARD_SWA_START_FRACTION
        swa = StochasticWeightAveraging(swa_lrs=config.SWA_LEARNING_RATE, swa_epoch_start=swa_start)
        callbacks.append(swa)
        
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS, accelerator=config.ACCELERATOR, devices=config.DEVICES,
        precision=config.PRECISION, log_every_n_steps=config.LOG_EVERY_N_STEPS,
        check_val_every_n_epoch=config.CHECK_VAL_EVERY_N_EPOCH,
        logger=logger,
        callbacks=callbacks,
        strategy='ddp_find_unused_parameters_true'
    )
    
    if global_rank == 0:
        print("\n--- Starting Training ---")
        print(f"To view logs, run: tensorboard --logdir={logger.save_dir}")
        
    trainer.fit(pl_model, datamodule=datamodule)
    
    if trainer.is_global_zero:
        print("--- Training Finished ---")
        print(f"Best model checkpoint saved at: {checkpointer.best_model_path}")

if __name__ == "__main__":
    run_training()
