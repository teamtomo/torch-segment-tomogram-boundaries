#!/usr/bin/env python3
"""
Optuna hyperparameter optimization wrapper for torch-tomo-slab network.

This script uses Optuna to optimize key hyperparameters:
- Learning rate
- Batch size  
- Decoder channel sizes

Usage:
    python optimize_hyperparameters.py --n-trials 50 --study-name tomo-optimization

The script will create/load an Optuna study and run hyperparameter optimization trials.
Each trial trains a model with different hyperparameters and reports the validation dice score.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import logging

import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Import torch-tomo-slab modules
from torch_tomo_slab import config
from torch_tomo_slab.processing import TrainingDataGenerator
from torch_tomo_slab.trainer import TomoSlabTrainer


def setup_logging():
    """Setup logging for Optuna optimization."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('optuna_optimization.log')
        ]
    )
    
    # Reduce pytorch lightning verbosity
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.core").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.utilities").setLevel(logging.WARNING)


def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Define hyperparameter search space for Optuna.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of suggested hyperparameters
    """
    # Learning rate optimization (log scale)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 4.5e-5, log=True)
    
    # Batch size optimization (powers of 2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    
    # Decoder channel configuration optimization
    # Try different channel sizes while maintaining decreasing pattern
    decoder_base = trial.suggest_categorical('decoder_base', [64, 128, 256, 512])
    
    # Generate decoder channels with fixed ratios, ensuring unique decreasing values
    if decoder_base >= 256:
        # For large bases: [512, 256, 128, 64, 32] or [256, 128, 64, 32, 16]
        decoder_channels = [
            decoder_base,
            decoder_base // 2,
            decoder_base // 4,
            decoder_base // 8,
            decoder_base // 16
        ]
    elif decoder_base >= 128:
        # For medium bases: [128, 64, 32, 16] (4 channels)
        decoder_channels = [
            decoder_base,
            decoder_base // 2,
            decoder_base // 4,
            decoder_base // 8
        ]
    else:
        # For small bases: [64, 32, 16] (3 channels)  
        decoder_channels = [
            decoder_base,
            decoder_base // 2,
            decoder_base // 4
        ]
    
    # Weight decay optimization
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    # Dropout rate optimization
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    return {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'decoder_channels': decoder_channels,
        'weight_decay': weight_decay,
        'dropout': dropout
    }


def create_config_with_hyperparameters(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Create a configuration dictionary with suggested hyperparameters.
    
    Parameters
    ----------
    hyperparams : Dict[str, Any]
        Dictionary of hyperparameters from Optuna trial
        
    Returns
    -------
    Dict[str, Any]
        Updated configuration dictionary
    """
    # Create a copy of the base model config
    model_config = config.MODEL_CONFIG.copy()
    model_config['decoder_channels'] = hyperparams['decoder_channels']
    model_config['dropout'] = hyperparams['dropout']
    
    # Create optimizer config with suggested learning rate and weight decay
    optimizer_config = config.OPTIMIZER_CONFIG.copy()
    optimizer_config['params'] = optimizer_config['params'].copy()
    optimizer_config['params']['lr'] = hyperparams['learning_rate']
    optimizer_config['params']['weight_decay'] = hyperparams['weight_decay']
    
    return {
        'model_config': model_config,
        'optimizer_config': optimizer_config,
        'batch_size': hyperparams['batch_size'],
        'learning_rate': hyperparams['learning_rate']
    }


def objective(trial: optuna.Trial, 
              tomo_dir: str, 
              mask_vol_dir: str, 
              base_output_dir: str,
              shared_train_dir: str = None,
              shared_val_dir: str = None) -> float:
    """Objective function for Optuna optimization.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    tomo_dir : str
        Directory containing tomogram volumes
    mask_vol_dir : str
        Directory containing mask volumes  
    base_output_dir : str
        Base directory for outputs
        
    Returns
    -------
    float
        Validation dice score to maximize
    """
    # Get hyperparameters for this trial
    hyperparams = suggest_hyperparameters(trial)
    
    logging.info(f"Trial {trial.number} hyperparameters: {hyperparams}")
    
    # Create trial-specific output directories
    trial_dir = Path(base_output_dir) / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    output_train_dir = trial_dir / "train"
    output_val_dir = trial_dir / "val"
    ckpt_save_dir = trial_dir / "checkpoints"
    
    # Set environment variable to force single GPU training during Optuna
    import os
    os.environ['OPTUNA_TRIAL'] = '1'
    
    # Initialize variables that might be used in exception handler
    original_config = {}
    original_batch_size = config.BATCH_SIZE
    original_lr = config.LEARNING_RATE 
    original_max_epochs = config.MAX_EPOCHS
    
    try:
        # Use shared dataset if provided, otherwise create trial-specific
        if shared_train_dir and shared_val_dir:
            train_data_dir = shared_train_dir
            val_data_dir = shared_val_dir
            logging.info(f"Trial {trial.number}: Using shared training data")
        else:
            # Fallback to old behavior for backward compatibility
            train_exists = output_train_dir.exists() and len(list(output_train_dir.glob('*'))) > 0
            val_exists = output_val_dir.exists() and len(list(output_val_dir.glob('*'))) > 0
            
            if not (train_exists and val_exists):
                logging.info(f"Trial {trial.number}: Preparing training data...")
                
                generator = TrainingDataGenerator(
                    volume_dir=tomo_dir,
                    mask_dir=mask_vol_dir,
                    output_train_dir=output_train_dir,
                    output_val_dir=output_val_dir
                )
                generator.run()
                logging.info(f"Trial {trial.number}: Data preparation complete")
            else:
                logging.info(f"Trial {trial.number}: Using existing training data")
            
            train_data_dir = str(output_train_dir)
            val_data_dir = str(output_val_dir)
        
        # Create configuration with suggested hyperparameters
        trial_config = create_config_with_hyperparameters(hyperparams)
        
        # Temporarily modify the config module
        for key, value in trial_config.items():
            if hasattr(config, key.upper()):
                original_config[key.upper()] = getattr(config, key.upper())
                setattr(config, key.upper(), value)
        
        # Set batch size
        config.BATCH_SIZE = hyperparams['batch_size']
        
        # Set learning rate 
        config.LEARNING_RATE = hyperparams['learning_rate']
        
        # Reduce max epochs for optimization speed
        config.MAX_EPOCHS = min(30, config.MAX_EPOCHS)  # Cap at 30 epochs for optimization
        
        logging.info(f"Trial {trial.number}: Starting training with {config.MAX_EPOCHS} epochs...")
        
        # Create trainer with trial-specific configuration
        trainer = TomoSlabTrainer(
            train_data_dir=train_data_dir,
            val_data_dir=val_data_dir,
            ckpt_save_dir=str(ckpt_save_dir)
        )
        
        # Add Optuna pruning callback
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_dice")
        if hasattr(trainer, 'callbacks'):
            trainer.callbacks.append(pruning_callback)
        
        # Train the model
        result = trainer.fit()
        
        # Get the best validation dice score
        if hasattr(trainer.trainer, 'callback_metrics'):
            val_dice = float(trainer.trainer.callback_metrics.get('val_dice', 0.0))
        else:
            # Fallback: try to get from checkpoints
            val_dice = 0.0
            if ckpt_save_dir.exists():
                checkpoints = list(ckpt_save_dir.glob('*.ckpt'))
                if checkpoints:
                    # Parse dice score from checkpoint filename if available
                    for ckpt in checkpoints:
                        if 'dice' in ckpt.name:
                            try:
                                # Extract dice score from filename pattern like "dice=0.85.ckpt"
                                dice_str = ckpt.name.split('dice=')[1].split('.ckpt')[0]
                                val_dice = max(val_dice, float(dice_str))
                            except (IndexError, ValueError):
                                pass
        
        logging.info(f"Trial {trial.number}: Completed with val_dice = {val_dice}")
        
        # Restore original configuration
        for key, value in original_config.items():
            setattr(config, key, value)
        config.BATCH_SIZE = original_batch_size
        config.LEARNING_RATE = original_lr
        config.MAX_EPOCHS = original_max_epochs
        
        return val_dice
        
    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {str(e)}")
        
        # Restore original configuration in case of error
        for key, value in original_config.items():
            setattr(config, key, value)
        config.BATCH_SIZE = original_batch_size
        config.LEARNING_RATE = original_lr 
        config.MAX_EPOCHS = original_max_epochs
        
        # Optuna handles exceptions by marking trial as failed
        raise optuna.TrialPruned()


def main():
    """Main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for torch-tomo-slab')
    
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of optimization trials to run')
    parser.add_argument('--study-name', type=str, default='tomo-slab-optimization',
                        help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (for distributed optimization)')
    parser.add_argument('--tomo-dir', type=str, 
                        default="/home/pranav/data/training/torch-tomo-slab/data_in/volumes",
                        help='Directory containing tomogram volumes')
    parser.add_argument('--mask-dir', type=str,
                        default="/home/pranav/data/training/torch-tomo-slab/data_in/boundary_mask_volumes",
                        help='Directory containing mask volumes')
    parser.add_argument('--output-dir', type=str,
                        default="/home/pranav/data/training/torch-tomo-slab/optuna_trials",
                        help='Base directory for optimization outputs')
    parser.add_argument('--direction', type=str, default='maximize',
                        choices=['maximize', 'minimize'],
                        help='Optimization direction for the objective')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting hyperparameter optimization with {args.n_trials} trials")
    logging.info(f"Study name: {args.study_name}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Create or load Optuna study
    if args.storage:
        study = optuna.create_study(
            direction=args.direction,
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True
        )
    else:
        # Use SQLite database for local storage
        storage_path = output_dir / f"{args.study_name}.db"
        storage_url = f"sqlite:///{storage_path}"
        
        study = optuna.create_study(
            direction=args.direction,
            study_name=args.study_name,
            storage=storage_url,
            load_if_exists=True
        )
    
    # Create shared training dataset once before all trials
    shared_data_dir = output_dir / "shared_data"
    shared_train_dir = shared_data_dir / "train"
    shared_val_dir = shared_data_dir / "val"
    
    # Check if shared data already exists
    train_exists = shared_train_dir.exists() and len(list(shared_train_dir.glob('*'))) > 0
    val_exists = shared_val_dir.exists() and len(list(shared_val_dir.glob('*'))) > 0
    
    if not (train_exists and val_exists):
        logging.info("Creating shared training dataset for all trials...")
        shared_data_dir.mkdir(parents=True, exist_ok=True)
        
        generator = TrainingDataGenerator(
            volume_dir=args.tomo_dir,
            mask_dir=args.mask_dir,
            output_train_dir=shared_train_dir,
            output_val_dir=shared_val_dir
        )
        generator.run()
        logging.info("Shared dataset creation complete")
    else:
        logging.info("Using existing shared training dataset")
    
    # Define objective function with fixed arguments and shared data
    def objective_with_args(trial):
        return objective(trial, args.tomo_dir, args.mask_dir, args.output_dir,
                        str(shared_train_dir), str(shared_val_dir))
    
    # Run optimization with sequential execution (n_jobs=1 forces sequential)
    try:
        study.optimize(objective_with_args, n_trials=args.n_trials, n_jobs=1)
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user")
    
    # Print results
    logging.info("Optimization completed!")
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value (val_dice): {study.best_value}")
    logging.info("Best parameters:")
    for key, value in study.best_params.items():
        logging.info(f"  {key}: {value}")
    
    # Save results
    results_file = output_dir / f"{args.study_name}_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Hyperparameter Optimization Results\n")
        f.write(f"Study: {args.study_name}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best validation dice: {study.best_value}\n\n")
        f.write(f"Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nAll trials:\n")
        for trial in study.trials:
            if trial.value is not None:
                f.write(f"Trial {trial.number}: {trial.value:.4f} - {trial.params}\n")
    
    logging.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()