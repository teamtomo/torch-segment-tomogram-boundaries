"""
Hyperparameter tuning script for torch-tomo-slab using Optuna.

This script performs hyperparameter optimization for the segmentation model.
It tunes the learning rate, base channel width, network dropout, and batch size.
The script is designed to be run on a machine with at least one GPU.

Prerequisites:
    - Install the project in editable mode with the 'optuna' extras:
      `pip install -e \".[optuna]\"` or `uv pip install -e . --extra optuna`
    - Ensure the training data has been generated and is located in the
      directory specified by `config.TRAIN_DATA_DIR` and `config.VAL_DATA_DIR`.

Usage (Single GPU):
    python tune_hyperparameters.py

Usage (Parallel on Multi-GPU):
    1. Open two terminals.
    2. In terminal 1, run: `python tune_hyperparameters.py --gpu 0`
    3. In terminal 2, run: `python tune_hyperparameters.py --gpu 1`
    
    This will run two optimization processes in parallel, with each process
    using a different GPU. They will coordinate through the SQLite database.
"""
import os
import sys
import argparse
import traceback
from copy import deepcopy
from pathlib import Path

import optuna
import torch

# Add src to path to allow imports
sys.path.append(str(Path(__file__).parent / "src"))

from torch_tomo_slab import config
from torch_tomo_slab.trainer import TomoSlabTrainer


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function."""

    # Hyperparameter suggestions scoped to the MONAI setup
    learning_rate = trial.suggest_float("learning_rate", 2e-5, 8e-5, log=True)
    dropout = trial.suggest_float("dropout", 0.08, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [4, 6, 8])
    base_width = trial.suggest_categorical("base_width", [32, 40, 48])

    # Snapshot originals so we can restore them after the trial
    original_model_config = deepcopy(config.MODEL_CONFIG)
    original_batch_size = config.BATCH_SIZE
    original_learning_rate = config.LEARNING_RATE
    original_devices = config.DEVICES

    # Update configuration for this trial
    config.MODEL_CONFIG['dropout'] = dropout
    depth = len(config.MODEL_CONFIG.get('channels', ())) or len(original_model_config.get('channels', ()))
    if depth == 0:
        depth = 4
    config.MODEL_CONFIG['channels'] = tuple(base_width * (2 ** i) for i in range(depth))
    config.BATCH_SIZE = batch_size
    config.LEARNING_RATE = learning_rate

    # Ensure single-device training inside trials
    config.DEVICES = 1

    metric = 0.0
    try:
        print("-"*80)
        print(f"Starting Trial {trial.number}")
        print(f"  - BASE_DATA_PATH: {config.BASE_DATA_PATH}")
        print(f"  - CKPT_SAVE_PATH: {config.CKPT_SAVE_PATH}")
        print(f"  - Params: lr={learning_rate:.2e}, dropout={dropout:.2f}, batch_size={batch_size}, base_width={base_width}")
        print(f"  - Channels: {config.MODEL_CONFIG['channels']}")
        
        # Each trial gets its own checkpoint directory
        ckpt_dir = config.CKPT_SAVE_PATH / f"trial_{trial.number}"
        print(f"  - This trial's ckpt_dir: {ckpt_dir}")
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        trainer = TomoSlabTrainer(
            learning_rate=learning_rate,
            loss_config=config.LOSS_CONFIG,
            train_data_dir=config.TRAIN_DATA_DIR,
            val_data_dir=config.VAL_DATA_DIR,
            ckpt_save_dir=ckpt_dir
        )

        trainer.fit(extra_callbacks=None)

        best_score = trainer.trainer.checkpoint_callback.best_model_score

        if best_score:
            metric = best_score.item()
        else:
            print(f"Trial {trial.number}: Could not find best_model_score, returning 0.0")
            metric = 0.0

    except optuna.exceptions.TrialPruned as e:
        print(f"Trial {trial.number} pruned.")
        raise e
    except Exception:
        print(f"Trial {trial.number} failed with a detailed traceback:")
        traceback.print_exc()
        return 0.0  # Report failure to Optuna
    finally:
        # Restore original configuration for subsequent trials
        config.MODEL_CONFIG.clear()
        config.MODEL_CONFIG.update(original_model_config)
        config.BATCH_SIZE = original_batch_size
        config.LEARNING_RATE = original_learning_rate
        config.DEVICES = original_devices

    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for torch-tomo-slab")
    parser.add_argument("--gpu", type=int, help="GPU device ID to use for this optimization process.")
    parser.add_argument("--study-name", type=str, default="torch-tomo-slab-tuning", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_tuning.db", help="Optuna storage URL.")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials to run.")

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Process started, visible GPU: {torch.cuda.get_device_name(0)}")

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        pruner=pruner,
    )

    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=86400)  # 24-hour timeout
    except KeyboardInterrupt:
        print("Optimization stopped by user.")

    # This final summary should ideally be run in a separate process after all workers are done
    # or just by one of the workers.
    if not args.gpu or args.gpu == 0:
        print(f"\nStudy statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")

        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print(f"  Number of pruned trials: {len(pruned_trials)}")
        print(f"  Number of complete trials: {len(complete_trials)}")

        print("\nBest trial:")
        try:
            trial = study.best_trial
            print(f"  Value: {trial.value}")
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
        except ValueError:
            print("  No best trial found. All trials may have failed.")


        # Save results
        df = study.trials_dataframe()
        df.to_csv("optuna_results.csv", index=False)
        print("\nOptuna results saved to optuna_results.csv")
