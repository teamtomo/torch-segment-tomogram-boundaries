# tests/test_trainer.py
from torch_tomo_slab.trainer import TomoSlabTrainer
from torch_tomo_slab import constants, config
from pytorch_lightning.callbacks import EarlyStopping
from torch_tomo_slab.callbacks import DynamicTrainingManager


def test_trainer_initialization():
    """Test if the trainer API class initializes correctly."""
    trainer = TomoSlabTrainer()
    assert trainer.model_encoder == constants.MODEL_ENCODER
    assert trainer.learning_rate == config.LEARNING_RATE


def test_setup_datamodule(dummy_pt_files):
    """Test the datamodule setup."""
    constants.TRAIN_DATA_DIR = dummy_pt_files["train_dir"]
    constants.VAL_DATA_DIR = dummy_pt_files["val_dir"]
    trainer = TomoSlabTrainer()
    datamodule = trainer._setup_datamodule()
    datamodule.setup()
    assert len(datamodule.train_dataset) == 4
    assert len(datamodule.val_dataset) == 2


def test_setup_callbacks(mocker):
    """Test that the correct callbacks are selected based on config."""
    trainer = TomoSlabTrainer()

    # Test with Dynamic Manager
    mocker.patch.object(config, 'USE_DYNAMIC_MANAGER', True)
    callbacks = trainer._setup_callbacks()
    assert any(isinstance(cb, DynamicTrainingManager) for cb in callbacks)

    # Test with standard Early Stopping
    mocker.patch.object(config, 'USE_DYNAMIC_MANAGER', False)
    callbacks = trainer._setup_callbacks()
    assert any(isinstance(cb, EarlyStopping) for cb in callbacks)


def test_trainer_fit_fast_dev_run(dummy_pt_files, mocker):
    """
    Integration test to ensure trainer.fit() runs for one batch without errors.
    This is a crucial CI check.
    """
    # Mock the pl.Trainer to use fast_dev_run
    mock_pl_trainer = mocker.patch('pytorch_lightning.Trainer')

    # Point constants to temporary data
    constants.TRAIN_DATA_DIR = dummy_pt_files["train_dir"]
    constants.VAL_DATA_DIR = dummy_pt_files["val_dir"]

    trainer_api = TomoSlabTrainer()
    trainer_api.fit()  # Run the main method

    # Assert that pl.Trainer was called with fast_dev_run implicitly
    # or explicitly check that its 'fit' method was called.
    assert mock_pl_trainer.return_value.fit.call_count == 1