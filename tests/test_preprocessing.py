# tests/test_preprocessing.py
import torch
from torch_tomo_slab.processing import TrainingDataGenerator
from torch_tomo_slab import constants


def test_generator_initialization(data_dirs):
    """Test if the TrainingDataGenerator initializes correctly."""
    gen = TrainingDataGenerator(
        volume_dir=data_dirs["vol_dir"],
        mask_dir=data_dirs["mask_dir"],
        output_train_dir=data_dirs["train_dir"],
        output_val_dir=data_dirs["val_dir"],
    )
    assert gen.volume_dir == data_dirs["vol_dir"]
    assert gen.mask_dir == data_dirs["mask_dir"]


def test_find_data_pairs(data_dirs, dummy_mrc_files):
    """Test the discovery of corresponding volume-mask pairs."""
    gen = TrainingDataGenerator(
        volume_dir=data_dirs["vol_dir"], mask_dir=data_dirs["mask_dir"]
    )
    pairs = gen._find_data_pairs()
    assert len(pairs) == 1
    assert pairs[0][0].name == "dummy_tomo.mrc"
    assert pairs[0][1].name == "dummy_tomo.mrc"


def test_resize_and_pad(data_dirs):
    """Test the 3D resizing and padding logic."""
    gen = TrainingDataGenerator(target_volume_shape=(32, 64, 64))

    # Test up-scaling (padding)
    input_tensor = torch.randn(16, 32, 32)
    resized = gen._resize_and_pad_3d(input_tensor, mode='image')
    assert resized.shape == (32, 64, 64)

    # Test down-scaling
    input_tensor = torch.randn(48, 96, 96)
    resized = gen._resize_and_pad_3d(input_tensor, mode='image')
    assert resized.shape == (32, 64, 64)


def test_generator_run(data_dirs, dummy_mrc_files):
    """Integration test for the TrainingDataGenerator.run() method."""
    # Use a smaller target shape for faster testing
    target_shape = (16, 32, 32)
    gen = TrainingDataGenerator(
        volume_dir=data_dirs["vol_dir"],
        mask_dir=data_dirs["mask_dir"],
        output_train_dir=data_dirs["train_dir"],
        output_val_dir=data_dirs["val_dir"],
        validation_fraction=0.5,
        target_volume_shape=target_shape
    )

    gen.run()

    # --- CORRECTED LOGIC ---
    # Check that output files were created in either train or val directory.
    train_files = list(data_dirs["train_dir"].glob("*.pt"))
    val_files = list(data_dirs["val_dir"].glob("*.pt"))
    total_files = len(train_files) + len(val_files)

    assert total_files > 0

    # Check that the number of generated files matches the constant
    assert total_files == constants.NUM_SECTIONS_PER_VOLUME

    # Inspect a generated sample from whichever directory has files
    output_files = train_files if train_files else val_files
    sample = torch.load(output_files[0])
    assert "image" in sample
    assert "label" in sample
    assert sample["image"].shape[0] == 2  # 2 channels
    # The label shape is (H, W), image is (C, H, W)
    assert sample["image"].shape[1:] == sample["label"].shape