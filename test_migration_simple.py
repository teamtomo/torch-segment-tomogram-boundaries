#!/usr/bin/env python3
"""
Simplified migration test for MONAI implementation.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from torch_tomo_slab.models import create_unet
from torch_tomo_slab.losses import get_loss_function
from torch_tomo_slab import config

def test_simplified_model():
    """Test simplified MONAI model creation."""
    print("Testing simplified MONAI model creation...")

    model = create_unet(**config.MODEL_CONFIG)
    print(f"✓ Model created: {type(model)}")

    # Test forward pass
    input_tensor = torch.randn(2, 1, 256, 256)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Forward pass successful")

    return model

def test_loss_functions():
    """Test simplified loss functions."""
    print("\nTesting loss functions...")

    # Test WeightedBCE (critical for the task)
    weighted_bce = get_loss_function({'name': 'weighted_bce'})
    print(f"✓ WeightedBCE loss: {type(weighted_bce)}")

    # Test dice loss
    dice_loss = get_loss_function({'name': 'dice'})
    print(f"✓ Dice loss: {type(dice_loss)}")

    # Test combined loss
    combined_loss = get_loss_function({'name': 'dice+weighted_bce', 'weights': [0.5, 0.5]})
    print(f"✓ Combined loss: {type(combined_loss)}")

    return weighted_bce

def test_end_to_end(model, loss_fn):
    """Test end-to-end pipeline."""
    print("\nTesting end-to-end pipeline...")

    # Synthetic data
    input_tensor = torch.randn(2, 1, 256, 256)
    target_tensor = torch.randint(0, 2, (2, 1, 256, 256)).float()
    weight_map = torch.ones_like(target_tensor) * 2.5  # Boundary weighting

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # Loss computation
    loss_value = loss_fn(output, target_tensor, weight_map)

    print(f"Loss value: {loss_value.item():.4f}")
    print("✓ End-to-end pipeline successful")

def main():
    print("=== Simplified MONAI Migration Test ===\n")

    try:
        # Test model
        model = test_simplified_model()

        # Test losses
        loss_fn = test_loss_functions()

        # Test end-to-end
        test_end_to_end(model, loss_fn)

        print("\n=== ALL TESTS PASSED ===")
        print("✓ Simplified MONAI migration successful!")
        print("✓ Model creation and inference working")
        print("✓ WeightedBCE loss preserved and functional")

    except Exception as e:
        print(f"\n=== TEST FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()