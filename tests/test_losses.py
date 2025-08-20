# tests/test_losses.py
import pytest
import torch
from torch_tomo_slab.losses import get_loss_function, CombinedLoss, WeightedBCELoss


def test_get_single_loss():
    """Test retrieval of individual, known loss functions."""
    loss_config = {'name': 'weighted_bce'}
    loss_fn = get_loss_function(loss_config)
    assert isinstance(loss_fn, WeightedBCELoss)


def test_get_combined_loss():
    """Test retrieval of a combined loss function."""
    loss_config = {'name': 'dice+bce', 'weights': [0.5, 0.5]}
    loss_fn = get_loss_function(loss_config)
    assert isinstance(loss_fn, CombinedLoss)
    assert "dice" in loss_fn.weights
    assert "bce" in loss_fn.weights


def test_get_invalid_loss():
    """Test that an unknown loss name raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown loss function: invalid_loss"):
        get_loss_function({'name': 'invalid_loss'})


def test_combined_loss_missing_weights():
    """Test that a combined loss without weights raises a ValueError."""
    with pytest.raises(ValueError, match="Loss weights must be provided"):
        get_loss_function({'name': 'dice+bce'})


def test_weighted_bce_forward_pass():
    """Test the forward pass of WeightedBCELoss."""
    loss_fn = WeightedBCELoss()
    pred = torch.randn(4, 1, 32, 32)
    target = torch.randint(0, 2, (4, 1, 32, 32)).float()
    weights = torch.ones_like(target) * 5  # High weights everywhere

    loss = loss_fn(pred, target, weights)
    assert loss.ndim == 0  # Should be a scalar
    assert loss.item() > 0