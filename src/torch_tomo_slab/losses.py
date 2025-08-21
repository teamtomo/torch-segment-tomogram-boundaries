"""Loss functions for tomographic boundary segmentation.

This module provides specialized loss functions designed for boundary detection
in 3D tomographic data, including weighted losses, boundary-focused losses,
and flexible loss combination utilities.
"""
from typing import Dict, Tuple, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class SMPLossWrapper(nn.Module):
    """
    Wrapper for segmentation-models-pytorch losses to match API signature.
    
    This wrapper ensures that SMP losses can be used with the consistent
    3-parameter signature (pred_logits, target, weight_map) expected by
    the training pipeline.
    
    Parameters
    ----------
    loss : nn.Module
        The SMP loss function to wrap.
    from_logits : bool
        Whether the loss expects logits (True) or probabilities (False).
    """

    def __init__(self, loss: nn.Module, from_logits: bool):
        super().__init__()
        self.loss = loss
        self.from_logits = from_logits
        self.name = getattr(loss, '__name__', loss.__class__.__name__)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.from_logits:
            return self.loss(pred_logits, target)
        else:
            return self.loss(torch.sigmoid(pred_logits), target)


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss with pixel-wise weighting.
    
    This loss applies different weights to different pixels based on a
    pre-computed weight map, allowing emphasis on important regions
    (e.g., boundaries) while de-emphasizing others.
    
    Parameters
    ----------
    from_logits : bool, default=True
        Whether input predictions are logits (True) or probabilities (False).
    """

    def __init__(self, from_logits: bool = True):
        super().__init__()
        self.from_logits = from_logits
        self.name = "WeightedBCE"

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(pred_logits, target.float(), reduction='none')
        else:
            loss = F.binary_cross_entropy(torch.sigmoid(pred_logits), target.float(), reduction='none')

        if weight_map is not None:
            weighted_loss = loss * weight_map
            return weighted_loss.mean()
        else:
            return loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-focused loss that emphasizes boundary pixels over regions.
    
    This loss automatically detects boundary pixels using morphological operations
    and applies higher weight to boundary regions while maintaining coverage
    of non-boundary areas.
    
    Parameters
    ----------
    from_logits : bool, default=True
        Whether input predictions are logits (True) or probabilities (False).
    boundary_weight_factor : float, default=10.0
        Multiplier for boundary pixel loss relative to region pixels.
    """

    def __init__(self, from_logits: bool = True, boundary_weight_factor: float = 10.0):
        super().__init__()
        self.from_logits = from_logits
        self.boundary_weight_factor = boundary_weight_factor
        self.name = "BoundaryLoss"

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.from_logits:
            loss_pixels = F.binary_cross_entropy_with_logits(pred_logits, target.float(), reduction='none')
        else:
            loss_pixels = F.binary_cross_entropy(torch.sigmoid(pred_logits), target.float(), reduction='none')
        max_pool = F.max_pool2d(target.float(), kernel_size=3, stride=1, padding=1)
        min_pool = -F.max_pool2d(-target.float(), kernel_size=3, stride=1, padding=1)
        boundary_mask = (max_pool - min_pool).bool()
        non_boundary_mask = ~boundary_mask
        loss_boundary = (loss_pixels * boundary_mask.float()).sum()
        num_boundary = boundary_mask.sum().clamp(min=1.0)
        mean_loss_boundary = loss_boundary / num_boundary
        loss_region = (loss_pixels * non_boundary_mask.float()).sum()
        num_region = non_boundary_mask.sum().clamp(min=1.0)
        mean_loss_region = loss_region / num_region
        return self.boundary_weight_factor * mean_loss_boundary + mean_loss_region


class CombinedLoss(nn.Module):
    """
    Flexible multi-component loss function with weighted combination.
    
    This loss allows combining multiple loss functions (e.g., Dice + BCE)
    with custom weights to leverage benefits of different loss types.
    
    Parameters
    ----------
    losses_dict : Dict[str, Tuple[nn.Module, float]]
        Dictionary mapping loss names to (loss_function, weight) tuples.
    from_logits : bool, default=True
        Whether input predictions are logits (True) or probabilities (False).
    """

    def __init__(self, losses_dict: Dict[str, Tuple[nn.Module, float]], from_logits: bool = True):
        super().__init__()
        self.losses = nn.ModuleDict({name: loss for name, (loss, weight) in losses_dict.items()})
        self.weights = {name: weight for name, (loss, weight) in losses_dict.items()}
        self.from_logits = from_logits
        self.name = "+".join(losses_dict.keys())

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=pred_logits.device)
        for name, loss_fn in self.losses.items():
            # Pass weight_map to all component losses, they will handle None appropriately
            component_loss = loss_fn(pred_logits, target, weight_map)
            total_loss += self.weights[name] * component_loss
        return total_loss


def get_loss_function(loss_config: dict) -> nn.Module:
    """
    Factory function to create loss functions from configuration.
    
    Parameters
    ----------
    loss_config : dict
        Configuration dictionary with keys:
        - 'name': Loss function name or combination (e.g., 'dice+bce')
        - 'weights': List of weights for combined losses (if applicable)
    
    Returns
    -------
    nn.Module
        Configured loss function ready for training.
    
    Raises
    ------
    ValueError
        If unknown loss name or mismatched weights for combined losses.
        
    Examples
    --------
    >>> config = {'name': 'weighted_bce'}
    >>> loss_fn = get_loss_function(config)
    
    >>> config = {'name': 'dice+bce', 'weights': [0.5, 0.5]}
    >>> combined_loss = get_loss_function(config)
    """
    name = loss_config['name'].lower()
    from_logits = True  # All our model outputs are logits

    loss_lib = {
        'bce': nn.BCEWithLogitsLoss(),
        'weighted_bce': WeightedBCELoss(from_logits=from_logits),
        'boundary': BoundaryLoss(from_logits=from_logits),
        'dice': SMPLossWrapper(smp.losses.DiceLoss(mode='binary'), from_logits=from_logits),
    }

    if '+' not in name:
        if name in loss_lib:
            return loss_lib[name]
        raise ValueError(f"Unknown loss function: {name}")

    # Handle combined losses
    loss_components_names = name.split('+')
    loss_weights = loss_config.get('weights')
    if not loss_weights or len(loss_components_names) != len(loss_weights):
        raise ValueError(
            f"Loss weights must be provided for combined loss '{name}' and match the number of components.")

    losses_to_combine = {}
    for comp_name, weight in zip(loss_components_names, loss_weights):
        if comp_name in loss_lib:
            losses_to_combine[comp_name] = (loss_lib[comp_name], weight)
        else:
            raise ValueError(f"Unknown component '{comp_name}' in combined loss.")

    return CombinedLoss(losses_to_combine, from_logits=from_logits)
