"""Common utility functions for the project."""
import logging

import torch

log = logging.getLogger(__name__)


def get_device(verbose: bool = False) -> torch.device:
    """
    Get the appropriate PyTorch device for computation.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU device.
    """
    if torch.cuda.is_available():
        if verbose:
            log.info("CUDA is available. Using GPU.")
        return torch.device("cuda")

    if verbose:
        log.info("CUDA not available. Using CPU.")
    return torch.device("cpu")