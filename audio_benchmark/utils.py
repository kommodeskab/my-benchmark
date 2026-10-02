"""General utility functions for the audio representation benchmark suite."""

from __future__ import annotations

import numpy as np
import torch


def to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor or array-like object to a NumPy array.

    Args:
        x: Input tensor or array-like data structure.

    Returns:
        np.ndarray: NumPy array representation of the input.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device specification to a concrete torch.device.

    If 'auto' or None, resolves to 'cuda' when available, otherwise 'cpu'.

    Args:
        device: Device string (e.g. 'auto', 'cpu', 'cuda') or torch.device instance.

    Returns:
        torch.device: Resolved PyTorch device object.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
